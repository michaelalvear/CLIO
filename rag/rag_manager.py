"""
CLIO RAG Manager
Console interface for managing the Chroma vector knowledge base.

Scans a directory (DOCS_PATH) for .pdf and .txt files and embeds them into
a named Chroma collection.  Runs in a loop until the user types 'exit'.

Required env vars:
  CHROMADB_PATH       path to the persistent Chroma database directory
  DOCS_PATH           directory containing .pdf and/or .txt source files
                      (falls back to PDF_PATH for backward compatibility)
  CHROMADB_COLLECTION default collection name (used when none is specified)
  GOOGLE_API_KEY      required by the Gemini embedding model

Author: Michael Alvear
"""

import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import List

from dotenv import load_dotenv
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions import GoogleGenaiEmbeddingFunction
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE    = 90
EMBED_MODEL   = "gemini-embedding-001"


# ── Initialisation helpers ─────────────────────────────────────────────────

def _get_client() -> ClientAPI:
    path = os.getenv("CHROMADB_PATH")
    if not path:
        raise EnvironmentError("CHROMADB_PATH is not set.")
    return chromadb.PersistentClient(path=path)


def _get_embedding_function() -> EmbeddingFunction:
    return GoogleGenaiEmbeddingFunction(model_name=EMBED_MODEL)


def _get_docs_dir() -> Path:
    raw = os.getenv("DOCS_PATH")
    if not raw:
        raise EnvironmentError(
            "DOCS_PATH is not set. Point it to a directory of .pdf/.txt files."
        )
    p = Path(raw)
    if p.is_file():
        # backward-compat: if a single file was given, use its parent directory
        return p.parent
    if not p.is_dir():
        raise FileNotFoundError(f"DOCS_PATH does not exist: {raw}")
    return p


def _default_collection() -> str:
    return os.getenv("CHROMADB_COLLECTION", "domain_knowledge")


def _resolve_collection(args: list, client: ClientAPI, prompt: str = "Collection") -> str:
    """
    Return a collection name from, in priority order:
      1. An explicit non-numeric first argument (e.g. `preview mycol 5`)
      2. The CHROMADB_COLLECTION env var
      3. The only existing collection (if there is exactly one)
      4. An interactive prompt listing available collections
    A numeric first argument is never treated as a collection name, so
    `preview 5` means n=5 against the default collection.
    """
    if args and not args[0].isdigit():
        return args[0]

    default = os.getenv("CHROMADB_COLLECTION")
    if default:
        return default

    existing = client.list_collections()
    if len(existing) == 1:
        return existing[0].name

    names = [c.name for c in existing]
    print(f"Available collections: {', '.join(names)}")
    return input(f"{prompt}: ").strip()


def _resolve_n(args: list, default: int = 3) -> int:
    """Return the numeric argument from args, ignoring any leading collection name."""
    for a in args:
        if a.isdigit():
            return int(a)
    return default


# ── File loading ───────────────────────────────────────────────────────────

def _discover_files(docs_dir: Path) -> List[Path]:
    files = []
    for ext in ("*.pdf", "*.txt"):
        files.extend(sorted(docs_dir.glob(ext)))
    return files


def _extract_title(file_path: Path) -> str:
    """
    Best-effort title extraction.
    PDFs: read the /Title field from the document info dictionary.
    TXTs: use the first non-empty line.
    Falls back to a cleaned-up version of the filename in both cases.
    """
    fallback = file_path.stem.replace("_", " ").replace("-", " ").title()
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            info = PdfReader(str(file_path)).metadata
            title = (info.get("/Title") or "").strip() if info else ""
            return title if title else fallback
        except Exception:
            return fallback

    if ext == ".txt":
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        return line
        except Exception:
            pass
        return fallback

    return fallback


def _extract_author(file_path: Path) -> str:
    """Best-effort /Author extraction from PDF metadata. Returns '' if
    unavailable -- caller decides the placeholder shown to the user."""
    if file_path.suffix.lower() != ".pdf":
        return ""
    try:
        from pypdf import PdfReader
        info = PdfReader(str(file_path)).metadata
        return (info.get("/Author") or "").strip() if info else ""
    except Exception:
        return ""


_DOI_RE = re.compile(r"(?:doi\.org/|doi:\s*)?(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)


def _extract_doi(file_path: Path) -> str:
    """
    Best-effort DOI extraction: checks the /Subject metadata field first
    (publisher-set metadata, most reliable), then scans the first several
    pages of text -- wide enough to reach a "suggested citation" block,
    which is where report-series publishers (e.g. USGS) often place the DOI
    a few pages into the front matter rather than on the title page itself.

    The match isn't guaranteed clean -- PDF text extraction can glue a DOI
    directly onto the following word or trailing punctuation with no space.
    That's fine: the suggested value is always shown to the user for
    confirmation during review, so an imprecise match gets caught and fixed
    there rather than needing a perfect regex here.
    """
    if file_path.suffix.lower() != ".pdf":
        return ""
    try:
        from pypdf import PdfReader
        reader  = PdfReader(str(file_path))
        subject = (reader.metadata.get("/Subject") or "") if reader.metadata else ""
        match   = _DOI_RE.search(subject)
        if match:
            return match.group(1).rstrip(".,;:")

        text = ""
        for page in reader.pages[:8]:
            text += page.extract_text() or ""
        match = _DOI_RE.search(text)
        return match.group(1).rstrip(".,;:") if match else ""
    except Exception:
        return ""


# ── Metadata review ───────────────────────────────────────────────────────

def _metadata_path(docs_dir: Path) -> Path:
    return docs_dir / "metadata.json"


def _load_metadata_store(docs_dir: Path) -> dict:
    path = _metadata_path(docs_dir)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save_metadata_store(docs_dir: Path, store: dict) -> None:
    with open(_metadata_path(docs_dir), "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2, sort_keys=True)


def _review_metadata(file_path: Path, store: dict) -> dict:
    """
    Show this file's currently known title/author/DOI -- from a previous
    review if present, otherwise freshly extracted -- and let the user
    accept each field with Enter or type a replacement. Always shown, even
    for previously-reviewed files, so a bad auto-filled value (or a bad
    earlier human answer) can be caught and corrected on any run, not just
    the first.
    """
    key    = file_path.name
    cached = store.get(key, {})
    label  = "previously reviewed" if key in store else "new"

    # A cached "unknown"/"none" isn't trusted as a real answer -- it might
    # just be a placeholder a prior run couldn't fill in (e.g. before an
    # extraction fix), so re-attempt extraction rather than getting stuck
    # repeating it forever. A cached real value is trusted and used as-is.
    cached_author = cached.get("author")
    cached_doi    = cached.get("doi")

    title_default  = cached.get("title") or _extract_title(file_path)
    author_default = (
        cached_author if cached_author and cached_author != "unknown" else None
    ) or _extract_author(file_path) or "unknown"
    doi_default = (
        cached_doi if cached_doi and cached_doi != "none" else None
    ) or _extract_doi(file_path) or "none"

    print(f"\n── {file_path.name} ({label}) ──")
    title  = input(f"  Title  [{title_default}]: ").strip()  or title_default
    author = input(f"  Author [{author_default}]: ").strip() or author_default
    doi    = input(f"  DOI    [{doi_default}]: ").strip()    or doi_default

    record = {"title": title, "author": author, "doi": doi}
    store[key] = record
    return record


def _load_file(file_path: Path, metadata: dict):
    """Return a list of LangChain Document objects from a .pdf or .txt file,
    stamped with the given reviewed metadata (title/author/doi) on every
    chunk."""
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(str(file_path))
        docs   = loader.load()
        for doc in docs:
            if "page_label" not in doc.metadata:
                doc.metadata["page_label"] = str(doc.metadata.get("page", "?"))
    elif ext == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
        docs   = loader.load()
        for doc in docs:
            doc.metadata["page_label"] = "n/a"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    for doc in docs:
        doc.metadata["source_title"]  = metadata["title"]
        doc.metadata["source_author"] = metadata["author"]
        doc.metadata["source_doi"]    = metadata["doi"]

    return docs


# ── Commands ───────────────────────────────────────────────────────────────

def cmd_add(
    args: list,
    client: ClientAPI,
    ef: EmbeddingFunction,
    docs_dir: Path,
) -> None:
    """add [collection] — review metadata, then embed all docs in DOCS_PATH into a collection."""
    collection_name = _resolve_collection(args, client, prompt="Target collection")

    files = _discover_files(docs_dir)
    if not files:
        print(f"No .pdf or .txt files found in {docs_dir}")
        return

    print(f"\nFound {len(files)} file(s) in {docs_dir}:")
    for f in files:
        size_kb = f.stat().st_size // 1024
        print(f"  [{f.suffix.upper()[1:]:3}]  {f.name}  ({size_kb} KB)")

    store = _load_metadata_store(docs_dir)
    print("\nReview metadata for each file (Enter to accept, or type a replacement):")
    print("Check each value against the source document before accepting -- an "
          "auto-extracted title/author/DOI can look like a plausible real value "
          "and still be wrong (e.g. a generic PDF metadata label instead of the "
          "document's actual title). Don't just press Enter through all of them.")
    reviewed = {}
    for file_path in files:
        reviewed[file_path.name] = _review_metadata(file_path, store)
        _save_metadata_store(docs_dir, store)
    print(f"\nMetadata saved to {_metadata_path(docs_dir)}.")

    confirm = input(f"\nEmbed all into '{collection_name}'? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    splitter   = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=ef
    )

    total = 0
    for file_path in files:
        try:
            print(f"\n  Loading   {file_path.name} ...", end=" ", flush=True)
            raw_docs = _load_file(file_path, reviewed[file_path.name])
            chunks   = splitter.split_documents(raw_docs)
            ids      = [str(uuid.uuid4()) for _ in chunks]
            texts    = [c.page_content for c in chunks]
            metas    = [c.metadata     for c in chunks]

            print(f"{len(chunks)} chunks", end=" — ", flush=True)
            print("embedding ...", end=" ", flush=True)

            for i in range(0, len(ids), BATCH_SIZE):
                collection.add(
                    ids=ids[i : i + BATCH_SIZE],
                    documents=texts[i : i + BATCH_SIZE],
                    metadatas=metas[i : i + BATCH_SIZE],
                )

            print("done.")
            total += len(chunks)

        except Exception as e:
            print(f"\n  ERROR embedding {file_path.name}: {e}")

    print(f"\nAdded {total} chunks to '{collection_name}'.")
    print(f"Collection '{collection_name}' now contains {collection.count()} chunks total.")


def cmd_list(client: ClientAPI) -> None:
    """list — show all collections and their chunk counts."""
    collections = client.list_collections()
    if not collections:
        print("No collections found.")
        return
    print(f"\n  {'Collection':<30} {'Chunks':>8}")
    print("  " + "-" * 40)
    for col in collections:
        count = client.get_collection(col.name).count()
        print(f"  {col.name:<30} {count:>8}")


def cmd_preview(args: list, client: ClientAPI) -> None:
    """preview [n] — show n sample chunks from the default collection (default n=3)."""
    if not client.list_collections():
        print("No collections found.")
        return

    collection_name = _resolve_collection(args, client, prompt="Collection to preview")
    n               = _resolve_n(args)

    try:
        col = client.get_collection(collection_name)
    except Exception:
        print(f"Collection '{collection_name}' not found.")
        return

    samples = col.get(limit=n)
    records = list(zip(samples["ids"], samples["documents"], samples["metadatas"]))

    print(f"\n── {collection_name}  ({col.count()} chunks total) "
          f"— showing {len(records)} sample(s) ──")

    for chunk_id, text, meta in records:
        title  = meta.get("source_title", Path(meta.get("source", "?")).name)
        author = meta.get("source_author", "unknown")
        doi    = meta.get("source_doi", "none")
        page   = meta.get("page_label", meta.get("page", "?"))
        print(f"\n  ID     : {chunk_id}")
        print(f"  Title  : {title}")
        print(f"  Author : {author}")
        print(f"  DOI    : {doi}")
        print(f"  Page   : {page}")
        print(f"  Text   : {text[:300]}{'...' if len(text) > 300 else ''}")
        print("  " + "─" * 60)


def cmd_query(args: list, client: ClientAPI, ef: EmbeddingFunction) -> None:
    """query [n] — similarity search against the default collection (default n=3)."""
    if not client.list_collections():
        print("No collections found.")
        return

    collection_name = _resolve_collection(args, client, prompt="Collection to query")
    n               = _resolve_n(args)

    try:
        col = client.get_collection(collection_name, embedding_function=ef)
    except Exception:
        print(f"Collection '{collection_name}' not found.")
        return

    query_text = input("Query: ").strip()
    if not query_text:
        print("No query entered.")
        return

    results = col.query(query_texts=[query_text], n_results=n)

    ids       = results["ids"][0]
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"\n── '{query_text}'")
    print(f"── {collection_name}  —  top {len(ids)} result(s) ──")

    for rank, (chunk_id, text, meta, dist) in enumerate(
        zip(ids, docs, metas, distances), start=1
    ):
        title  = meta.get("source_title", Path(meta.get("source", "?")).name)
        author = meta.get("source_author", "unknown")
        doi    = meta.get("source_doi", "none")
        page   = meta.get("page_label", meta.get("page", "?"))
        print(f"\n  Rank     : {rank}  (distance: {dist:.4f})")
        print(f"  Title    : {title}")
        print(f"  Author   : {author}")
        print(f"  DOI      : {doi}")
        print(f"  Page     : {page}")
        print(f"  Text     : {text[:400]}{'...' if len(text) > 400 else ''}")
        print("  " + "─" * 60)


def cmd_delete(args: list, client: ClientAPI) -> None:
    """delete [all] — delete the default collection, or all collections."""
    collections = client.list_collections()
    if not collections:
        print("No collections to delete.")
        return

    if args and args[0] == "all":
        to_delete = [c.name for c in collections]
    else:
        to_delete = [_resolve_collection(args, client, prompt="Collection to delete")]

    confirm = input(
        f"Permanently delete {to_delete}? This cannot be undone. [y/N]: "
    ).strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    for name in to_delete:
        try:
            client.delete_collection(name=name)
            print(f"  Deleted '{name}'.")
        except Exception as e:
            print(f"  Could not delete '{name}': {e}")


def _print_help() -> None:
    print("""
  Commands:
    add     [collection]   embed all docs in DOCS_PATH (defaults to CHROMADB_COLLECTION)
    list                   list collections with chunk counts
    preview [n]            show n sample chunks from the default collection  (default n=3)
    query   [n]            similarity search against the default collection  (default n=3)
    delete  [all]          delete the default collection, or all with 'all'
    help                   show this message
    exit                   quit

  Collection is resolved automatically from CHROMADB_COLLECTION env var.
  Pass an explicit name as the first argument to any command to override it.
""")


# ── Main loop ──────────────────────────────────────────────────────────────

def main() -> None:
    try:
        client   = _get_client()
        ef       = _get_embedding_function()
        docs_dir = _get_docs_dir()
    except (EnvironmentError, FileNotFoundError) as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    # ── Header ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  CLIO RAG Manager")
    print("=" * 50)
    print(f"  Chroma DB   : {os.getenv('CHROMADB_PATH')}")
    print(f"  Docs dir    : {docs_dir}")
    print(f"  Embed model : {EMBED_MODEL}")

    collections = client.list_collections()
    if collections:
        summary = ", ".join(
            f"{c.name} ({client.get_collection(c.name).count()})"
            for c in collections
        )
        print(f"  Collections : {summary}")
    else:
        print("  Collections : (none)")

    _print_help()

    # ── REPL ─────────────────────────────────────────────────────────────────
    while True:
        try:
            raw = input("rag> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        parts, command, args = raw.split(), "", []
        command = parts[0].lower()
        args    = parts[1:]

        if command in ("exit", "quit"):
            print("Goodbye.")
            break
        elif command == "add":
            cmd_add(args, client, ef, docs_dir)
        elif command == "list":
            cmd_list(client)
        elif command == "preview":
            cmd_preview(args, client)
        elif command == "query":
            cmd_query(args, client, ef)
        elif command == "delete":
            cmd_delete(args, client)
        elif command == "help":
            _print_help()
        else:
            print(f"  Unknown command '{command}'. Type 'help' for options.")


if __name__ == "__main__":
    main()
