"""Tools provided to the agent.
The only tool is a RAG retriever over a domain knowledge base whose
collection name is configured via the CHROMADB_COLLECTION env var.
"""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.tools import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def get_vectorstore() -> Chroma:
    """
    Build the Chroma vectorstore connection. Shared by build_tools() (for
    retrieval) and knowledge_base_prompt_block() (for listing source titles)
    so both use the same connection instead of opening two.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    collection = os.getenv("CHROMADB_COLLECTION", "domain_knowledge")

    return Chroma(
        persist_directory=os.getenv("CHROMADB_PATH"),
        embedding_function=embeddings,
        collection_name=collection,
    )


def build_tools(vectorstore: Chroma | None = None) -> list:
    """Return the list of tools available to the agent."""
    vectorstore = vectorstore or get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8},
    )

    doc_prompt = PromptTemplate.from_template(
        "--- DOCUMENT CHUNK ---\n"
        "SOURCE: {source_title}\n"
        "AUTHOR: {source_author}\n"
        "DOI: {source_doi}\n"
        "PAGE: {page_label}\n"
        "CONTENT: {page_content}\n"
    )

    retrieve_domain_context = create_retriever_tool(
        retriever=retriever,
        name="retrieve_domain_context",
        description=(
            "Search and return relevant context from the domain knowledge base. "
            "Use this for questions about model methodology, scientific background, "
            "or explanation of dataset assumptions."
        ),
        document_prompt=doc_prompt,
        response_format="content",
    )

    return [retrieve_domain_context]


def knowledge_base_prompt_block(vectorstore: Chroma | None = None) -> str:
    """
    List the distinct source document titles in the knowledge base, for
    injection into the agent system prompt.

    Titles only -- never chunk content or summaries -- so the agent can scope
    what it can retrieve without being tempted to answer from title
    recognition instead of actually calling retrieve_domain_context.
    """
    vectorstore = vectorstore or get_vectorstore()
    metadatas   = vectorstore.get()["metadatas"]
    titles      = sorted({m["source_title"] for m in metadatas if m.get("source_title")})

    if not titles:
        return "KNOWLEDGE BASE: no documents currently loaded."

    lines = ["KNOWLEDGE BASE -- documents available via retrieve_domain_context:"]
    lines.extend(f"  - {t}" for t in titles)
    return "\n".join(lines)
