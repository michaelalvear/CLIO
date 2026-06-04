"""Tools provided to the agent. The agent is now an interpreter, so the
only tool it needs is RAG access to the BISECT paper."""

import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.tools import create_retriever_tool
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def build_tools() -> list:
    """Return the list of tools available to the agent."""
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    vectorstore = Chroma(
        persist_directory=os.getenv("CHROMADB_PATH"),
        embedding_function=embeddings,
        collection_name="BISECT",
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    doc_prompt = PromptTemplate.from_template(
        "--- DOCUMENT CHUNK ---\n"
        "SOURCE PAGE: {page_label}\n"
        "CONTENT: {page_content}\n"
    )

    bisect_context_retriever = create_retriever_tool(
        retriever=retriever,
        name="bisect_context_retriever",
        description="Search and return relevant portions of the BISECT paper.",
        document_prompt=doc_prompt,
        response_format="content",
    )

    return [bisect_context_retriever]
