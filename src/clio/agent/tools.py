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


def build_tools() -> list:
    """Return the list of tools available to the agent."""
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    collection = os.getenv("CHROMADB_COLLECTION", "domain_knowledge")

    vectorstore = Chroma(
        persist_directory=os.getenv("CHROMADB_PATH"),
        embedding_function=embeddings,
        collection_name=collection,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    doc_prompt = PromptTemplate.from_template(
        "--- DOCUMENT CHUNK ---\n"
        "SOURCE: {source_title}\n"
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
