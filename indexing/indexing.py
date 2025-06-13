import asyncio
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document
from tools.contants import (
    INDEX_NAME,
    MODEL_EMBEDDING,
    DOCS_LOCATION,
    BULK_SIZE,
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
)

load_dotenv()

# Custom mapping configuration
CUSTOM_MAPPING = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "text": {"type": "text"},
            "vector_field": {
                "type": "knn_vector",
                "dimension": 1536,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                },
            },
        }
    },
}

OPENSEARCH_URL = f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}"


async def run_in_executor(func):
    """Execute synchronous functions in an executor"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func)


async def process_in_batches(documents, embeddings, batch_size=BULK_SIZE):
    """Process documents in batches to prevent memory overload"""
    total_docs = len(documents)
    for i in range(0, total_docs, batch_size):
        batch = documents[i : i + batch_size]
        print(
            f"Processing batch {i // batch_size + 1}/{(total_docs // batch_size) + 1} "
            f"({len(batch)} documents)"
        )

        def sync_index_batch():
            return OpenSearchVectorSearch.from_documents(
                documents=batch,
                embedding=embeddings,
                opensearch_url=OPENSEARCH_URL,
                index_name=INDEX_NAME,
                custom_mapping=CUSTOM_MAPPING,
                bulk_size=len(batch),
                verify_certs=False,
            )

        await run_in_executor(sync_index_batch)


async def indexing_docs():
    """Main function to load, process and index documents into OpenSearch"""

    # Load all markdown files
    loader = DirectoryLoader(
        path=DOCS_LOCATION,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        loader_kwargs={"mode": "single"},
    )

    # Assign title to each document
    raw_documents = loader.load()
    for doc in raw_documents:
        if isinstance(doc, Document):
            source_path = doc.metadata.get("source", "")
            filename = Path(source_path).name
            title = filename.replace(".md", "")
            doc.metadata["title"] = title
        else:
            print("Loaded item is not a Document instance.")

    print(f"Loaded {len(raw_documents)} documents")

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(raw_documents)

    print(f"Text splitter resulted in {len(documents)} documents")

    # Create embeddings
    embeddings = OpenAIEmbeddings(model=MODEL_EMBEDDING)

    # Index documents in batches
    print(f"Starting bulk indexing in batches of {BULK_SIZE} documents...")
    await process_in_batches(documents, embeddings)
    print(f"Successfully indexed {len(documents)} documents into {INDEX_NAME}")


if __name__ == "__main__":
    asyncio.run(indexing_docs())
