from dotenv import load_dotenv
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()

from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import List, Dict, Any, Optional
from tools.contants import (
    INDEX_NAME,
    MODEL_EMBEDDING,
    MODEL_LLM,
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
)
import mlflow

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
mlflow.langchain.autolog()


def run_llm(
    query: str, chat_history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Executes a conversational question-answering pipeline using LangChain and OpenSearch.

    This function:
    1. Connects to an OpenSearch vector store
    2. Uses a chat model to understand the query in context of conversation history
    3. Retrieves relevant documents
    4. Generates a well-formed answer with sources

    Args:
        query: The user's question or input string
        chat_history: List of previous conversation turns in the format:
                     [{"human": "user input", "ai": "bot response"}, ...]

    Returns:
        A dictionary containing:
        {
            "query": The processed query (after history-aware rephrasing),
            "result": The generated answer,
            "source_documents": Relevant documents used for answering
        }
    """
    chat_history = chat_history or []

    OPENSEARCH_URL = f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}"

    # Initialize components docsearch
    embeddings = OpenAIEmbeddings(model=MODEL_EMBEDDING)
    docsearch = OpenSearchVectorSearch(
        index_name=INDEX_NAME,
        embedding_function=embeddings,
        opensearch_url=OPENSEARCH_URL,
    )

    # Configure chat model
    chat = ChatOpenAI(model=MODEL_LLM, temperature=0)

    # Load prompts from hub
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    # Create processing chain
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    history_aware_retriever = create_history_aware_retriever(
        llm=chat,
        retriever=docsearch.as_retriever(search_kwargs={"k": 4}),
        prompt=rephrase_prompt,
    )

    qa_chain = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    # Execute the pipeline
    result = qa_chain.invoke({"input": query, "chat_history": chat_history})

    # Format consistent output
    return {
        "query": result.get("input", query),
        "result": result["answer"],
        "source_documents": result.get("context", []),
        "search_terms": result.get("search_terms", ""),
    }


if __name__ == "__main__":
    try:
        res = run_llm(query="What is Sagemaker?")
        print("Answer:", res["result"])
    except Exception as e:
        print(f"Error processing query: {str(e)}")
