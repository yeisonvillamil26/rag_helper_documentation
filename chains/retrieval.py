from dotenv import load_dotenv
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from chains.scratch_evaluator import RagEvaluator

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
import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
mlflow.langchain.autolog()


def run_llm(
    query: str, chat_history: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Handles conversational queries against documentation using a RAG pipeline.

    Combines OpenSearch retrieval with LLM conversation to provide
    answers grounded in documentation while maintaining chat context.

    Steps:
    1. Rewrites the query considering chat history
    2. Finds relevant docs in OpenSearch
    3. Synthesizes a response

    Args:
        query: Current user question
        chat_history: Previous message pairs

    Returns:
        {
            "query": Final query used (after rephrasing),
            "result": Generated answer,
            "source_documents": Retrieved documents path,
            "search_terms": Actual terms used for search
        }

    NOTE: Additional, here is an implementation of an evaluator to track LLM quality
    """
    chat_history = chat_history or []

    # Initialize evaluator
    evaluator = RagEvaluator()

    OPENSEARCH_URL = f"http://{OPENSEARCH_HOST}:{OPENSEARCH_PORT}"

    # Initialize docsearch
    embeddings = OpenAIEmbeddings(model=MODEL_EMBEDDING)
    docsearch = OpenSearchVectorSearch(
        index_name=INDEX_NAME,
        embedding_function=embeddings,
        opensearch_url=OPENSEARCH_URL,
    )

    # Configure chat model
    chat = ChatOpenAI(model=MODEL_LLM, temperature=0)

    # Load prompts from hub (langchain-ai)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    # Create processing chain
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)
    history_aware_retriever = create_history_aware_retriever(
        llm=chat,
        retriever=docsearch.as_retriever(search_kwargs={"k": 6}),
        prompt=rephrase_prompt,
    )

    qa_chain = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    # Execute QA chain
    result = qa_chain.invoke({"input": query, "chat_history": chat_history})

    # Format response
    output = {
        "query": result.get("input", query),
        "result": result["answer"],
        "source_documents": result.get("context", []),
        "search_terms": result.get("search_terms", ""),
    }

    # Evaluation
    evaluation = evaluator.evaluate_response(
        query=result.get("input", query),
        response=result["answer"],
        context=result.get("context", []),
    )

    # log experiment of evaluator
    with mlflow.start_run(nested=True) as run:

        # log metrics
        mlflow.log_metrics(
            {
                "eval_score": float(evaluation.get("score", 0)),
                "eval_is_relevant": int(evaluation.get("is_relevant", False)),
            }
        )

        # log the response as text file
        mlflow.log_text(str(output["result"]), "response.txt")

        # Tags to track the experiment
        mlflow.set_tags({"component": "rag_pipeline", "evaluator": "simple"})

    output["evaluation"] = evaluation
    return output


if __name__ == "__main__":
    try:
        res = run_llm(query="What are all AWS regions where SageMaker is available?")
        print("Answer:", res["result"])
    except Exception as e:
        print(f"Error processing query: {str(e)}")
