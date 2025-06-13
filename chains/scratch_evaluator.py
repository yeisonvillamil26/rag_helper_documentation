from typing import Dict, List, Any
from langchain.schema import Document
from langchain_openai import ChatOpenAI
import json
from tools.contants import (
    MODEL_LLM,
)


class RagEvaluator:
    def __init__(self):
        self.llm = ChatOpenAI(model=MODEL_LLM)

    def parse_response(self, raw_response: str) -> Dict[str, Any]:
        """Parsing of LLM response with different strategies"""
        text = raw_response.strip()

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        if not text.startswith("{"):
            text = text[text.find("{") :]
        if not text.endswith("}"):
            text = text[: text.rfind("}") + 1]

        return json.loads(text)

    def evaluate_response(
        self, query: str, response: str, context: List[Document] = None, **kwargs
    ) -> Dict[str, Any]:
        """Evaluates whether the response adequately answers the question based on the context."""
        context_text = (
            "\n\n".join(doc.page_content for doc in context)
            if context
            else "No context provided"
        )

        prompt = f"""
        You are a strict JSON-only evaluator. Analyze this chatbot response and return ONLY valid JSON with 
        these EXACT keys:
        - "score" (0-1)
        - "is_relevant" (true/false)
        - "feedback" (string)

        Evaluation Rules:
        1. Score based on accuracy against context
        2. Relevance to the question
        3. Feedback should be concise

        Context:
        {context_text}

        Question: {query}
        Response: {response}

        Your response MUST be valid JSON between ```json``` markers like this:
        ```json
        {{
            "score": 0.85,
            "is_relevant": true,
            "feedback": "The response accurately addresses the question using context"
        }}
        ```
        """

        evaluation = self.llm.invoke(prompt)
        return self.parse_response(evaluation.content)
