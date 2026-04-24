from __future__ import annotations
from enum import Enum
from typing import List

from core.models import Document
from utils.config import get_config
from utils.logger import log_step


class QueryType(str, Enum):
    FACTUAL    = "factual"
    REASONING  = "reasoning"
    MULTI_HOP  = "multi_hop"
    DEFINITION = "definition"
    GENERAL    = "general"


_REASONING_SIGNALS = [
    "why", "how does", "explain", "compare",
    "difference", "advantage", "disadvantage",
    "impact", "effect", "cause"
]
_MULTIHOP_SIGNALS = [
    "and", "both", "relationship between",
    "connection", "first", "then", "after"
]
_DEFINITION_SIGNALS = [
    "what is", "what are", "define",
    "definition of", "meaning of"
]


def classify_query(query: str) -> QueryType:
    q = query.lower().strip()
    if any(s in q for s in _DEFINITION_SIGNALS):
        return QueryType.DEFINITION
    if any(s in q for s in _REASONING_SIGNALS):
        return QueryType.REASONING
    if any(s in q for s in _MULTIHOP_SIGNALS):
        return QueryType.MULTI_HOP
    return QueryType.FACTUAL


class PromptOptimizer:

    def __init__(self) -> None:
        self.cfg = get_config().si_layer

    def optimize_query(self, raw_query: str) -> tuple[str, QueryType]:
        qtype = classify_query(raw_query)

        if not self.cfg.enable_prompt_optimization:
            return raw_query, qtype

        q = raw_query.strip()

        if qtype == QueryType.DEFINITION:
            q = q.rstrip("?") + " definition explanation overview"
        elif qtype == QueryType.REASONING:
            q = q.rstrip("?") + " explanation reasoning mechanism"
        elif qtype == QueryType.MULTI_HOP:
            pass

        log_step("SI:OPTIMIZE", f"[{qtype.value}] '{raw_query}' -> '{q}'")
        return q, qtype

    def build_generation_prompt(
        self,
        query: str,
        docs: List[Document],
        query_type: QueryType,
        use_cot: bool = False,
    ) -> str:
        context_block = self._format_context(docs)
        instruction   = self._instruction_for_type(query_type, use_cot)

        prompt = f"""You are a precise, factual AI assistant.

CONTEXT (retrieved documents - use ONLY this information):
{context_block}

QUESTION: {query}

{instruction}

ANSWER:"""
        return prompt

    def _format_context(self, docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            source = f"[Source: {doc.source}]" if doc.source else ""
            parts.append(f"[{i}] {source}\n{doc.text.strip()}")
        return "\n\n".join(parts)

    def _instruction_for_type(
        self,
        qtype: QueryType,
        use_cot: bool
    ) -> str:
        base = (
            "Answer using ONLY the information in the CONTEXT above. "
            "Keep your answer strictly to what the documents say. "
            "Do NOT mention what something does not do. "
            "Do NOT compare with other methods. "
            "Do NOT add information outside the context. "
            "Cite document numbers like [1], [2] when using a source. "
            "Give a direct, positive, factual answer only."
        )

        if use_cot and qtype in (QueryType.REASONING, QueryType.MULTI_HOP):
            return (
                "Think step by step:\n"
                "Step 1: What key facts does the context give?\n"
                "Step 2: How do they answer the question?\n"
                "Step 3: Write a direct factual answer.\n\n"
                + base
            )

        if qtype == QueryType.DEFINITION:
            return base + " Start with a one sentence definition, then elaborate."

        return base

    def decompose_query(self, query: str) -> List[str]:
        q = query.strip().rstrip("?")

        if " and " in q.lower():
            parts = q.lower().split(" and ", 1)
            sub1  = parts[0].strip().capitalize() + "?"
            sub2  = parts[1].strip().capitalize() + "?"
            log_step("SI:DECOMPOSE", f"Split into: {sub1} | {sub2}")
            return [sub1, sub2]

        if "between" in q.lower() and "and" in q.lower():
            return [
                f"What is {q.split('between')[-1].split('and')[0].strip()}?",
                f"What is {q.split('and')[-1].strip()}?",
                query,
            ]

        return [query]