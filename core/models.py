from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class RetryStrategy(str, Enum):
    REWRITE_QUERY  = "REWRITE_QUERY"
    EXPAND_CONTEXT = "EXPAND_CONTEXT"
    DECOMPOSE      = "DECOMPOSE"


class NLILabel(str, Enum):
    ENTAILED     = "ENTAILED"
    CONTRADICTED = "CONTRADICTED"
    BASELESS     = "BASELESS"


@dataclass
class Document:
    id: str
    text: str
    source: str = ""
    score: float = 0.0
    bm25_score: float = 0.0
    faiss_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class QueryState:
    original_query: str
    optimized_query: str = ""
    current_query: str = ""
    attempt: int = 0
    strategy_history: List[RetryStrategy] = field(default_factory=list)

    def __post_init__(self):
        if not self.optimized_query:
            self.optimized_query = self.original_query
        if not self.current_query:
            self.current_query = self.optimized_query


@dataclass
class ClaimVerdict:
    claim: str
    label: NLILabel
    supporting_doc_id: Optional[str] = None
    score: float = 0.0


@dataclass
class VerificationResult:
    confidence: float
    verdicts: List[ClaimVerdict] = field(default_factory=list)
    passed: bool = False
    failure_reason: Optional[str] = None


@dataclass
class RetryDecision:
    strategy: RetryStrategy
    reason: str
    rewritten_query: Optional[str] = None
    new_top_k: Optional[int] = None
    sub_questions: Optional[List[str]] = None


@dataclass
class PipelineTrace:
    query_state: QueryState
    retrieved_docs: List[Document] = field(default_factory=list)
    reranked_docs: List[Document] = field(default_factory=list)
    answer: str = ""
    verification: Optional[VerificationResult] = None
    retry_decisions: List[RetryDecision] = field(default_factory=list)
    final_answer: str = ""
    final_confidence: float = 0.0
    total_retries: int = 0
    citations: List[str] = field(default_factory=list)


@dataclass
class SIRAGResponse:
    answer: str
    confidence: float
    citations: List[str]
    total_retries: int
    trace: Optional[PipelineTrace] = None