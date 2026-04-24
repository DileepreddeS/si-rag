from __future__ import annotations
from typing import List

from core.models import (
    VerificationResult, RetryDecision,
    RetryStrategy, NLILabel, Document
)
from si_layer.prompt_optimizer import (
    PromptOptimizer, classify_query, QueryType
)
from utils.config import get_config
from utils.logger import log_step, log_retry


class RetryStrategist:

    def __init__(self) -> None:
        self.cfg       = get_config()
        self.optimizer = PromptOptimizer()

    def _diagnose(
        self,
        query: str,
        docs: List[Document],
        result: VerificationResult,
    ) -> str:
        if not docs:
            return "empty_retrieval"

        labels = [v.label for v in result.verdicts]

        baseless_ratio     = labels.count(NLILabel.BASELESS) / max(len(labels), 1)
        contradicted_ratio = labels.count(NLILabel.CONTRADICTED) / max(len(labels), 1)

        qtype = classify_query(query)
        if qtype == QueryType.MULTI_HOP:
            return "multi_hop_query"

        if baseless_ratio > 0.5:
            return "low_retrieval_coverage"

        if contradicted_ratio > 0.3:
            return "contradicted_by_sources"

        if result.confidence < 0.4:
            return "vague_query"

        return "low_confidence_generic"

    def decide(
        self,
        query: str,
        docs: List[Document],
        result: VerificationResult,
        attempt: int,
    ) -> RetryDecision:
        reason   = result.failure_reason or self._diagnose(query, docs, result)
        cfg_ret  = self.cfg.retrieval

        log_retry(attempt, "diagnosing...", reason)

        if reason == "multi_hop_query":
            sub_qs = self.optimizer.decompose_query(query)
            decision = RetryDecision(
                strategy=RetryStrategy.DECOMPOSE,
                reason=reason,
                sub_questions=sub_qs,
            )

        elif reason in ("low_retrieval_coverage", "empty_retrieval"):
            new_k = min(cfg_ret.top_k_retrieval * 2, 60)
            decision = RetryDecision(
                strategy=RetryStrategy.EXPAND_CONTEXT,
                reason=reason,
                new_top_k=new_k,
            )

        else:
            rewritten = self._rewrite_query(query, reason)
            decision  = RetryDecision(
                strategy=RetryStrategy.REWRITE_QUERY,
                reason=reason,
                rewritten_query=rewritten,
            )

        log_retry(attempt, decision.strategy.value, reason)
        return decision

    def _rewrite_query(self, query: str, reason: str) -> str:
        q = query.strip().rstrip("?")

        if reason == "contradicted_by_sources":
            return f"specifically, {q} according to documented sources?"

        if reason == "vague_query":
            return f"{q} - provide a detailed factual explanation"

        return f"{q} definition explanation details facts"