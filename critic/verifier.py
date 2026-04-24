from __future__ import annotations
import re
import numpy as np
from typing import List

from sentence_transformers import CrossEncoder

from core.models import (
    Document, VerificationResult,
    ClaimVerdict, NLILabel
)
from utils.config import get_config
from utils.logger import log_step, log_confidence

_IDX_CONTRADICTION = 0
_IDX_ENTAILMENT    = 2
_IDX_NEUTRAL       = 1


class VerificationCritic:

    def __init__(self) -> None:
        cfg = get_config()
        self.nli_model_name = cfg.critic.nli_model
        self.threshold      = cfg.critic.confidence_threshold
        self._nli: CrossEncoder | None = None

    @property
    def nli(self) -> CrossEncoder:
        if self._nli is None:
            log_step("CRITIC", f"Loading NLI model: {self.nli_model_name}")
            self._nli = CrossEncoder(self.nli_model_name)
        return self._nli

    def _split_into_claims(self, answer: str) -> List[str]:
        raw    = re.split(r'(?<=[.!?])\s+', answer.strip())
        claims = [s.strip() for s in raw if len(s.strip()) > 20]
        if not claims:
            claims = [answer.strip()]
        return claims

    def _score_claim_against_docs(
        self,
        claim: str,
        docs: List[Document],
    ) -> tuple[NLILabel, float, str | None]:
        if not docs:
            return NLILabel.BASELESS, 0.0, None

        pairs      = [[doc.text, claim] for doc in docs]
        raw_scores = self.nli.predict(pairs)

        best_label = NLILabel.BASELESS
        best_score = 0.0
        best_doc   = None

        for doc, score_vec in zip(docs, raw_scores):
            e     = np.exp(score_vec - np.max(score_vec))
            probs = e / e.sum()

            contra  = float(probs[_IDX_CONTRADICTION])
            entail  = float(probs[_IDX_ENTAILMENT])
            neutral = float(probs[_IDX_NEUTRAL])

            if entail > best_score and entail > contra and entail > neutral:
                best_score = entail
                best_label = NLILabel.ENTAILED
                best_doc   = doc.id

            if contra > 0.85 and contra > entail:
                best_score = contra
                best_label = NLILabel.CONTRADICTED
                best_doc   = doc.id

        return best_label, best_score, best_doc

    def _aggregate_confidence(
        self,
        verdicts: List[ClaimVerdict]
    ) -> float:
        if not verdicts:
            return 0.0

        total = 0.0
        for v in verdicts:
            if v.label == NLILabel.ENTAILED:
                total += v.score
            elif v.label == NLILabel.CONTRADICTED:
                total -= v.score * 0.5

        raw = total / len(verdicts)
        return max(0.0, min(1.0, raw))

    def _diagnose_failure(
        self,
        verdicts: List[ClaimVerdict],
        query: str
    ) -> str:
        labels             = [v.label for v in verdicts]
        baseless_ratio     = labels.count(NLILabel.BASELESS) / max(len(labels), 1)
        contradicted_ratio = labels.count(NLILabel.CONTRADICTED) / max(len(labels), 1)

        if baseless_ratio > 0.5:
            return "low_retrieval_coverage"
        if contradicted_ratio > 0.3:
            return "contradicted_by_sources"
        return "low_confidence_generic"

    def verify(
        self,
        answer: str,
        docs: List[Document],
        query: str = "",
    ) -> VerificationResult:
        log_step("CRITIC", "Splitting answer into claims...")
        claims = self._split_into_claims(answer)
        log_step("CRITIC", f"Checking {len(claims)} claim(s) against {len(docs)} doc(s)...")

        verdicts: List[ClaimVerdict] = []

        for claim in claims:
            label, score, doc_id = self._score_claim_against_docs(claim, docs)
            verdicts.append(ClaimVerdict(
                claim=claim,
                label=label,
                score=score,
                supporting_doc_id=doc_id,
            ))
            short = claim[:60] + "..." if len(claim) > 60 else claim
            log_step("CRITIC", f"  [{label.value:12s}] score={score:.2f} | {short}")

        confidence = self._aggregate_confidence(verdicts)
        passed     = confidence >= self.threshold

        failure_reason = None
        if not passed:
            failure_reason = self._diagnose_failure(verdicts, query)

        log_confidence(confidence, self.threshold)

        return VerificationResult(
            confidence=confidence,
            verdicts=verdicts,
            passed=passed,
            failure_reason=failure_reason,
        )