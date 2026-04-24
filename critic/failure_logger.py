from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import List

from core.models import VerificationResult, RetryDecision, RetryStrategy
from utils.config import get_config


class FailureLogger:

    def __init__(self) -> None:
        self.log_path = Path(get_config().debug.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_failure(
        self,
        query: str,
        attempt: int,
        result: VerificationResult,
        decision: RetryDecision,
    ) -> None:
        entry = {
            "timestamp":        datetime.utcnow().isoformat(),
            "query":            query,
            "attempt":          attempt,
            "confidence":       round(result.confidence, 4),
            "failure_reason":   result.failure_reason,
            "strategy_chosen":  decision.strategy.value,
            "rewritten_query":  decision.rewritten_query,
            "new_top_k":        decision.new_top_k,
            "sub_questions":    decision.sub_questions,
            "verdicts": [
                {
                    "claim": v.claim[:120],
                    "label": v.label.value,
                    "score": round(v.score, 4),
                }
                for v in result.verdicts
            ],
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def log_success(
        self,
        query: str,
        total_retries: int,
        final_confidence: float,
        winning_strategy: RetryStrategy | None,
    ) -> None:
        entry = {
            "timestamp":        datetime.utcnow().isoformat(),
            "type":             "success_after_retry",
            "query":            query,
            "total_retries":    total_retries,
            "final_confidence": round(final_confidence, 4),
            "winning_strategy": winning_strategy.value if winning_strategy else None,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def get_strategy_stats(self) -> dict:
        if not self.log_path.exists():
            return {}

        wins:     dict[str, int] = {}
        attempts: dict[str, int] = {}

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry.get("type") == "success_after_retry":
                    s = entry.get("winning_strategy")
                    if s:
                        wins[s] = wins.get(s, 0) + 1
                if "strategy_chosen" in entry:
                    s = entry["strategy_chosen"]
                    attempts[s] = attempts.get(s, 0) + 1

        stats = {}
        for strategy in RetryStrategy:
            s = strategy.value
            a = attempts.get(s, 0)
            w = wins.get(s, 0)
            stats[s] = {
                "attempts": a,
                "wins":     w,
                "win_rate": round(w / a, 3) if a > 0 else 0.0,
            }
        return stats