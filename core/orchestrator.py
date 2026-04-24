from __future__ import annotations
from typing import List

from core.models import (
    QueryState, PipelineTrace, SIRAGResponse,
    RetryDecision, RetryStrategy, Document
)
from core.llm_client import OllamaClient
from retrieval.hybrid import HybridRetriever
from si_layer.prompt_optimizer import PromptOptimizer, QueryType
from si_layer.retry_strategist import RetryStrategist
from critic.verifier import VerificationCritic
from critic.failure_logger import FailureLogger
from utils.config import get_config
from utils.logger import (
    log_step, log_query, log_retrieved_docs,
    log_answer, log_final_output, log_error, console
)


class SIOrchestrator:

    def __init__(self) -> None:
        cfg = get_config()
        self.max_retries = cfg.critic.max_retries
        self.debug       = cfg.debug.enabled

        self._retriever:  HybridRetriever    | None = None
        self._llm:        OllamaClient       | None = None
        self._optimizer:  PromptOptimizer    | None = None
        self._critic:     VerificationCritic | None = None
        self._strategist: RetryStrategist    | None = None
        self._logger:     FailureLogger      | None = None

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever()
            self._retriever.load()
        return self._retriever

    @property
    def llm(self) -> OllamaClient:
        if self._llm is None:
            self._llm = OllamaClient()
        return self._llm

    @property
    def optimizer(self) -> PromptOptimizer:
        if self._optimizer is None:
            self._optimizer = PromptOptimizer()
        return self._optimizer

    @property
    def critic(self) -> VerificationCritic:
        if self._critic is None:
            self._critic = VerificationCritic()
        return self._critic

    @property
    def strategist(self) -> RetryStrategist:
        if self._strategist is None:
            self._strategist = RetryStrategist()
        return self._strategist

    @property
    def failure_logger(self) -> FailureLogger:
        if self._logger is None:
            self._logger = FailureLogger()
        return self._logger

    def _single_attempt(
        self,
        query: str,
        query_type: QueryType,
        top_k_override: int | None = None,
    ) -> tuple[str, List[Document], List[Document]]:
        retrieved, reranked = self.retriever.retrieve(query, top_k_override)

        if self.debug:
            log_retrieved_docs(
                [{"text": d.text, "source": d.source, "score": d.score}
                 for d in reranked],
                top_k=len(reranked),
            )

        cfg     = get_config().si_layer
        use_cot = cfg.enable_cot and query_type in (
            QueryType.REASONING, QueryType.MULTI_HOP
        )
        prompt = self.optimizer.build_generation_prompt(
            query=query,
            docs=reranked,
            query_type=query_type,
            use_cot=use_cot,
        )

        answer = self.llm.generate(prompt)

        if self.debug:
            log_answer(answer)

        return answer, retrieved, reranked

    def _handle_decompose(
        self,
        decision: RetryDecision,
        query_type: QueryType,
    ) -> str:
        sub_answers = []
        for sub_q in (decision.sub_questions or []):
            log_step("DECOMPOSE", f"Sub-question: {sub_q}")
            opt_q, qt = self.optimizer.optimize_query(sub_q)
            answer, _, _ = self._single_attempt(opt_q, qt)
            sub_answers.append(f"- {sub_q}\n  {answer}")
        return "\n\n".join(sub_answers)

    def run(self, query: str) -> SIRAGResponse:
        console.rule("[bold blue]SI-RAG[/bold blue]")
        log_step("ORCHESTRATOR", f"Query: {query}")

        optimized_q, query_type = self.optimizer.optimize_query(query)

        state = QueryState(
            original_query=query,
            optimized_query=optimized_q,
            current_query=optimized_q,
        )

        if self.debug:
            log_query(query, optimized_q)

        trace = PipelineTrace(query_state=state) if self.debug else None

        final_answer     = ""
        final_docs: List[Document] = []
        last_result      = None
        winning_strategy: RetryStrategy | None = None
        top_k_override:  int | None = None

        for attempt in range(self.max_retries + 1):
            state.attempt = attempt
            log_step("ORCHESTRATOR", f"Attempt {attempt + 1}/{self.max_retries + 1}")

            try:
                answer, retrieved, reranked = self._single_attempt(
                    state.current_query,
                    query_type,
                    top_k_override,
                )

                if trace:
                    trace.retrieved_docs = retrieved
                    trace.reranked_docs  = reranked
                    trace.answer         = answer

                log_step("ORCHESTRATOR", "Verifying answer...")
                result      = self.critic.verify(answer, reranked, state.current_query)
                last_result = result

                if trace:
                    trace.verification = result

                if result.passed:
                    log_step("ORCHESTRATOR", f"Verification PASSED on attempt {attempt + 1}")
                    if attempt > 0:
                        self.failure_logger.log_success(
                            query, attempt, result.confidence, winning_strategy
                        )
                    final_answer = answer
                    final_docs   = reranked
                    break

                if attempt < self.max_retries:
                    decision = self.strategist.decide(
                        state.current_query, reranked, result, attempt + 1
                    )
                    state.strategy_history.append(decision.strategy)
                    winning_strategy = decision.strategy

                    self.failure_logger.log_failure(
                        state.current_query, attempt, result, decision
                    )

                    if trace:
                        trace.retry_decisions.append(decision)

                    if decision.strategy == RetryStrategy.REWRITE_QUERY:
                        state.current_query = decision.rewritten_query or state.current_query

                    elif decision.strategy == RetryStrategy.EXPAND_CONTEXT:
                        top_k_override = decision.new_top_k

                    elif decision.strategy == RetryStrategy.DECOMPOSE:
                        answer      = self._handle_decompose(decision, query_type)
                        result      = self.critic.verify(answer, reranked, query)
                        last_result = result
                        final_answer = answer
                        final_docs   = reranked
                        break

                else:
                    log_step("ORCHESTRATOR", "Max retries reached - returning best answer")
                    final_answer = answer
                    final_docs   = reranked

            except Exception as e:
                log_error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries:
                    raise

        citations        = list({d.source for d in final_docs if d.source})
        final_confidence = last_result.confidence if last_result else 0.0

        if self.debug:
            if trace:
                trace.final_answer     = final_answer
                trace.final_confidence = final_confidence
                trace.total_retries    = state.attempt
                trace.citations        = citations

            log_final_output(
                final_answer,
                final_confidence,
                state.attempt,
                citations,
            )

        return SIRAGResponse(
            answer=final_answer,
            confidence=final_confidence,
            citations=citations,
            total_retries=state.attempt,
            trace=trace,
        )