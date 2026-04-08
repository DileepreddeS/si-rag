import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.hybrid import HybridRetriever
from utils.logger import log_step, log_retrieved_docs, console

QUERIES = [
    "What is Retrieval-Augmented Generation?",
    "How does BM25 work for information retrieval?",
    "What is hallucination in language models?",
]


def test_retrieval_returns_results() -> None:
    retriever = HybridRetriever()
    retriever.load()

    for query in QUERIES:
        console.rule(f"[bold]Query: {query}[/bold]")
        fused, reranked = retriever.retrieve(query)

        assert len(reranked) > 0, f"No results for: {query}"
        assert reranked[0].score > 0.0, "Top doc has zero score"

        log_retrieved_docs(
            [{"text": d.text, "source": d.source, "score": d.score}
             for d in reranked],
            top_k=len(reranked),
        )
        log_step("TEST", f"PASS -- {len(reranked)} docs returned\n")


def test_top_k_respected() -> None:
    retriever = HybridRetriever()
    retriever.load()
    _, reranked = retriever.retrieve("transformer attention mechanism")

    from utils.config import get_config
    expected_k = get_config().retrieval.top_k_rerank
    assert len(reranked) <= expected_k
    log_step("TEST", f"PASS -- reranked count {len(reranked)} <= {expected_k}")


if __name__ == "__main__":
    log_step("PHASE1", "Running retrieval tests...\n")
    test_retrieval_returns_results()
    test_top_k_respected()
    log_step("PHASE1", "All Phase 1 tests passed.")