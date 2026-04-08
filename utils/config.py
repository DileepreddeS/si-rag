from __future__ import annotations
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 1024


@dataclass
class RetrievalConfig:
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    bm25_weight: float = 0.4
    faiss_weight: float = 0.6
    index_path: str = "data/indexes/faiss.index"
    bm25_path: str = "data/indexes/bm25.pkl"
    corpus_path: str = "data/raw/corpus.jsonl"


@dataclass
class CriticConfig:
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    confidence_threshold: float = 0.75
    max_retries: int = 3


@dataclass
class SILayerConfig:
    enable_prompt_optimization: bool = True
    enable_cot: bool = True
    retry_strategies: List[str] = field(
        default_factory=lambda: ["REWRITE_QUERY", "EXPAND_CONTEXT", "DECOMPOSE"]
    )


@dataclass
class DebugConfig:
    enabled: bool = True
    show_retrieved_docs: bool = True
    show_confidence: bool = True
    show_retry_trace: bool = True
    log_path: str = "data/failure_log.jsonl"


@dataclass
class EvalConfig:
    dataset: str = "natural_questions"
    metrics: List[str] = field(
        default_factory=lambda: ["faithfulness", "answer_relevancy", "context_precision"]
    )


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    si_layer: SILayerConfig = field(default_factory=SILayerConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        root = Path(__file__).resolve().parent.parent
        path = root / "configs" / "config.yaml"

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(
        llm=LLMConfig(**raw.get("llm", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        critic=CriticConfig(**raw.get("critic", {})),
        si_layer=SILayerConfig(**raw.get("si_layer", {})),
        debug=DebugConfig(**raw.get("debug", {})),
        eval=EvalConfig(**raw.get("eval", {})),
    )


_cfg: Config | None = None


def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg