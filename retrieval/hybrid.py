from __future__ import annotations
from typing import List, Dict

from sentence_transformers import CrossEncoder

from core.models import Document
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from utils.config import get_config
from utils.logger import log_step


class HybridRetriever:

    RRF_K = 60

    def __init__(self) -> None:
        cfg = get_config().retrieval
        self.top_k_retrieval = cfg.top_k_retrieval
        self.top_k_rerank    = cfg.top_k_rerank
        self.bm25_weight     = cfg.bm25_weight
        self.faiss_weight    = cfg.faiss_weight
        self.reranker_name   = cfg.reranker_model
        self.dense           = DenseRetriever()
        self.sparse          = SparseRetriever()
        self._reranker: CrossEncoder | None = None

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            log_step("RERANK", f"Loading cross-encoder: {self.reranker_name}")
            self._reranker = CrossEncoder(self.reranker_name, max_length=512)
        return self._reranker

    def _rrf_fusion(
        self,
        dense_docs: List[Document],
        sparse_docs: List[Document],
    ) -> List[Document]:
        scores:  Dict[str, float]    = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs, 1):
            scores[doc.id]  = scores.get(doc.id, 0.0) + self.faiss_weight / (self.RRF_K + rank)
            doc_map[doc.id] = doc

        for rank, doc in enumerate(sparse_docs, 1):
            scores[doc.id] = scores.get(doc.id, 0.0) + self.bm25_weight / (self.RRF_K + rank)
            if doc.id not in doc_map:
                doc_map[doc.id] = doc

        fused = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)

        result = []
        for doc_id in fused:
            doc       = doc_map[doc_id]
            doc.score = scores[doc_id]
            result.append(doc)
        return result

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        pairs         = [[query, doc.text] for doc in docs]
        rerank_scores = self.reranker.predict(pairs)

        for doc, score in zip(docs, rerank_scores):
            doc.rerank_score = float(score)
            doc.score        = float(score)

        reranked = sorted(docs, key=lambda d: d.rerank_score, reverse=True)
        return reranked[:self.top_k_rerank]

    def load(self) -> None:
        self.dense.load()
        self.sparse.load()

    def retrieve(
        self,
        query: str,
        top_k_override: int | None = None,
    ) -> tuple[List[Document], List[Document]]:
        top_k = top_k_override or self.top_k_retrieval

        log_step("RETRIEVAL", f"Dense retrieval  top_k={top_k}")
        dense_docs  = self.dense.retrieve(query, top_k)

        log_step("RETRIEVAL", f"Sparse retrieval top_k={top_k}")
        sparse_docs = self.sparse.retrieve(query, top_k)

        log_step("RETRIEVAL", "RRF fusion")
        fused = self._rrf_fusion(dense_docs, sparse_docs)

        log_step("RETRIEVAL", f"Reranking -> top {self.top_k_rerank}")
        reranked = self._rerank(query, fused)

        return fused, reranked