from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from core.models import Document
from utils.config import get_config
from utils.logger import log_step


class SparseRetriever:

    def __init__(self) -> None:
        cfg = get_config().retrieval
        self.bm25_path   = Path(cfg.bm25_path)
        self.corpus_path = Path(cfg.corpus_path)
        self._bm25: BM25Okapi | None = None
        self._docs: List[Document] = []

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def build_index(self) -> None:
        log_step("SPARSE", f"Building BM25 index from {self.corpus_path}")
        self._docs = self._load_corpus()
        tokenized  = [self._tokenize(d.text) for d in self._docs]
        bm25       = BM25Okapi(tokenized)

        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bm25_path, "wb") as f:
            pickle.dump({"bm25": bm25, "docs": self._docs}, f)

        log_step("SPARSE", f"BM25 index saved -> {self.bm25_path} ({len(self._docs)} docs)")

    def _load_corpus(self) -> List[Document]:
        docs = []
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line.strip())
                docs.append(Document(
                    id=str(row["id"]),
                    text=row["text"],
                    source=row.get("source", ""),
                ))
        return docs

    def load(self) -> None:
        if not self.bm25_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {self.bm25_path}. "
                "Run: python scripts/build_index.py"
            )
        log_step("SPARSE", f"Loading BM25 index from {self.bm25_path}")
        with open(self.bm25_path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._docs = data["docs"]
        log_step("SPARSE", f"Loaded {len(self._docs)} docs")

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        if self._bm25 is None:
            self.load()

        tokens  = self._tokenize(query)
        scores  = self._bm25.get_scores(tokens)
        top_idx = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in top_idx:
            doc = self._docs[idx]
            results.append(Document(
                id=doc.id,
                text=doc.text,
                source=doc.source,
                bm25_score=float(scores[idx]),
            ))
        return results