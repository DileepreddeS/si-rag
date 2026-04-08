from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from core.models import Document
from utils.config import get_config
from utils.logger import log_step


class DenseRetriever:

    def __init__(self) -> None:
        cfg = get_config().retrieval
        self.model_name  = cfg.embedding_model
        self.index_path  = Path(cfg.index_path)
        self.corpus_path = Path(cfg.corpus_path)
        self._model: SentenceTransformer | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._id_map: List[str] = []
        self._corpus: dict[str, Document] = {}

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            log_step("DENSE", f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return vecs.astype(np.float32)

    def build_index(self) -> None:
        log_step("DENSE", f"Building FAISS index from {self.corpus_path}")
        docs = self._load_corpus()
        texts = [d.text for d in docs]
        ids   = [d.id   for d in docs]

        log_step("DENSE", f"Encoding {len(texts)} documents...")
        vecs = self._encode(texts)

        dim   = vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))

        id_map_path = self.index_path.with_suffix(".idmap.pkl")
        with open(id_map_path, "wb") as f:
            pickle.dump({"ids": ids, "corpus": {d.id: d for d in docs}}, f)

        log_step("DENSE", f"Index saved -> {self.index_path} ({len(ids)} docs)")

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
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Run: python scripts/build_index.py"
            )
        log_step("DENSE", f"Loading FAISS index from {self.index_path}")
        self._index = faiss.read_index(str(self.index_path))

        id_map_path = self.index_path.with_suffix(".idmap.pkl")
        with open(id_map_path, "rb") as f:
            data = pickle.load(f)

        self._id_map = data["ids"]
        self._corpus = data["corpus"]
        log_step("DENSE", f"Loaded {len(self._id_map)} docs into memory")

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        if self._index is None:
            self.load()

        q_vec = self._encode([query])
        scores, indices = self._index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc_id = self._id_map[idx]
            doc    = self._corpus[doc_id]
            results.append(Document(
                id=doc.id,
                text=doc.text,
                source=doc.source,
                faiss_score=float(score),
            ))
        return results