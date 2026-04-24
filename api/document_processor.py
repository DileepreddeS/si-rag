from __future__ import annotations
import json
import pickle
import uuid
from pathlib import Path
from typing import List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from core.models import Document
from utils.config import get_config
from utils.logger import log_step


class DocumentProcessor:

    CHUNK_SIZE    = 500
    CHUNK_OVERLAP = 50

    def __init__(self) -> None:
        cfg = get_config().retrieval
        self.index_path  = Path(cfg.index_path)
        self.bm25_path   = Path(cfg.bm25_path)
        self.corpus_path = Path(cfg.corpus_path)
        self.upload_path = Path("data/uploads")
        self.upload_path.mkdir(parents=True, exist_ok=True)
        self._embed_model: SentenceTransformer | None = None

    @property
    def embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            log_step("PROCESSOR", "Loading embedding model...")
            self._embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return self._embed_model

    # ── Text extraction ───────────────────────────────────────────────────────

    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        import fitz  # PyMuPDF
        doc  = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()

    def extract_text_from_txt(self, file_bytes: bytes) -> str:
        return file_bytes.decode("utf-8", errors="ignore").strip()

    # ── Chunking ──────────────────────────────────────────────────────────────

    def chunk_text(
        self,
        text: str,
        source: str = "",
    ) -> List[Document]:
        chunks  = []
        start   = 0
        text    = text.replace("\n", " ").strip()

        while start < len(text):
            end   = min(start + self.CHUNK_SIZE, len(text))
            chunk = text[start:end].strip()

            if len(chunk) > 50:
                chunks.append(Document(
                    id=str(uuid.uuid4())[:8],
                    text=chunk,
                    source=source,
                ))

            start += self.CHUNK_SIZE - self.CHUNK_OVERLAP

        return chunks

    # ── Index update ──────────────────────────────────────────────────────────

    def add_documents_to_index(self, docs: List[Document]) -> int:
        if not docs:
            return 0

        # ── Update corpus.jsonl ───────────────────────────────────────────
        with open(self.corpus_path, "a", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps({
                    "id": doc.id,
                    "text": doc.text,
                    "source": doc.source,
                }) + "\n")

        # ── Update FAISS index ────────────────────────────────────────────
        texts = [d.text for d in docs]
        vecs  = self.embed_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        index = faiss.read_index(str(self.index_path))
        index.add(vecs)
        faiss.write_index(index, str(self.index_path))

        # Update id map
        id_map_path = self.index_path.with_suffix(".idmap.pkl")
        with open(id_map_path, "rb") as f:
            data = pickle.load(f)

        data["ids"].extend([d.id for d in docs])
        for doc in docs:
            data["corpus"][doc.id] = doc

        with open(id_map_path, "wb") as f:
            pickle.dump(data, f)

        # ── Update BM25 index ─────────────────────────────────────────────
        with open(self.bm25_path, "rb") as f:
            bm25_data = pickle.load(f)

        existing_docs = bm25_data["docs"]
        existing_docs.extend(docs)

        tokenized = [d.text.lower().split() for d in existing_docs]
        new_bm25  = BM25Okapi(tokenized)

        with open(self.bm25_path, "wb") as f:
            pickle.dump({"bm25": new_bm25, "docs": existing_docs}, f)

        log_step("PROCESSOR", f"Added {len(docs)} chunks to indexes")
        return len(docs)

    def get_total_docs(self) -> int:
        if not self.corpus_path.exists():
            return 0
        with open(self.corpus_path, "r") as f:
            return sum(1 for _ in f)

    # ── Main process method ───────────────────────────────────────────────────

    def process_file(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> tuple[int, int]:
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            text = self.extract_text_from_pdf(file_bytes)
        elif ext in (".txt", ".md"):
            text = self.extract_text_from_txt(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        chunks       = self.chunk_text(text, source=filename)
        chunks_added = self.add_documents_to_index(chunks)
        total        = self.get_total_docs()

        return chunks_added, total