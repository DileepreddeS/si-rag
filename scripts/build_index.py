import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from utils.logger import log_step, log_error


def main() -> None:
    log_step("BUILD", "Starting index build...")

    try:
        dense = DenseRetriever()
        dense.build_index()
        log_step("BUILD", "FAISS index done.")
    except Exception as e:
        log_error(f"Dense index failed: {e}")
        raise

    try:
        sparse = SparseRetriever()
        sparse.build_index()
        log_step("BUILD", "BM25 index done.")
    except Exception as e:
        log_error(f"Sparse index failed: {e}")
        raise

    log_step("BUILD", "Both indexes built successfully.")


if __name__ == "__main__":
    main()