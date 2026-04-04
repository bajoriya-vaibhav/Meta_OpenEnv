"""
ChronoVeritas — BM25 search index over document metadata.
"""
from __future__ import annotations

from typing import List, Optional

from rank_bm25 import BM25Okapi

from env.models import DocMeta, Document


class BM25Index:
    """BM25Okapi wrapper that indexes title + snippet + tags."""

    def __init__(self) -> None:
        self._index: Optional[BM25Okapi] = None
        self._metas: List[DocMeta] = []

    # ── Build ────────────────────────────────────────────────────────

    def build(self, docs: List[Document]) -> None:
        """Build the BM25 index from a list of full documents."""
        self._metas = [
            DocMeta(
                doc_id=d.doc_id,
                title=d.title,
                source=d.source,
                timestamp=d.timestamp,
                tags=d.tags,
                snippet=d.snippet,
            )
            for d in docs
        ]
        tokenized = [self._tokenize(m) for m in self._metas]
        if tokenized:
            self._index = BM25Okapi(tokenized)
        else:
            self._index = None

    @staticmethod
    def _tokenize(meta: DocMeta) -> List[str]:
        text = f"{meta.title} {meta.snippet} {' '.join(meta.tags)}"
        return text.lower().split()

    # ── Query ────────────────────────────────────────────────────────

    def query(
        self,
        q: str,
        date_from: Optional[int] = None,
        date_to: Optional[int] = None,
        top_k: int = 10,
    ) -> List[DocMeta]:
        """Return ranked DocMeta list. Optional date filters narrow results."""
        if self._index is None or not self._metas:
            return []

        tokens = q.lower().split()
        scores = self._index.get_scores(tokens)

        # Pair scores with metas, apply date filters
        results: list[tuple[float, DocMeta]] = []
        for score, meta in zip(scores, self._metas):
            if date_from is not None and meta.timestamp < date_from:
                continue
            if date_to is not None and meta.timestamp > date_to:
                continue
            results.append((score, meta))

        # Sort descending by score
        results.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in results[:top_k]]

    @classmethod
    def build_all(cls) -> None:
        """Utility called at Docker build time to warm-test the index."""
        import json
        import glob
        import os

        task_dir = os.path.join(os.path.dirname(__file__), "..", "data", "tasks")
        for path in glob.glob(os.path.join(task_dir, "*.json")):
            with open(path) as f:
                task = json.load(f)
            docs = [Document(**d) for d in task.get("corpus", [])]
            idx = cls()
            idx.build(docs)
            print(f"[BM25] Built index for {task.get('task_id', path)}: {len(docs)} docs")
