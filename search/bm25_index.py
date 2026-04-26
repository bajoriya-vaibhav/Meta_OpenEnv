"""
ChronoVeritas — BM25 search index over document metadata.

"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from env.models import DocMeta, Document

log = logging.getLogger(__name__)

# Common English stopwords — short, curated for a fact-checking domain
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "was", "are", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "that", "this",
        "it", "its", "not", "no", "as", "if", "so", "than", "then",
    }
)

# Regex: keep only word characters (letters, digits, underscore)
_TOKEN_RE = re.compile(r"\w+")


def _tokenize_text(text: str, *, boost_repeat: int = 1) -> List[str]:
    """
    Lowercase, strip punctuation, remove stopwords.
    Set boost_repeat > 1 to repeat tokens (for field-weight boosting).
    """
    tokens = [
        t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1
    ]
    return tokens * boost_repeat


class BM25Index:
    """
    BM25Okapi wrapper that indexes title + snippet + tags from DocMeta.

    Field weights (via token repetition)
    -------------------------------------
    title   ×2  — most discriminative for a claim-verification task
    snippet ×1
    tags    ×2  — curated labels carry high signal
    """

    TITLE_BOOST = 2
    SNIPPET_BOOST = 1
    TAG_BOOST = 2

    def __init__(self) -> None:
        self._index: Optional[BM25Okapi] = None
        self._metas: List[DocMeta] = []
        self._tokenized: List[List[str]] = []   # kept for incremental rebuild

    # ── Build ──────────────────────────────────────────────────────────

    def build(self, docs: List[Document]) -> None:
        """
        Build the BM25 index from a list of full documents.
        Existing index is replaced entirely.
        """
        self._metas = [d.to_meta() for d in docs]
        self._tokenized = [self._tokenize_meta(m) for m in self._metas]
        self._rebuild_index()
        log.debug("BM25Index built: %d documents", len(self._metas))

    def add_doc(self, doc: Document) -> bool:
        """
        Incrementally add a single document.
        Returns True if added, False if doc_id already exists (no-op).

        Note: BM25Okapi does not support true incremental updates, so this
        triggers a full index rebuild. Acceptable for episode-sized corpora
        (< 1 000 docs); not for large-scale production use.
        """
        existing_ids = {m.doc_id for m in self._metas}
        if doc.doc_id in existing_ids:
            return False

        meta = doc.to_meta()
        self._metas.append(meta)
        self._tokenized.append(self._tokenize_meta(meta))
        self._rebuild_index()
        return True

    def _rebuild_index(self) -> None:
        if self._tokenized:
            self._index = BM25Okapi(self._tokenized)
        else:
            self._index = None

    # ── Tokenisation ───────────────────────────────────────────────────

    def _tokenize_meta(self, meta: DocMeta) -> List[str]:
        title_tokens   = _tokenize_text(meta.title,   boost_repeat=self.TITLE_BOOST)
        snippet_tokens = _tokenize_text(meta.snippet, boost_repeat=self.SNIPPET_BOOST)
        tag_tokens     = _tokenize_text(
            " ".join(meta.tags), boost_repeat=self.TAG_BOOST
        )
        return title_tokens + snippet_tokens + tag_tokens

    # ── Query ──────────────────────────────────────────────────────────

    def query(
        self,
        q: str,
        date_from: Optional[int] = None,
        date_to: Optional[int] = None,
        top_k: int = 10,
        score_threshold: float = float('-inf'),
    ) -> List[DocMeta]:
        """
        Return ranked DocMeta list.

        Parameters
        ----------
        q:
            Natural language query string.
        date_from / date_to:
            Inclusive Unix-epoch timestamp bounds. None = unbounded.
        top_k:
            Maximum results to return.
        score_threshold:
            Minimum BM25 score to include. Defaults to 0.0 (exclude
            zero-score / no-overlap results when any term matches anything).
            Pass -inf to include all.

        Returns
        -------
        List of DocMeta sorted by BM25 score descending.
        """
        if self._index is None or not self._metas:
            return []

        tokens = _tokenize_text(q)
        if not tokens:
            log.debug("BM25Index.query: empty token list after filtering for query %r", q)
            return []

        raw_scores: List[float] = self._index.get_scores(tokens)

        ranked: List[Tuple[float, DocMeta]] = []
        for score, meta in zip(raw_scores, self._metas):
            if score <= score_threshold:
                continue
            if date_from is not None and meta.timestamp < date_from:
                continue
            if date_to is not None and meta.timestamp > date_to:
                continue
            ranked.append((score, meta))

        ranked.sort(key=lambda x: x[0], reverse=True)
        results = [m for _, m in ranked[:top_k]]

        log.debug(
            "BM25Index.query %r → %d candidates, %d returned (top_k=%d)",
            q, len(ranked), len(results), top_k,
        )
        return results

    def query_with_scores(
        self,
        q: str,
        date_from: Optional[int] = None,
        date_to: Optional[int] = None,
        top_k: int = 10,
        score_threshold: float = float('-inf'),
    ) -> List[Tuple[float, DocMeta]]:
        """
        Same as query() but returns (score, meta) pairs.
        Useful for agents that want to weight results by retrieval confidence.
        """
        if self._index is None or not self._metas:
            return []

        tokens = _tokenize_text(q)
        if not tokens:
            return []

        raw_scores = self._index.get_scores(tokens)

        ranked: List[Tuple[float, DocMeta]] = [
            (score, meta)
            for score, meta in zip(raw_scores, self._metas)
            if score > score_threshold
            and (date_from is None or meta.timestamp >= date_from)
            and (date_to is None or meta.timestamp <= date_to)
        ]
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked[:top_k]

    # ── Introspection ──────────────────────────────────────────────────

    @property
    def doc_count(self) -> int:
        return len(self._metas)

    @property
    def is_built(self) -> bool:
        return self._index is not None

    def stats(self) -> Dict[str, object]:
        """Lightweight summary for logging and monitoring."""
        avg_len = (
            sum(len(t) for t in self._tokenized) / len(self._tokenized)
            if self._tokenized
            else 0.0
        )
        return {
            "doc_count": self.doc_count,
            "is_built": self.is_built,
            "avg_tokens_per_doc": round(avg_len, 1),
        }

    # ── Warm-up utility ────────────────────────────────────────────────

    @classmethod
    def build_all(cls) -> List[Dict[str, object]]:
        """
        Utility called at Docker build time to warm-test the index over
        all task files. Returns a list of per-task summary dicts.
        """
        import json

        task_dir = Path(__file__).resolve().parent.parent / "data" / "tasks"
        summaries: List[Dict[str, object]] = []

        for path in sorted(task_dir.glob("*.json")):
            try:
                with open(path) as f:
                    task = json.load(f)
                docs = [Document(**d) for d in task.get("corpus", [])]
                idx = cls()
                idx.build(docs)
                summary: Dict[str, object] = {
                    "task_id": task.get("task_id", path.stem),
                    **idx.stats(),
                }
                summaries.append(summary)
                log.info("[BM25] %s", summary)
            except Exception as exc:  # noqa: BLE001
                log.error("[BM25] Failed to build index for %s: %s", path, exc)
                summaries.append({"task_id": path.stem, "error": str(exc)})

        return summaries