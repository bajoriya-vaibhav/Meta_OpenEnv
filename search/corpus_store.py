"""
ChronoVeritas — Corpus store: load and serve documents.

"""

from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Optional

from env.models import DocMeta, Document

log = logging.getLogger(__name__)


class CorpusStore:
    """In-memory document store loaded from a task corpus list."""

    def __init__(self) -> None:
        self._docs: Dict[str, Document] = {}

    # ── Loading ────────────────────────────────────────────────────────

    def load_from_task_corpus(
        self,
        corpus: List[Document],
        *,
        merge: bool = False,
    ) -> int:
        """
        Load documents from a task's corpus list.

        Parameters
        ----------
        corpus:
            Documents to load.
        merge:
            If False (default), existing documents are cleared first.
            If True, new documents are added alongside existing ones;
            conflicting doc_ids are overwritten with the new version.

        Returns
        -------
        Number of documents newly added or updated.
        """
        if not merge:
            self._docs.clear()

        added = 0
        for doc in corpus:
            existed = doc.doc_id in self._docs
            self._docs[doc.doc_id] = doc
            if not existed:
                added += 1

        log.debug(
            "CorpusStore loaded %d docs (%s mode, store size now %d)",
            added,
            "merge" if merge else "replace",
            len(self._docs),
        )
        return added

    # ── Single-document mutations ──────────────────────────────────────

    def add_doc(self, doc: Document) -> bool:
        """
        Add a single document.
        Returns True if newly added, False if doc_id already existed (overwritten).
        """
        existed = doc.doc_id in self._docs
        self._docs[doc.doc_id] = doc
        return not existed

    def remove_doc(self, doc_id: str) -> bool:
        """
        Remove a document by ID.
        Returns True if found and removed, False if not present.
        """
        if doc_id in self._docs:
            del self._docs[doc_id]
            return True
        return False

    # ── Retrieval ──────────────────────────────────────────────────────

    def get_doc(self, doc_id: str) -> Optional[Document]:
        """Retrieve a full document by ID. Returns None if not found."""
        return self._docs.get(doc_id)

    def get_meta(self, doc_id: str) -> Optional[DocMeta]:
        """
        Retrieve only the metadata stub for a document.
        Uses Document.to_meta() — no manual field copying.
        """
        doc = self._docs.get(doc_id)
        return doc.to_meta() if doc is not None else None

    def get_docs(self, doc_ids: List[str]) -> Dict[str, Optional[Document]]:
        """
        Batch retrieval. Returns a dict mapping each requested ID to its
        Document (or None if not found). Preserves request order.
        """
        return {doc_id: self._docs.get(doc_id) for doc_id in doc_ids}

    def all_metas(self) -> List[DocMeta]:
        """Return metadata stubs for all documents in the store."""
        return [doc.to_meta() for doc in self._docs.values()]

    def all_docs(self) -> List[Document]:
        """Return all full documents in the store."""
        return list(self._docs.values())

    # ── Membership & iteration ─────────────────────────────────────────

    def contains(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._docs

    def __len__(self) -> int:
        return len(self._docs)

    def __iter__(self) -> Iterator[Document]:
        return iter(self._docs.values())

    @property
    def doc_ids(self) -> List[str]:
        """Ordered list of all document IDs currently in the store."""
        return list(self._docs.keys())

    # ── Introspection ──────────────────────────────────────────────────

    def stats(self) -> Dict[str, object]:
        """Lightweight summary for logging and monitoring."""
        if not self._docs:
            return {"doc_count": 0, "avg_content_tokens": 0.0, "total_content_tokens": 0}
        total_tokens = sum(d.estimated_tokens for d in self._docs.values())
        return {
            "doc_count": len(self._docs),
            "total_content_tokens": total_tokens,
            "avg_content_tokens": round(total_tokens / len(self._docs), 1),
        }

    # ── Snapshot / restore ─────────────────────────────────────────────

    def snapshot(self) -> "CorpusStore":
        """
        Return a shallow copy of this store (Documents are immutable Pydantic
        models, so shallow copy is safe and avoids the cost of deepcopy).
        """
        clone = CorpusStore()
        clone._docs = dict(self._docs)
        return clone

    @classmethod
    def restore(cls, snap: "CorpusStore") -> "CorpusStore":
        """Restore from a previously taken snapshot."""
        return snap.snapshot()

    def __repr__(self) -> str:
        return f"CorpusStore(docs={len(self._docs)})"