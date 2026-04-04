"""
ChronoVeritas — Corpus store: load and serve documents.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from env.models import DocMeta, Document


class CorpusStore:
    """In-memory document store loaded from task corpus list."""

    def __init__(self) -> None:
        self._docs: Dict[str, Document] = {}

    def load_from_task_corpus(self, corpus: List[Document]) -> None:
        """Load documents directly from a task's corpus list."""
        self._docs.clear()
        for doc in corpus:
            self._docs[doc.doc_id] = doc

    def get_doc(self, doc_id: str) -> Optional[Document]:
        """Retrieve a full document by ID."""
        return self._docs.get(doc_id)

    def get_meta(self, doc_id: str) -> Optional[DocMeta]:
        """Retrieve only the metadata for a document."""
        doc = self._docs.get(doc_id)
        if doc is None:
            return None
        return DocMeta(
            doc_id=doc.doc_id,
            title=doc.title,
            source=doc.source,
            timestamp=doc.timestamp,
            tags=doc.tags,
            snippet=doc.snippet,
        )

    def all_metas(self) -> List[DocMeta]:
        """Return metadata for all documents in the store."""
        return [self.get_meta(did) for did in self._docs]  # type: ignore

    def all_docs(self) -> List[Document]:
        return list(self._docs.values())

    @property
    def doc_ids(self) -> List[str]:
        return list(self._docs.keys())
