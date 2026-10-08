"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        try:
            Path(self._settings.chroma_db_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self._settings.chroma_db_path
            )
            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Initialised ChromaDB collection '{}' with {} items",
                self._settings.chroma_collection_name,
                self._collection.count(),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise ChromaDB at {self._settings.chroma_db_path}"
            ) from exc

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects identical
        content even when files are renamed or re-uploaded.
        """
        result = self._collection.get(ids=[chunk_id], include=[])
        return bool(result["ids"])

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.

        Checks each chunk for duplicates before embedding. Skips duplicates
        silently and records the count in the returned IngestionResult.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Prepared chunks with text and metadata. Use DocumentChunker
            to produce these from raw files.

        Returns
        -------
        IngestionResult
            Summary with counts of ingested, skipped, and errored chunks.

        Notes
        -----
        Embeds in batches of 100 to avoid memory issues with large corpora.
        Uses upsert (not add) so re-ingestion of modified content updates
        existing chunks rather than raising an error.

        Interview talking point: batch processing with a configurable
        batch size is a production pattern that prevents OOM errors when
        ingesting large document sets.
        """
        result = IngestionResult()
        seen_ids: set[str] = set()
        document_sources: set[str] = set()

        for offset in range(0, len(chunks), 100):
            batch: list[DocumentChunk] = []
            for chunk in chunks[offset : offset + 100]:
                try:
                    if chunk.chunk_id in seen_ids or self.check_duplicate(chunk.chunk_id):
                        result.skipped += 1
                        continue
                    seen_ids.add(chunk.chunk_id)
                    batch.append(chunk)
                except Exception as exc:
                    result.errors.append(f"Chunk {chunk.chunk_id}: {exc}")

            if not batch:
                continue

            try:
                documents = [chunk.chunk_text for chunk in batch]
                embeddings = self._embeddings.embed_documents(documents)
                self._collection.upsert(
                    ids=[chunk.chunk_id for chunk in batch],
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=[chunk.metadata.to_dict() for chunk in batch],
                )
            except Exception as exc:
                for chunk in batch:
                    seen_ids.discard(chunk.chunk_id)
                    result.errors.append(f"Chunk {chunk.chunk_id}: {exc}")
                logger.error("Failed to ingest batch of {} chunks: {}", len(batch), exc)
                continue

            result.ingested += len(batch)
            for chunk in batch:
                source = chunk.metadata.source
                if source not in document_sources:
                    document_sources.add(source)
                    result.document_ids.append(source)

        logger.info(
            "Ingestion complete: {} ingested, {} skipped, {} errors",
            result.ingested,
            result.skipped,
            len(result.errors),
        )
        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Applies similarity threshold filtering — chunks below
        settings.similarity_threshold are excluded from results.

        Parameters
        ----------
        query_text : str
            The user query or rewritten query to retrieve against.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.retrieval_k.
        topic_filter : str, optional
            Restrict retrieval to a specific topic (e.g. 'LSTM').
            Maps to ChromaDB where-filter on metadata.topic.
        difficulty_filter : str, optional
            Restrict retrieval to a difficulty level.
            Maps to ChromaDB where-filter on metadata.difficulty.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by similarity score descending.
            Empty list if no chunks meet the similarity threshold.

        Interview talking point: returning an empty list (not hallucinating)
        when no relevant context exists is the hallucination guard. This is
        a critical production RAG pattern — the system must know what it
        does not know.
        """
        k = k if k is not None else self._settings.retrieval_k
        filters = []
        if topic_filter is not None:
            filters.append({"topic": topic_filter})
        if difficulty_filter is not None:
            filters.append({"difficulty": difficulty_filter})
        where_filter = (
            {"$and": filters} if len(filters) > 1 else filters[0] if filters else None
        )

        query_embedding = self._embeddings.embed_query(query_text)
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for chunk_id, document, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            score = 1 - distance
            if score >= self._settings.similarity_threshold:
                chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        chunk_text=document,
                        metadata=ChunkMetadata.from_dict(metadata),
                        score=score,
                    )
                )
        chunks.sort(key=lambda chunk: chunk.score, reverse=True)
        return chunks

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.

        Used by the UI to populate the document viewer panel.

        Returns
        -------
        list[dict]
            Each item contains: source (str), topic (str), chunk_count (int).
        """
        result = self._collection.get(include=["metadatas"])
        documents: dict[str, dict] = {}
        for metadata in result["metadatas"] or []:
            source = metadata["source"]
            if source not in documents:
                documents[source] = {
                    "source": source,
                    "topic": metadata["topic"],
                    "chunk_count": 0,
                }
            documents[source]["chunk_count"] += 1
        return [documents[source] for source in sorted(documents)]

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.

        Used by the document viewer to display document content.

        Parameters
        ----------
        source : str
            The source filename to retrieve chunks for.

        Returns
        -------
        list[DocumentChunk]
            All chunks from this source, ordered by their position
            in the original document.
        """
        result = self._collection.get(
            where={"source": source}, include=["documents", "metadatas"]
        )
        return [
            DocumentChunk(
                chunk_id=chunk_id,
                chunk_text=document,
                metadata=ChunkMetadata.from_dict(metadata),
            )
            for chunk_id, document, metadata in zip(
                result["ids"],
                result["documents"] or [],
                result["metadatas"] or [],
            )
        ]

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.

        Used by the UI to show corpus health at a glance.

        Returns
        -------
        dict
            Keys: total_chunks, topics (list), sources (list),
            bonus_topics_present (bool).
        """
        result = self._collection.get(include=["metadatas"])
        topics: set[str] = set()
        sources: set[str] = set()
        bonus_topics_present = False
        for metadata in result["metadatas"] or []:
            topics.add(metadata["topic"])
            sources.add(metadata["source"])
            if metadata.get("is_bonus", "false") in ("true", True):
                bonus_topics_present = True

        return {
            "total_chunks": len(result["ids"]),
            "topics": sorted(topics),
            "sources": sorted(sources),
            "bonus_topics_present": bonus_topics_present,
        }

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        result = self._collection.get(where={"source": source}, include=[])
        chunk_ids = result["ids"]
        if chunk_ids:
            self._collection.delete(ids=chunk_ids)
        return len(chunk_ids)
