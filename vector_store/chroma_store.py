"""
ChromaDB vector store implementation with persistence.

This module provides a persistent vector store for property embeddings
using ChromaDB with FastEmbed embeddings.
"""

import platform
from datetime import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, cast

import pandas as pd

import streamlit as st
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from config.settings import settings
from data.schemas import Property, PropertyCollection

_ChromaSettings: Any = None
try:
    from chromadb.config import Settings as _ChromaSettings
except Exception:
    pass

_FastEmbedEmbeddings: Any = None
try:
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings as _FastEmbedEmbeddings
except Exception:
    pass

# Configure logger
logger = logging.getLogger(__name__)


class ChromaPropertyStore:
    """
    Persistent vector store for property data using ChromaDB.

    This class handles:
    - Property embedding and storage
    - Semantic search
    - Metadata filtering
    - Persistence to disk
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "properties",
        embedding_model: str = "BAAI/bge-small-en-v1.5"
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory for persistent storage (default: ./chroma_db)
            collection_name: Name of the collection
            embedding_model: FastEmbed model to use
        """
        # Set default persist directory
        if persist_directory is None:
            persist_directory = os.path.join(os.getcwd(), "chroma_db")

        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name

        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize embeddings
        self.embeddings: Optional[Embeddings] = self._create_embeddings(embedding_model)

        # Initialize or load vector store
        self.vector_store: Optional[Chroma] = self._initialize_vector_store()

        # Text splitter for long descriptions
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        self._documents: List[Document] = []
        self._doc_ids: Set[str] = set()

    @st.cache_resource
    def _create_embeddings(_self, model_name: str) -> Optional[Embeddings]:
        try:
            is_windows = platform.system().lower() == "windows"
            force_fastembed = os.getenv("CHROMA_FORCE_FASTEMBED") == "1" or os.getenv("FORCE_FASTEMBED") == "1"
            if _FastEmbedEmbeddings is not None and not is_windows:
                return cast(Embeddings, _FastEmbedEmbeddings(model_name=model_name))
            if _FastEmbedEmbeddings is not None and is_windows and force_fastembed:
                return cast(Embeddings, _FastEmbedEmbeddings(model_name=model_name))
            if _FastEmbedEmbeddings is not None and is_windows:
                st.warning("FastEmbed is disabled on Windows for stability. Set CHROMA_FORCE_FASTEMBED=1 to force enable.")
        except Exception as e:
            st.warning(f"FastEmbed initialization failed: {e}")

        try:
            from config import settings
            if settings.openai_api_key:
                from langchain_openai import OpenAIEmbeddings
                return cast(Embeddings, OpenAIEmbeddings())
        except Exception as e:
            st.warning(f"OpenAI embeddings unavailable: {e}")

        return None

    def _initialize_vector_store(self) -> Optional[Chroma]:
        """Initialize or load existing ChromaDB vector store."""
        if self.embeddings is None:
            st.warning("Embeddings unavailable; vector store features are disabled")
            return None

        try:
            client_settings = None
            if _ChromaSettings is not None:
                client_settings = _ChromaSettings(anonymized_telemetry=False)

            force_persist = os.getenv("CHROMA_FORCE_PERSIST") == "1"
            use_persist = force_persist or bool(settings.vector_persist_enabled)

            if use_persist:
                vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=str(self.persist_directory),
                    client_settings=client_settings,
                )

                # Check if collection has any documents
                collection_stats = vector_store._collection.count()
                logger.info(f"Loaded existing ChromaDB collection with {collection_stats} documents")
                try:
                    existing = vector_store._collection.get(include=[], limit=None)
                    for _id in existing.get("ids", []) or []:
                        self._doc_ids.add(str(_id))
                except Exception:
                    pass
                return vector_store
            else:
                # Directly use in-memory on platforms where persistence is disabled
                vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                )
                logger.info("Using in-memory Chroma vector store (persistence disabled for this platform)")
                st.warning("Persistent vector store unavailable; using in-memory store")
                try:
                    existing = vector_store._collection.get(include=[], limit=None)
                    for _id in existing.get("ids", []) or []:
                        self._doc_ids.add(str(_id))
                except Exception:
                    pass
                return vector_store

        except BaseException as e:
            logger.warning(f"Persistent Chroma init failed: {e}")

            # Fallback: in-memory Chroma (no persistence)
            try:
                vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                )
                logger.info("Initialized in-memory Chroma vector store (no persistence)")
                st.warning("Persistent vector store unavailable; using in-memory store")
                try:
                    existing = vector_store._collection.get(include=[], limit=None)
                    for _id in existing.get("ids", []) or []:
                        self._doc_ids.add(str(_id))
                except Exception:
                    pass
                return vector_store
            except BaseException as e2:
                logger.error(f"In-memory Chroma init failed: {e2}")
                raise

    def property_to_document(self, prop: Property) -> Document:
        """
        Convert Property to LangChain Document.

        Args:
            property: Property instance

        Returns:
            Document with property text and metadata
        """
        # Create comprehensive text representation
        text = prop.to_search_text()

        # Create metadata (must be JSON-serializable)
        def _nf(x: Any) -> Optional[float]:
            return float(x) if (x is not None and not pd.isna(x)) else None

        def _ni(x: Any) -> Optional[int]:
            if x is None or pd.isna(x):
                return None
            try:
                return int(float(x))
            except (TypeError, ValueError):
                return None

        raw_energy = getattr(prop, "energy_cert", None)
        energy_cert = str(raw_energy).strip() if raw_energy is not None else None
        if energy_cert == "":
            energy_cert = None

        metadata = {
            "id": prop.id or "unknown",
            "country": getattr(prop, "country", None),
            "region": getattr(prop, "region", None),
            "city": prop.city,
            "district": getattr(prop, "district", None),
            "price": _nf(prop.price),
            "rooms": (_nf(prop.rooms) or 0.0),
            "bathrooms": (_nf(prop.bathrooms) or 0.0),
            "price_per_sqm": _nf(getattr(prop, "price_per_sqm", None)),
            "currency": getattr(prop, "currency", None),
            "has_parking": prop.has_parking,
            "has_garden": prop.has_garden,
            "has_pool": prop.has_pool,
            "has_garage": prop.has_garage,
            "has_elevator": prop.has_elevator,
            "property_type": prop.property_type.value if hasattr(prop.property_type, "value") else str(prop.property_type),
            "listing_type": prop.listing_type.value if hasattr(prop.listing_type, "value") else str(prop.listing_type),
            "source_url": prop.source_url or "",
            "lat": _nf(getattr(prop, "latitude", None)),
            "lon": _nf(getattr(prop, "longitude", None)),
            "year_built": _ni(getattr(prop, "year_built", None)),
            "energy_cert": energy_cert,
        }

        # Add optional fields if present
        if prop.neighborhood:
            metadata["neighborhood"] = prop.neighborhood

        if prop.area_sqm is not None and not pd.isna(prop.area_sqm):
            metadata["area_sqm"] = float(prop.area_sqm)

        if prop.price_per_sqm is not None and not pd.isna(prop.price_per_sqm):
            metadata["price_per_sqm"] = float(prop.price_per_sqm)

        if prop.negotiation_rate:
            metadata["negotiation_rate"] = prop.negotiation_rate.value if hasattr(prop.negotiation_rate, "value") else str(prop.negotiation_rate)

        # Sanitize metadata: only primitives (str, int, float, bool, None); convert datetimes
        def _sanitize_val(v: Any) -> Any:
            try:
                if v is None:
                    return None
                if isinstance(v, (str, int, float, bool)):
                    if isinstance(v, float):
                        return None if (pd.isna(v) or v != v) else float(v)
                    return v
                if isinstance(v, (datetime, pd.Timestamp)):
                    return v.isoformat()
                # numpy types
                if hasattr(v, "item"):
                    return _sanitize_val(v.item())
                # lists/dicts or other complex types are not allowed in Chroma metadata
                return None
            except Exception:
                return None

        sanitized = {}
        for k, v in metadata.items():
            sv = _sanitize_val(v)
            if sv is not None or v is None:
                sanitized[k] = sv

        metadata = sanitized

        return Document(
            page_content=text,
            metadata=metadata
        )

    def add_properties(
        self,
        properties: List[Property],
        batch_size: int = 100
    ) -> int:
        """
        Add properties to the vector store.

        Args:
            properties: List of Property instances
            batch_size: Number of properties to process at once

        Returns:
            Number of properties added
        """
        documents: List[Document] = []

        for prop in properties:
            try:
                doc = self.property_to_document(prop)
                doc_id = str(doc.metadata.get("id", ""))
                if doc_id:
                    if doc_id in self._doc_ids:
                        continue
                    self._doc_ids.add(doc_id)
                documents.append(doc)
                self._documents.append(doc)
            except Exception as e:
                logger.warning(f"Skipping property {prop.id}: {e}")
                continue

        if not documents:
            logger.warning("No valid documents to add")
            return 0

        # If vector store is unavailable, keep documents in fallback cache only
        if self.vector_store is None:
            logger.info(f"Vector store disabled; cached {len(documents)} properties in memory")
            return len(documents)

        # Add documents in batches
        total_added = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            try:
                ids = [str(d.metadata.get("id", f"doc-{i+j}")) for j, d in enumerate(batch)]
                self.vector_store.add_documents(batch, ids=ids)
                total_added += len(batch)
                logger.info(f"Added batch {i // batch_size + 1}: {len(batch)} properties")

            except Exception as e:
                logger.error(f"Error adding batch: {e}")
                continue

        if total_added > 0:
            logger.info(f"Total properties added to vector store: {total_added}")
            return total_added
        else:
            logger.info(f"Stored {len(documents)} properties in fallback cache")
            return len(documents)

    def add_property_collection(
        self,
        collection: PropertyCollection,
        replace_existing: bool = False
    ) -> int:
        """
        Add a PropertyCollection to the vector store.

        Args:
            collection: PropertyCollection instance
            replace_existing: Whether to clear existing data first

        Returns:
            Number of properties added
        """
        if replace_existing:
            self.clear()

        return self.add_properties(collection.properties)

    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[tuple[Document, float]]:
        """
        Search for properties by semantic similarity.

        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional search parameters

        Returns:
            List of (Document, score) tuples
        """
        try:
            if self.vector_store is not None:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter,
                    **kwargs
                )
                return results
            else:
                raise RuntimeError("no_vector_store")
        except Exception as e:
            try:
                q = [t for t in query.lower().split() if t]
                scored: List[tuple[Document, float]] = []
                for d in self._documents:
                    txt = d.page_content.lower()
                    s = float(sum(1 for t in q if t in txt))
                    if s > 0:
                        scored.append((d, s))
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored[:k]
            except Exception:
                logger.error(f"Search error: {e}")
                return []

    def search_by_metadata(
        self,
        city: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rooms: Optional[float] = None,
        has_parking: Optional[bool] = None,
        k: int = 5
    ) -> List[Document]:
        """
        Search properties by metadata filters.

        Args:
            city: Filter by city
            min_price: Minimum price
            max_price: Maximum price
            min_rooms: Minimum number of rooms
            has_parking: Filter by parking availability
            k: Number of results

        Returns:
            List of matching documents
        """
        if self.vector_store is None:
            return []

        filter_dict: Dict[str, Any] = {}

        if city:
            filter_dict["city"] = city

        if has_parking is not None:
            filter_dict["has_parking"] = has_parking

        # Note: ChromaDB has limited support for range queries
        # For complex filtering, retrieve more results and filter in Python
        results = self.vector_store.similarity_search(
            query="",  # Empty query for metadata-only search
            k=k * 5,  # Retrieve more for filtering
            filter=filter_dict if filter_dict else None
        )

        # Apply additional filters
        filtered = []
        for doc in results:
            metadata = doc.metadata

            # Price filters
            if min_price is not None and metadata.get("price", 0) < min_price:
                continue
            if max_price is not None and metadata.get("price", float('inf')) > max_price:
                continue

            # Rooms filter
            if min_rooms is not None and metadata.get("rooms", 0) < min_rooms:
                continue

            filtered.append(doc)

            if len(filtered) >= k:
                break

        return filtered

    def get_retriever(
        self,
        search_type: str = "mmr",
        k: int = 5,
        fetch_k: int = 20,
        **kwargs: Any
    ) -> BaseRetriever:
        """
        Get a LangChain retriever for this vector store.

        Args:
            search_type: Type of search ('similarity', 'mmr', 'similarity_score_threshold')
            k: Number of documents to return
            fetch_k: Number of documents to fetch for MMR
            **kwargs: Additional retriever parameters

        Returns:
            LangChain retriever instance
        """
        stats = self.get_stats()
        total = stats.get("total_documents", 0)
        if self.vector_store is not None and total > 0:
            return self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs={
                    "k": k,
                    "fetch_k": fetch_k,
                    **kwargs
                }
            )
        else:
            class FallbackRetriever(BaseRetriever):
                docs: List[Document]
                kk: int
                class Config:
                    arbitrary_types_allowed = True

                def _get_relevant_documents(
                    self,
                    query: str,
                    *,
                    run_manager: Optional[CallbackManagerForRetrieverRun] = None,
                ) -> List[Document]:
                    q = [t for t in query.lower().split() if t]
                    scored: List[tuple[Document, float]] = []
                    for d in self.docs:
                        txt = d.page_content.lower()
                        s = float(sum(1 for t in q if t in txt))
                        if s > 0:
                            scored.append((d, s))
                    scored.sort(key=lambda x: x[1], reverse=True)
                    return [d for d, _s in scored[:self.kk]]
            return FallbackRetriever(docs=self._documents, kk=k)

    def clear(self) -> None:
        """Clear all documents from the vector store."""
        try:
            if self.vector_store is not None:
                self.vector_store.delete_collection()
                self.vector_store = self._initialize_vector_store()
            self._documents = []
            self._doc_ids = set()
            logger.info("Vector store cleared")

        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            self._documents = []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with store statistics
        """
        try:
            if self.vector_store is not None:
                count = self.vector_store._collection.count()
                if count == 0:
                    count = len(self._documents)
            else:
                count = len(self._documents)

            emb_cls = type(self.embeddings).__name__ if self.embeddings is not None else "None"
            if "OpenAIEmbeddings" in emb_cls:
                emb_provider = "openai"
                emb_model = getattr(self.embeddings, "model", "openai")
            elif "FastEmbedEmbeddings" in emb_cls:
                emb_provider = "fastembed"
                emb_model = getattr(self.embeddings, "model_name", "BAAI/bge-small-en-v1.5")
            else:
                emb_provider = "none"
                emb_model = "none"

            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory),
                "embedding_model": emb_model,
                "embedding_provider": emb_provider,
            }
        except Exception as e:
            return {"error": str(e), "total_documents": len(self._documents)}

    def delete_by_source(self, source_url: str) -> None:
        """
        Delete all properties from a specific source.

        Args:
            source_url: Source URL to filter by
        """
        try:
            if self.vector_store is None:
                return
            self.vector_store.delete(
                filter={"source_url": source_url}
            )
            logger.info(f"Deleted properties from source: {source_url}")

        except Exception as e:
            logger.error(f"Error deleting by source: {e}")

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"<ChromaPropertyStore: {stats.get('total_documents', 0)} documents, "
            f"collection='{self.collection_name}'>"
        )


@st.cache_resource
def get_vector_store(
    persist_directory: Optional[str] = None,
    collection_name: str = "properties",
    embedding_model: Optional[str] = None,
) -> ChromaPropertyStore:
    """
    Get cached vector store instance.

    Args:
        persist_directory: Directory for persistent storage
        collection_name: Collection name
        embedding_model: Embedding model identifier

    Returns:
        ChromaPropertyStore instance
    """
    return ChromaPropertyStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model or settings.embedding_model,
    )
