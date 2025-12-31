from unittest.mock import patch

from langchain_core.documents import Document

from ai.app_services import create_property_retriever
from vector_store.chroma_store import ChromaPropertyStore


def test_property_retriever_forced_listing_type_filters_results(tmp_path, monkeypatch):
    with patch.object(ChromaPropertyStore, "_create_embeddings", return_value=None):
        store = ChromaPropertyStore(persist_directory=str(tmp_path))

    docs = [
        Document(page_content="rent 1", metadata={"listing_type": "rent"}),
        Document(page_content="rent 2", metadata={"listing_type": "rent"}),
        Document(page_content="sale 1", metadata={"listing_type": "sale"}),
    ]

    class FakeInnerRetriever:
        def get_relevant_documents(self, query: str):
            return docs

    def fake_get_retriever(*, search_type: str, k: int, fetch_k: int, lambda_mult: float):
        return FakeInnerRetriever()

    monkeypatch.setattr(store, "get_retriever", fake_get_retriever)

    retriever = create_property_retriever(
        vector_store=store,
        k_results=10,
        center_lat=None,
        center_lon=None,
        radius_km=None,
        listing_type_filter="Rent",
    )

    results = retriever.get_relevant_documents("apartments in Warsaw")
    assert len(results) == 2
    assert {doc.metadata.get("listing_type") for doc in results} == {"rent"}

