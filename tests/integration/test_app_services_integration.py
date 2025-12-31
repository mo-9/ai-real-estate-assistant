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

    captured = {}

    def fake_get_retriever(
        *,
        search_type: str,
        k: int,
        fetch_k: int,
        lambda_mult: float,
        filter=None,
        **kwargs,
    ):
        captured["search_type"] = search_type
        captured["k"] = k
        captured["fetch_k"] = fetch_k
        captured["lambda_mult"] = lambda_mult
        captured["filter"] = filter
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

    results = retriever.get_relevant_documents("apartments")
    assert len(results) == 2
    assert {doc.metadata.get("listing_type") for doc in results} == {"rent"}
    assert captured["filter"] == {"listing_type": "rent"}
    assert captured["k"] == 20
    assert captured["fetch_k"] == 20


def test_property_retriever_geo_radius_filters_results(tmp_path, monkeypatch):
    with patch.object(ChromaPropertyStore, "_create_embeddings", return_value=None):
        store = ChromaPropertyStore(persist_directory=str(tmp_path))

    docs = [
        Document(page_content="near", metadata={"lat": 52.23, "lon": 21.01}),
        Document(page_content="far", metadata={"lat": 50.06, "lon": 19.94}),
    ]

    class FakeInnerRetriever:
        def get_relevant_documents(self, query: str):
            return docs

    def fake_get_retriever(**kwargs):
        return FakeInnerRetriever()

    monkeypatch.setattr(store, "get_retriever", fake_get_retriever)

    retriever = create_property_retriever(
        vector_store=store,
        k_results=10,
        center_lat=52.23,
        center_lon=21.01,
        radius_km=10.0,
        listing_type_filter=None,
    )

    results = retriever.get_relevant_documents("apartments")
    assert [d.page_content for d in results] == ["near"]
