from unittest.mock import patch

from vector_store.hybrid_retriever import AdvancedPropertyRetriever
from vector_store.chroma_store import ChromaPropertyStore
from langchain_core.documents import Document


def test_retriever_geo_radius_filters_docs(tmp_path):
    docs = [
        Document(page_content="Warsaw apt", metadata={"city": "Warsaw", "lat": 52.23, "lon": 21.01, "price": 5000}),
        Document(page_content="Krakow apt", metadata={"city": "Krakow", "lat": 50.06, "lon": 19.94, "price": 4400}),
    ]
    with patch.object(ChromaPropertyStore, "_create_embeddings", return_value=None):
        store = ChromaPropertyStore(persist_directory=str(tmp_path))
    retr = AdvancedPropertyRetriever(vector_store=store, center_lat=52.23, center_lon=21.01, radius_km=10.0)
    filtered = retr._filter_by_geo(docs)
    assert len(filtered) == 1
    assert filtered[0].metadata["city"] == "Warsaw"


def test_retriever_price_filter_skips_none_prices(tmp_path):
    docs = [
        Document(page_content="missing price", metadata={"price": None}),
        Document(page_content="ok price", metadata={"price": 5000}),
        Document(page_content="str price", metadata={"price": "4500"}),
        Document(page_content="bad price", metadata={"price": "n/a"}),
    ]
    with patch.object(ChromaPropertyStore, "_create_embeddings", return_value=None):
        store = ChromaPropertyStore(persist_directory=str(tmp_path))
    retr = AdvancedPropertyRetriever(vector_store=store, min_price=4600)
    filtered = retr._filter_by_price(docs)
    assert [d.page_content for d in filtered] == ["ok price"]


def test_retriever_sorting_handles_none_and_non_numeric(tmp_path):
    docs = [
        Document(page_content="a", metadata={"price_per_sqm": 20}),
        Document(page_content="b", metadata={"price_per_sqm": None}),
        Document(page_content="c", metadata={"price_per_sqm": "n/a"}),
        Document(page_content="d", metadata={"price_per_sqm": 10}),
    ]
    with patch.object(ChromaPropertyStore, "_create_embeddings", return_value=None):
        store = ChromaPropertyStore(persist_directory=str(tmp_path))
    retr = AdvancedPropertyRetriever(vector_store=store, sort_by="price_per_sqm", sort_ascending=True)
    sorted_docs = retr._sort_results(docs)
    assert [d.page_content for d in sorted_docs[:2]] == ["d", "a"]
