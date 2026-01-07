import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from vector_store.reranker import StrategicReranker

@pytest.fixture
def strategic_reranker():
    # Mock valuation model not strictly needed for basic strategy tests 
    # as we mocked the behavior in the reranker for now (simple heuristic fallback)
    return StrategicReranker()

def test_investor_strategy(strategic_reranker):
    docs = [
        Document(page_content="Prop A", metadata={"title": "A", "price": 100000, "area_sqm": 50}), # 2000/sqm
        Document(page_content="Prop B", metadata={"title": "B", "price": 200000, "area_sqm": 50}), # 4000/sqm
    ]
    
    # Investor strategy should favor Prop A (lower price/sqm)
    results = strategic_reranker.rerank_with_strategy(
        query="apartment",
        documents=docs,
        strategy="investor",
        initial_scores=[1.0, 1.0]
    )
    
    # Prop A should be first and have higher score
    assert results[0][0].metadata["title"] == "A"
    assert results[0][1] > results[1][1]

def test_family_strategy(strategic_reranker):
    docs = [
        Document(page_content="Small", metadata={"title": "Small", "rooms": 1, "has_garden": False}),
        Document(page_content="Big", metadata={"title": "Big", "rooms": 4, "has_garden": True}),
    ]
    
    results = strategic_reranker.rerank_with_strategy(
        query="home",
        documents=docs,
        strategy="family",
        initial_scores=[1.0, 1.0]
    )
    
    # Big should be first
    assert results[0][0].metadata["title"] == "Big"
    assert results[0][1] > results[1][1]

def test_bargain_strategy(strategic_reranker):
    docs = [
        Document(page_content="Expensive", metadata={"title": "Exp", "price": 500000}),
        Document(page_content="Cheap", metadata={"title": "Chp", "price": 150000}),
    ]
    
    results = strategic_reranker.rerank_with_strategy(
        query="home",
        documents=docs,
        strategy="bargain",
        initial_scores=[1.0, 1.0]
    )
    
    assert results[0][0].metadata["title"] == "Chp"

def test_balanced_strategy(strategic_reranker):
    # Balanced just calls base rerank (which does nothing special here since no query match or metadata)
    docs = [
        Document(page_content="Prop A", metadata={"title": "A"}),
    ]
    
    results = strategic_reranker.rerank_with_strategy(
        query="foo",
        documents=docs,
        strategy="balanced"
    )
    
    assert len(results) == 1
