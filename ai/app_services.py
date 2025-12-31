from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from agents.hybrid_agent import create_hybrid_agent
from models.provider_factory import ModelProviderFactory
from vector_store.hybrid_retriever import create_retriever


def build_forced_filters(listing_type_filter: Optional[str]) -> Optional[Dict[str, Any]]:
    if listing_type_filter == "Rent":
        return {"listing_type": "rent"}
    if listing_type_filter == "Sale":
        return {"listing_type": "sale"}
    return None


def create_llm(
    *,
    provider_name: str,
    model_id: str,
    temperature: float,
    max_tokens: int,
    streaming: bool,
    callbacks: Optional[Sequence[BaseCallbackHandler]] = None,
) -> BaseChatModel:
    return ModelProviderFactory.create_model(
        model_id=model_id,
        provider_name=provider_name,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        callbacks=list(callbacks) if callbacks else None,
    )


def create_property_retriever(
    *,
    vector_store: Any,
    k_results: int,
    center_lat: Optional[float],
    center_lon: Optional[float],
    radius_km: Optional[float],
    listing_type_filter: Optional[str],
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    sort_by: Optional[str] = None,
    sort_ascending: bool = True,
) -> BaseRetriever:
    return create_retriever(
        vector_store=vector_store,
        k=k_results,
        search_type="mmr",
        center_lat=center_lat,
        center_lon=center_lon,
        radius_km=radius_km,
        min_price=min_price,
        max_price=max_price,
        sort_by=sort_by,
        sort_ascending=sort_ascending,
        forced_filters=build_forced_filters(listing_type_filter),
    )


def create_conversation_chain(
    *,
    llm: BaseChatModel,
    retriever: BaseRetriever,
    verbose: bool,
) -> ConversationalRetrievalChain:
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=verbose,
    )


def create_hybrid_agent_instance(
    *,
    llm: BaseChatModel,
    retriever: BaseRetriever,
    verbose: bool,
) -> Any:
    return create_hybrid_agent(
        llm=llm,
        retriever=retriever,
        use_tools=True,
        verbose=verbose,
    )

