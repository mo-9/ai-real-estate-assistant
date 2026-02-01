import inspect
import json
import logging
import uuid
from typing import Annotated, Any, Optional

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models import BaseChatModel

from agents.hybrid_agent import create_hybrid_agent
from ai.memory import get_session_history
from api.chat_sources import serialize_chat_sources, serialize_web_sources
from api.dependencies import get_llm, get_vector_store
from api.models import ChatRequest, ChatResponse
from config.settings import get_settings
from services.router import RouteTarget, QueryRouter
from services.search import build_tavily_search_service
from utils.sanitization import sanitize_chat_message, sanitize_intermediate_steps

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(
    request: ChatRequest,
    req: Request,
    llm: Annotated[BaseChatModel, Depends(get_llm)],
    store: Annotated[Optional[Any], Depends(get_vector_store)],
):
    """
    Process a chat message using the hybrid agent with session persistence.
    """
    # Get request_id from middleware for correlation
    request_id = getattr(req.state, "request_id", None) or str(uuid.uuid4())
    # Sanitize chat message to prevent injection attacks
    try:
        sanitized_message = sanitize_chat_message(request.message)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from None

    try:
        settings = get_settings()
        sources_max_items = max(0, int(settings.chat_sources_max_items))
        sources_max_content_chars = max(0, int(settings.chat_source_content_max_chars))
        sources_max_total_bytes = max(0, int(settings.chat_sources_max_total_bytes))

        # Handle Session ID
        session_id = request.session_id
        if not session_id:
            session_id = str(uuid.uuid4())

        # Initialize Memory with Persistence
        chat_history = get_session_history(session_id)
        memory = ConversationBufferMemory(
            chat_memory=chat_history,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        router = QueryRouter()
        decision = router.route(sanitized_message)

        # Create Agent (local property retrieval only)
        agent_kwargs = {
            "llm": llm,
            "retriever": store.get_retriever() if store else None,
            "memory": memory,
            "internet_enabled": False,
            "searxng_url": None,
            "web_search_max_results": int(getattr(settings, "web_search_max_results", 5)),
            "web_fetch_timeout_seconds": float(getattr(settings, "web_fetch_timeout_seconds", 10)),
            "web_fetch_max_bytes": int(getattr(settings, "web_fetch_max_bytes", 300_000)),
            "web_allowlist_domains": list(getattr(settings, "web_allowlist_domains", []) or []),
        }
        sig = inspect.signature(create_hybrid_agent)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            filtered_kwargs = agent_kwargs
        else:
            filtered_kwargs = {k: v for k, v in agent_kwargs.items() if k in sig.parameters}
        agent = create_hybrid_agent(**filtered_kwargs) if store else None

        def _build_tavily_payload() -> tuple[str, list[dict[str, Any]], bool, list[dict[str, Any]]]:
            try:
                tavily = build_tavily_search_service()
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=str(exc),
                ) from None

            results = tavily.search(sanitized_message)
            answer = tavily.answer_with_llm(llm, sanitized_message, results)
            steps = [
                {
                    "tool": "tavily_search",
                    "input": {"query": sanitized_message},
                    "output": results,
                }
            ]
            sources, sources_truncated = serialize_web_sources(
                results,
                max_items=sources_max_items,
                max_content_chars=sources_max_content_chars,
                max_total_bytes=sources_max_total_bytes,
            )
            return answer, sources, sources_truncated, steps

        if request.stream:
            async def event_generator():
                sources_payload: dict[str, object] = {
                    "sources": [],
                    "sources_truncated": False,
                    "session_id": session_id,
                    "request_id": request_id,
                }

                try:
                    # Add timeout protection for streaming (5 minutes)
                    with anyio.fail_after(300):
                        if decision.target in {RouteTarget.WEB, RouteTarget.HYBRID}:
                            (
                                web_answer,
                                web_sources,
                                web_sources_truncated,
                                web_steps,
                            ) = _build_tavily_payload()
                            combined_answer = web_answer
                            combined_sources = web_sources
                            combined_truncated = web_sources_truncated
                            combined_steps = list(web_steps)

                            if decision.target == RouteTarget.HYBRID:
                                if not agent:
                                    raise HTTPException(
                                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                        detail="Vector store unavailable",
                                    )
                                local_result = agent.process_query(sanitized_message)
                                local_answer = str(local_result.get("answer", "")).strip()
                                combined_answer = (
                                    f"{local_answer}\n\nWeb insights:\n{web_answer}"
                                    if local_answer
                                    else web_answer
                                )
                                local_sources, local_truncated = serialize_chat_sources(
                                    local_result.get("source_documents") or [],
                                    max_items=sources_max_items,
                                    max_content_chars=sources_max_content_chars,
                                    max_total_bytes=sources_max_total_bytes,
                                )
                                combined_sources = local_sources + web_sources
                                combined_truncated = local_truncated or web_sources_truncated
                                combined_steps = (
                                    list(local_result.get("intermediate_steps") or []) + web_steps
                                )

                            if combined_answer:
                                yield f"data: {json.dumps({'content': combined_answer}, ensure_ascii=False, default=str)}\n\n"
                            sources_payload["sources"] = combined_sources
                            sources_payload["sources_truncated"] = combined_truncated
                            if request.include_intermediate_steps and combined_steps:
                                sources_payload["intermediate_steps"] = sanitize_intermediate_steps(
                                    combined_steps
                                )
                        else:
                            if not agent:
                                raise HTTPException(
                                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                    detail="Vector store unavailable",
                                )
                            async for chunk in agent.astream_query(sanitized_message):
                                yield f"data: {chunk}\n\n"
                            if hasattr(agent, "get_sources_for_query"):
                                try:
                                    docs = agent.get_sources_for_query(sanitized_message)
                                    sources, sources_truncated = serialize_chat_sources(
                                        docs,
                                        max_items=sources_max_items,
                                        max_content_chars=sources_max_content_chars,
                                        max_total_bytes=sources_max_total_bytes,
                                    )
                                    sources_payload["sources"] = sources
                                    sources_payload["sources_truncated"] = sources_truncated
                                except Exception:
                                    sources_payload["sources"] = []
                                    sources_payload["sources_truncated"] = False

                        yield "event: meta\n"
                        yield f"data: {json.dumps(sources_payload)}\n\n"
                        yield "data: [DONE]\n\n"
                except TimeoutError:
                    # Send timeout error event
                    error_payload = {"error": "Stream timeout exceeded", "request_id": request_id}
                    yield "event: error\n"
                    yield f"data: {json.dumps(error_payload)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    # Log error and send error event for client recovery
                    logger.error(f"Stream error (request_id={request_id}): {e}")
                    error_payload = {"error": str(e), "request_id": request_id}
                    yield "event: error\n"
                    yield f"data: {json.dumps(error_payload)}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    # Always ensure stream terminates
                    pass

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        if decision.target in {RouteTarget.WEB, RouteTarget.HYBRID}:
            web_answer, web_sources, web_sources_truncated, web_steps = _build_tavily_payload()
            answer = web_answer
            sources = web_sources
            sources_truncated = web_sources_truncated
            intermediate_steps = list(web_steps)

            if decision.target == RouteTarget.HYBRID:
                if not agent:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Vector store unavailable",
                    )
                local_result = agent.process_query(sanitized_message)
                local_answer = str(local_result.get("answer", "")).strip()
                answer = f"{local_answer}\n\nWeb insights:\n{web_answer}" if local_answer else web_answer
                local_sources, local_truncated = serialize_chat_sources(
                    local_result.get("source_documents") or [],
                    max_items=sources_max_items,
                    max_content_chars=sources_max_content_chars,
                    max_total_bytes=sources_max_total_bytes,
                )
                sources = local_sources + web_sources
                sources_truncated = local_truncated or web_sources_truncated
                intermediate_steps = (
                    list(local_result.get("intermediate_steps") or []) + web_steps
                )
        else:
            if not agent:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Vector store unavailable",
                )
            result = agent.process_query(sanitized_message)

            answer = result.get("answer", "")
            sources, sources_truncated = serialize_chat_sources(
                result.get("source_documents") or [],
                max_items=sources_max_items,
                max_content_chars=sources_max_content_chars,
                max_total_bytes=sources_max_total_bytes,
            )
            intermediate_steps = list(result.get("intermediate_steps") or [])

        # Sanitize intermediate steps if requested
        if request.include_intermediate_steps:
            intermediate_steps = (
                sanitize_intermediate_steps(intermediate_steps) if intermediate_steps else None
            )
        else:
            intermediate_steps = None

        return ChatResponse(
            response=answer,
            sources=sources,
            sources_truncated=sources_truncated,
            session_id=session_id,
            intermediate_steps=intermediate_steps,
        )

    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}",
        ) from e
