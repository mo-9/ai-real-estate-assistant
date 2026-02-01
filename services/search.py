from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from langchain_core.language_models import BaseChatModel

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TavilySearchConfig:
    api_key: str
    max_results: int = 5
    search_depth: str = "basic"
    max_content_chars: int = 2000


class TavilySearchService:
    def __init__(self, config: TavilySearchConfig) -> None:
        self.config = config

    def search(self, query: str) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []
        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise RuntimeError(
                "Tavily SDK is not installed. Add `tavily-python` to requirements.txt."
            ) from exc

        client = TavilyClient(api_key=self.config.api_key)
        response = client.search(
            query=query,
            max_results=max(1, int(self.config.max_results)),
            search_depth=self.config.search_depth,
            include_answer=False,
            include_raw_content=False,
        )
        results = response.get("results", []) if isinstance(response, dict) else []
        return [self._normalize_result(r) for r in results if isinstance(r, dict)]

    def _normalize_result(self, raw: dict[str, Any]) -> dict[str, Any]:
        content = str(raw.get("content") or raw.get("snippet") or "")
        content = content[: max(0, int(self.config.max_content_chars))]
        return {
            "title": raw.get("title") or "",
            "url": raw.get("url") or "",
            "snippet": content.strip(),
            "content": content.strip(),
            "provider": "tavily",
            "score": raw.get("score"),
        }

    def answer_with_llm(
        self, llm: BaseChatModel, query: str, results: Iterable[dict[str, Any]]
    ) -> str:
        sources = list(results)
        if not sources:
            return "I couldn't find relevant web results."

        context = "\n\n".join(
            f"[{idx + 1}] {item.get('title', '')}\nURL: {item.get('url', '')}\n"
            f"Content: {item.get('content', '')}"
            for idx, item in enumerate(sources)
        )
        prompt = (
            "Use ONLY the web sources below to answer the question.\n"
            "If the answer cannot be verified, say you don't know.\n"
            "Cite each fact like [n] where n is the source number.\n\n"
            f"Sources:\n{context}\n\nQuestion: {query}"
        )
        msg = llm.invoke(prompt)
        return msg.content if hasattr(msg, "content") else str(msg)


def build_tavily_search_service(
    max_results: Optional[int] = None,
    search_depth: str = "basic",
    max_content_chars: int = 2000,
) -> TavilySearchService:
    settings = get_settings()
    api_key = (settings.tavily_api_key or "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured.")

    config = TavilySearchConfig(
        api_key=api_key,
        max_results=int(max_results or settings.web_search_max_results),
        search_depth=search_depth,
        max_content_chars=max_content_chars,
    )
    return TavilySearchService(config=config)
