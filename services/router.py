from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agents.query_analyzer import QueryAnalysis, QueryAnalyzer, QueryIntent, Tool


class RouteTarget(str, Enum):
    LOCAL = "local"
    WEB = "web"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class RouteDecision:
    target: RouteTarget
    analysis: QueryAnalysis


class QueryRouter:
    def __init__(self, analyzer: QueryAnalyzer | None = None) -> None:
        self.analyzer = analyzer or QueryAnalyzer()

    def route(self, query: str) -> RouteDecision:
        analysis = self.analyzer.analyze(query)
        wants_web = bool(
            analysis.requires_external_data or Tool.WEB_SEARCH in analysis.tools_needed
        )
        wants_local = bool(
            analysis.intent
            in {
                QueryIntent.SIMPLE_RETRIEVAL,
                QueryIntent.FILTERED_SEARCH,
                QueryIntent.COMPARISON,
                QueryIntent.ANALYSIS,
                QueryIntent.RECOMMENDATION,
                QueryIntent.CONVERSATION,
            }
            or Tool.RAG_RETRIEVAL in analysis.tools_needed
        )

        if wants_web and wants_local:
            target = RouteTarget.HYBRID
        elif wants_web:
            target = RouteTarget.WEB
        else:
            target = RouteTarget.LOCAL
        return RouteDecision(target=target, analysis=analysis)
