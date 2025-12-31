# Architecture (V3)

## System Overview
- Streamlit UI (model selection, settings, chat, Expert Panel)
- Query Analyzer (intent, filters)
- Hybrid processing: simple → RAG, complex → Agent + Tools
- Tools: mortgage, comparison, analysis
- Retriever: ChromaDB (semantic + keyword, MMR), AdvancedPropertyRetriever (geo radius, price filters, sorting)
- Reranker (multi-signal)
- Response formatter

## Technology Stack
- UI: Streamlit 1.37+
- AI Orchestration: LangChain 0.2+
- Providers: OpenAI, Anthropic, Google, Grok, DeepSeek, Ollama
- Vector store: ChromaDB 0.5+
- Embeddings: FastEmbed (BGE)
- Data: Pandas 2.2+, Pydantic 2.5+
- Python 3.11+

## Key Data Flows
- Data load → embeddings → ChromaDB persist → hybrid retrieval → rerank → response
- Expert Panel → retrieval filters (geo radius, listing type, price, sort) + indices/time‑series → export digest

---

# Target Architecture (V4)

## System Overview
- Next.js web app (chat, search, dashboards, exports)
- Backend API (chat orchestration, tools, retrieval, analytics, notifications)
- Retrieval service (vector store + keyword + metadata + geo + sorting) behind API contracts
- Provider abstraction for multi-LLM support with consistent settings and policy controls
- Optional voice layer (ElevenLabs) for TTS/voice UX

## Deployment Target
- Web: Vercel
- API: container-based deployment (separate from web), versioned endpoints
