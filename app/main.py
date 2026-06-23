"""
RAG Agent 的 FastAPI 后端。

运行方式：
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
import asyncio
import sys
import uuid
from functools import lru_cache
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.agent import RagAgent
from app.config import RAGConfig
from app.mcp_client import McpClientManager
from app.memory_embeddings import get_memory_embeddings

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class AddDocumentRequest(BaseModel):
    content: str = Field(..., min_length=1)
    source: str = "api"


class AddDocumentResponse(BaseModel):
    result: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    token_report: Optional[Dict[str, Any]] = None


@lru_cache(maxsize=1)
def get_config() -> RAGConfig:
    return RAGConfig()


async def load_mcp_tools():
    config = get_config()
    if not config.mcp_enabled:
        print("MCP tools disabled by MCP_ENABLED=false")
        return []

    manager = McpClientManager(config)
    location_tools = await manager.get_tools_for("location")
    weather_tools = await manager.get_tools_for("weather")
    return [*location_tools, *weather_tools]


async def build_persistence(config: RAGConfig):
    """Create LangGraph checkpointer/store. Prefer Postgres; fall back to memory for local smoke tests."""
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from langgraph.store.postgres.aio import AsyncPostgresStore

        memory_index = {
            "dims": config.dense_vector_dim,
            "embed": get_memory_embeddings(config.embedding_model),
            "fields": ["memory"],
            "distance_type": "cosine",
            "ann_index_config": {"kind": "hnsw"},
        }
        checkpointer_cm = AsyncPostgresSaver.from_conn_string(config.postgres_uri)
        store_cm = AsyncPostgresStore.from_conn_string(config.postgres_uri, index=memory_index)
        checkpointer = await checkpointer_cm.__aenter__()
        store = await store_cm.__aenter__()
        await checkpointer.setup()
        await store.setup()
        print("[memory] using Postgres checkpointer/store")
        return checkpointer, store, checkpointer_cm, store_cm, "postgres"
    except Exception as exc:
        print(f"[memory] Postgres persistence unavailable, using in-memory fallback: {exc}")
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.store.memory import InMemoryStore

        memory_index = {
            "dims": config.dense_vector_dim,
            "embed": get_memory_embeddings(config.embedding_model),
            "fields": ["memory"],
        }
        return InMemorySaver(), InMemoryStore(index=memory_index), None, None, "memory"


app = FastAPI(
    title="扫地机器人智能客服 API",
    description="基于 FastAPI、LangGraph、PostgreSQL/ParadeDB、BGE 的后端服务。",
    version="2.0.0",
)


@app.on_event("startup")
async def startup_event():
    config = get_config()
    app.state.checkpointer, app.state.memory_store, app.state.checkpointer_cm, app.state.memory_store_cm, app.state.persistence_backend = await build_persistence(config)
    app.state.mcp_tools = await load_mcp_tools()
    user_ip = config.ip or await config.get_public_ip()
    if user_ip:
        print(f"[IP] {user_ip}")
    else:
        print("[IP] public IP unavailable")
    app.state.agent = RagAgent(
        config=config,
        mcp_tools=app.state.mcp_tools,
        user_ip=user_ip,
        checkpointer=app.state.checkpointer,
        store=app.state.memory_store,
    )


@app.on_event("shutdown")
async def shutdown_event():
    if getattr(app.state, "memory_store_cm", None) is not None:
        await app.state.memory_store_cm.__aexit__(None, None, None)
    if getattr(app.state, "checkpointer_cm", None) is not None:
        await app.state.checkpointer_cm.__aexit__(None, None, None)


def get_agent() -> RagAgent:
    return app.state.agent


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "vector_store": "PostgreSQL/ParadeDB",
        "agent": "LangGraph",
        "short_term_memory": "checkpoint",
        "long_term_memory": "store",
        "persistence_backend": getattr(app.state, "persistence_backend", "unknown"),
        "default_user_id": get_config().default_user_id,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    agent: RagAgent = Depends(get_agent),
):
    session_id = payload.session_id or uuid.uuid4().hex
    answer, token_report = await agent.query_with_report(payload.question, session_id=session_id)
    return ChatResponse(
        session_id=session_id,
        answer=answer,
        token_report=token_report,
    )

@app.post("/sessions/{session_id}/clear")
async def clear_session(session_id: str, agent: RagAgent = Depends(get_agent)):
    await agent.clear_history(session_id)
    return {"session_id": session_id, "cleared": True}


@app.post("/knowledge/documents", response_model=AddDocumentResponse)
async def add_document(
    payload: AddDocumentRequest,
    agent: RagAgent = Depends(get_agent),
):
    result = await run_in_threadpool(agent.kb.add_text, payload.content, payload.source)
    return AddDocumentResponse(result=result)






