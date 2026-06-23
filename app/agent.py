"""显式 LangGraph 流程：短期 checkpoint 记忆、长期 store 记忆、问题改写、路由和回答。"""
import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
try:
    from langchain_core.messages import RemoveMessage
except ImportError:  # pragma: no cover - compatibility with older LangChain packages
    from langchain.messages import RemoveMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from app.business_tools import get_business_tools
from app.config import RAGConfig
from app.knowledge_base import KnowledgeBaseBuilder
from app.token_counter import TokenCounter

RouteName = Literal["chat", "rag"]
FinalRouteName = Literal["chat", "rag"]


class RouterDecision(BaseModel):
    """LLM Router 的结构化分类结果。"""

    route: FinalRouteName = Field(
        description="只能选择 chat 或 rag。chat 表示普通聊天；rag 表示需要查询本地知识库。"
    )

class MemoryCandidate(BaseModel):
    """单条可长期保存的用户记忆。"""

    memory: str = Field(description="一条简洁、稳定、可跨会话复用的中文记忆。")


class MemoryExtraction(BaseModel):
    """长期记忆抽取结果。"""

    memories: List[MemoryCandidate] = Field(
        default_factory=list,
        description="本轮对话中值得长期保存的用户记忆；没有则为空列表。",
    )


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str
    summary_source_tokens: int
    question: str
    rewritten_query: str
    route: RouteName
    memory_context: str
    last_route: FinalRouteName
    last_prompt_tokens: int
    last_rag_context_tokens: int


CHAT_SYSTEM_PROMPT = """你是一个专业、简洁的中文聊天助手。

规则：
- 正常处理寒暄、闲聊和不需要知识库的普通问题。
- 只有当用户明确询问当前位置、所在地天气，或问题包含“这里”“本地”“天气”“气候”“下雨”“温度”等需要位置/天气信息的场景时，才调用高德地图 MCP 工具。
- 需要位置/天气时，先调用 maps_ip_location 获取城市，再按需调用 maps_weather；其他普通聊天不要调用工具。
- 如果用户询问扫地/拖地机器人产品使用、维护保养、故障排查、选购建议或常见问题，请直接提醒用户该问题会转交知识库客服分支处理，不要自行调用业务工具。

{ip_hint}"""

RAG_SYSTEM_PROMPT = """你是一个专业的扫地/拖地机器人中文客服助手。

规则：
- 始终使用中文回答，语气简洁、专业、像真实客服。
- 优先依据知识库上下文回答产品使用、维护保养、故障排查、选购建议和常见问题。
- 可以按需调用故障诊断、保养计划、选购推荐等业务工具辅助组织答案。
- 用户询问产品使用、维护保养、故障排查、选购建议、常见问题时，不要调用高德地图 MCP，优先使用知识库上下文回答。
- 只有当问题确实需要当前位置或天气信息时，才调用高德地图 MCP 工具。
- 只有当用户明确询问当前位置、所在地天气，或问题包含“这里”“本地”“天气”“气候”“下雨”“温度”等需要位置/天气信息的场景时，才调用高德地图 MCP 工具。
- 如果知识库资料不足，请明确说明，不要编造。

{ip_hint}"""


SUMMARY_SYSTEM_PROMPT = """你是一个会话记忆压缩器。请把扫地机器人客服对话压缩成供后续 Agent 使用的会话摘要。
任务是把“已有摘要”和“新增旧对话”合并成新的摘要，而不是只总结新增旧对话。

要求：
- 保留用户明确表达的需求、偏好、设备型号、故障现象、已尝试方案、关键结论。
- 保留尚未解决的问题和下一步需要确认的事项。
- 删除寒暄、重复内容、无关细节。
- 如果已有摘要和新增旧对话冲突，优先相信新增旧对话，并在摘要中体现最新状态。
- 不要编造历史中没有的信息。
- 使用简洁中文输出。"""


LONG_TERM_MEMORY_SYSTEM_PROMPT = """你是长期记忆抽取器。请从本轮对话中抽取值得跨会话保存的用户记忆。

抽取目标：
- 只保存和用户本人、使用环境、设备情况、长期偏好或反复问题有关的信息。
- 每条记忆必须是独立、简洁、可复用的事实或偏好，后续单独看到也能理解。
- 记忆应尽量写成“用户...”开头的中文短句。

可以保存：
- 用户偏好：例如更喜欢低噪音、低水量、回答简洁。
- 用户稳定事实：例如家里地面材质、户型、宠物、主要使用场景。
- 设备信息：例如扫地机器人型号、配件、使用年限、常用设置。
- 长期问题模式：例如经常出现拖地水渍、避障异常、某类故障反复发生。

不要保存：
- 寒暄、感谢、一次性提问、临时上下文。
- 助手给出的建议、排查步骤或知识库内容本身。
- 只是模型推测出来、用户没有明确表达或确认的信息。
- 本轮已经解决且不体现长期偏好、设备信息或问题模式的普通问题。

输出要求：
- 如果没有值得长期保存的记忆，memories 返回空列表。
- 如果有，每条只填写 memory 字段，不需要分类、置信度或解释。"""

class RagAgent:
    """包含 checkpoint 短期记忆和 store 长期记忆的 LangGraph Agent。"""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        mcp_tools: Optional[List[BaseTool]] = None,
        user_ip: str = "",
        checkpointer: Any = None,
        store: Any = None,
    ):
        self.config = config or RAGConfig()
        self.mcp_tools = mcp_tools or []
        self.checkpointer = checkpointer
        self.store = store
        self.user_id = self.config.default_user_id
        self.business_tools = get_business_tools()
        self.rag_tools = [*self.mcp_tools, *self.business_tools]
        self.kb = KnowledgeBaseBuilder(self.config)
        self.retriever = self.kb.get_retriever()
        self.token_counter = TokenCounter()
        self.llm = ChatTongyi(
            model=self.config.chat_model,
            api_key=self.config.api_key,
            streaming=False,
        )
        self.summary_llm = ChatTongyi(
            model=self.config.summary_model,
            api_key=self.config.api_key,
            streaming=False,
        )
        self.router_llm = self.llm.with_structured_output(RouterDecision)
        self.memory_llm = self.summary_llm.with_structured_output(MemoryExtraction)
        ip_hint = (
            f"当前用户公网 IP 是：{user_ip}。"
            "当用户询问当前位置、所在地天气或与本地环境有关的问题时，"
            "请优先使用此 IP 调用 maps_ip_location，再按需要调用 maps_weather。"
            if user_ip
            else ""
        )
        self.chat_system_prompt = CHAT_SYSTEM_PROMPT.format(ip_hint=ip_hint)
        self.rag_system_prompt = RAG_SYSTEM_PROMPT.format(ip_hint=ip_hint)
        self.chat_agent = (
            create_react_agent(model=self.llm, tools=self.mcp_tools)
            if self.mcp_tools
            else None
        )
        self.rag_agent = (
            create_react_agent(model=self.llm, tools=self.rag_tools)
            if self.rag_tools
            else None
        )
        self.graph = self._build_graph()

    @property
    def recent_message_limit(self) -> int:
        return max(self.config.history_recent_turns, 0) * 2

    @property
    def memory_namespace(self) -> tuple[str, str]:
        return (self.user_id, "memories")

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("prepare_context", self._prepare_context_node)
        workflow.add_node("rewrite", self._rewrite_node)
        workflow.add_node("llm_router", self._llm_router_node)
        workflow.add_node("chat_agent", self._chat_agent_node)
        workflow.add_node("rag_agent", self._rag_agent_node)
        workflow.add_node("extract_long_term_memory", self._extract_long_term_memory_node)
        workflow.add_node("summarize_if_needed", self._summarize_if_needed_node)

        workflow.set_entry_point("prepare_context")
        workflow.add_edge("prepare_context", "llm_router")
        workflow.add_conditional_edges(
            "llm_router",
            self._select_route,
            {
                "chat": "chat_agent",
                "rag": "rewrite",
            },
        )
        workflow.add_edge("rewrite", "rag_agent")
        workflow.add_edge("chat_agent", "extract_long_term_memory")
        workflow.add_edge("rag_agent", "extract_long_term_memory")
        workflow.add_edge("extract_long_term_memory", "summarize_if_needed")
        workflow.add_edge("summarize_if_needed", END)

        compile_kwargs = {}
        if self.checkpointer is not None:
            compile_kwargs["checkpointer"] = self.checkpointer
        if self.store is not None:
            compile_kwargs["store"] = self.store
        try:
            return workflow.compile(**compile_kwargs)
        except TypeError:
            # Older LangGraph versions do not accept store= at compile time.
            # The agent still uses self.store directly for long-term memory.
            compile_kwargs.pop("store", None)
            return workflow.compile(**compile_kwargs)

    @staticmethod
    def _latest_human_text(messages: List[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return str(message.content)
        return ""

    @staticmethod
    def _latest_ai_text(messages: List[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return RagAgent._chunk_text(message.content)
        return ""

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        parts = []
        for doc in docs:
            source = doc.metadata.get("source", "未知来源")
            score = doc.metadata.get("rerank_score", doc.metadata.get("retrieval_score", ""))
            score_text = f" score={score:.4f}" if isinstance(score, float) else ""
            parts.append(f"[source: {source}{score_text}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)


    @staticmethod
    def _select_route(state: AgentState) -> RouteName:
        return state.get("route", "chat")

    @staticmethod
    def _message_role(message: BaseMessage) -> str:
        if isinstance(message, HumanMessage):
            return "用户"
        if isinstance(message, AIMessage):
            return "助手"
        if isinstance(message, SystemMessage):
            return "系统"
        return message.__class__.__name__

    @classmethod
    def _format_messages_for_summary(cls, messages: List[BaseMessage]) -> str:
        lines = []
        for message in messages:
            lines.append(f"{cls._message_role(message)}：{cls._chunk_text(message.content)}")
        return "\n".join(lines)

    def _build_prompt_messages(self, system_prompt: str, state: AgentState) -> List[BaseMessage]:
        prompt: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        summary = state.get("summary", "").strip()
        if summary:
            prompt.append(
                SystemMessage(
                    content=(
                        "以下是本会话更早历史的摘要，只用于理解上下文；"
                        "如果摘要和最近原始消息冲突，以最近原始消息为准。\n\n"
                        f"{summary}"
                    )
                )
            )
        memory_context = state.get("memory_context", "").strip()
        if memory_context:
            prompt.append(
                SystemMessage(
                    content=(
                        "以下是该用户的长期记忆，用于个性化回答；"
                        "不要主动暴露记忆来源。\n\n"
                        f"{memory_context}"
                    )
                )
            )
        prompt.extend(state.get("messages", []))
        return prompt

    async def _load_long_term_memories(self, query: str) -> str:
        if self.store is None or not query.strip():
            return ""
        try:
            items = await self.store.asearch(
                self.memory_namespace,
                query=query,
                limit=5,
            )
        except Exception as exc:
            print(f"[memory_store] semantic search failed: {exc}")
            return ""

        memories = []
        for item in items:
            value = getattr(item, "value", {}) or {}
            memory = str(value.get("memory") or value.get("data") or "").strip()
            if memory:
                memories.append(f"- {memory}")
        return "\n".join(memories)

    async def _prepare_context_node(self, state: AgentState) -> AgentState:
        question = self._latest_human_text(state.get("messages", []))
        memory_context = await self._load_long_term_memories(question)
        return {
            "question": question,
            "memory_context": memory_context,
            "rewritten_query": "",
            "route": "chat",
            "last_route": "chat",
            "last_prompt_tokens": 0,
            "last_rag_context_tokens": 0,
        }
    async def _rewrite_node(self, state: AgentState) -> AgentState:
        question = state.get("question") or self._latest_human_text(state["messages"])
        prompt = [
            SystemMessage(
                content=(
                    "请将用户问题改写成一句适合知识库检索的中文查询语句。"
                    "保留产品名、错误码、故障现象、场景限制等关键信息。"
                    "只返回改写后的查询语句，不要解释。"
                )
            ),
            HumanMessage(content=question),
        ]
        response = await self.llm.ainvoke(prompt)
        rewritten = str(response.content).strip() or question
        return {"question": question, "rewritten_query": rewritten}
    async def _llm_router_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        decision = await self.router_llm.ainvoke(
            [
                SystemMessage(
                    content=(
                        "请判断用户请求应该进入哪个处理分支。\n"
                        "chat：寒暄、闲聊，或不需要项目知识库的问题。\n"
                        "rag：扫地/拖地机器人产品知识、维护保养、故障排查、选购建议、常见问题。\n"
                        "必须只在结构化字段 route 中选择 chat 或 rag。"
                    )
                ),
                HumanMessage(content=question),
            ]
        )
        route: FinalRouteName = decision.route
        return {"route": route}
    async def _chat_agent_node(self, state: AgentState) -> AgentState:
        messages = self._build_prompt_messages(self.chat_system_prompt, state)
        prompt_tokens = self.token_counter.count_messages(messages)
        if self.chat_agent is not None:
            answer = await self._invoke_react_agent(self.chat_agent, messages)
        else:
            answer = await self._invoke_llm(messages)
        return {
            "messages": [AIMessage(content=answer, id=str(uuid.uuid4()))],
            "last_route": "chat",
            "last_prompt_tokens": prompt_tokens,
            "last_rag_context_tokens": 0,
        }
    async def _rag_agent_node(self, state: AgentState) -> AgentState:
        docs = self.retriever.invoke(state["rewritten_query"])
        context = self._format_docs(docs)
        answer_prompt = self._build_rag_answer_prompt(state["question"], context=context)
        rag_messages = [
            *self._build_prompt_messages(self.rag_system_prompt, state),
            HumanMessage(content=answer_prompt),
        ]
        prompt_tokens = self.token_counter.count_messages(rag_messages)
        rag_context_tokens = self.token_counter.count_text(context)
        if self.rag_agent is not None:
            answer = await self._invoke_react_agent(self.rag_agent, rag_messages)
        else:
            answer = await self._invoke_llm(rag_messages)
        return {
            "messages": [AIMessage(content=answer, id=str(uuid.uuid4()))],
            "last_route": "rag",
            "last_prompt_tokens": prompt_tokens,
            "last_rag_context_tokens": rag_context_tokens,
        }
    def _build_rag_answer_prompt(self, question: str, context: str) -> str:
        return (
            "请基于以下信息回答用户问题。知识库上下文必须优先采用；如果资料不足，请明确说明。\n"
            "你可以按需调用故障诊断、保养计划、选购推荐等业务工具。"
            "只有当问题确实需要当前位置或天气信息时，才调用高德地图 MCP 工具。\n\n"
            f"用户问题：{question}\n\n"
            f"知识库上下文：\n{context}"
        )

    async def _extract_long_term_memory_node(self, state: AgentState) -> AgentState:
        if self.store is None:
            return {}

        question = state.get("question") or self._latest_human_text(state.get("messages", []))
        answer = self._latest_ai_text(state.get("messages", []))
        if not question or not answer:
            return {}

        extraction_input = f"用户：{question}\n\n助手：{answer}"
        prompt = [
            SystemMessage(content=LONG_TERM_MEMORY_SYSTEM_PROMPT),
            HumanMessage(content=extraction_input),
        ]
        try:
            extraction = await self.memory_llm.ainvoke(prompt)
        except Exception as exc:
            print(f"[memory_store] structured extract failed: {exc}")
            return {}

        raw_memories = getattr(extraction, "memories", None)
        if raw_memories is None and isinstance(extraction, dict):
            raw_memories = extraction.get("memories")
        if not raw_memories:
            return {}

        memories = []
        for item in raw_memories:
            if isinstance(item, dict):
                memory = str(item.get("memory") or "").strip()
            else:
                memory = str(getattr(item, "memory", "") or "").strip()
            if memory:
                memories.append(memory)
        if not memories:
            return {}

        existing_texts = set()
        try:
            existing_items = await self.store.asearch(self.memory_namespace, limit=100)
            for item in existing_items:
                value = getattr(item, "value", {}) or {}
                text = str(value.get("memory") or value.get("data") or "").strip()
                if text:
                    existing_texts.add(text)
        except Exception:
            existing_texts = set()

        saved = 0
        for memory in memories:
            if not memory or memory in existing_texts:
                continue
            try:
                await self.store.aput(
                    self.memory_namespace,
                    str(uuid.uuid4()),
                    {
                        "memory": memory,
                        "source": "chat",
                    },
                )
                saved += 1
            except Exception as exc:
                print(f"[memory_store] save failed: {exc}")
        if saved:
            print(f"[memory_store] saved {saved} memories for user={self.user_id}")
        return {}
    async def _summarize_if_needed_node(self, state: AgentState) -> AgentState:
        limit = self.recent_message_limit
        messages = state.get("messages", [])
        if limit <= 0 or len(messages) <= limit:
            return {}

        old_messages = messages[:-limit]
        removable_messages = [m for m in old_messages if getattr(m, "id", None)]
        if not removable_messages:
            return {}

        old_summary = state.get("summary", "").strip() or "无"
        old_messages_text = self._format_messages_for_summary(old_messages)
        old_messages_tokens = self.token_counter.count_messages(old_messages)
        previous_source_tokens = int(state.get("summary_source_tokens") or 0)
        prompt = [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"已有会话摘要：\n{old_summary}\n\n"
                    f"新增旧对话：\n{old_messages_text}\n\n"
                    "请输出合并后的新会话摘要。"
                )
            ),
        ]
        response = await self.summary_llm.ainvoke(prompt)
        summary = self._chunk_text(getattr(response, "content", "")).strip()
        if not summary:
            return {}

        return {
            "summary": summary,
            "summary_source_tokens": previous_source_tokens + old_messages_tokens,
            "messages": [RemoveMessage(id=m.id) for m in removable_messages],
        }
    @staticmethod
    def _chunk_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "".join(parts)
        return str(content or "")

    async def _invoke_llm(self, messages: List[BaseMessage]) -> str:
        response = await self.llm.ainvoke(messages)
        return self._chunk_text(getattr(response, "content", "")).strip()

    async def _invoke_react_agent(self, agent, messages: List[BaseMessage]) -> str:
        result = await agent.ainvoke({"messages": messages})
        result_messages = result.get("messages", [])
        for message in reversed(result_messages):
            if isinstance(message, AIMessage):
                return self._chunk_text(message.content).strip()
        return ""

    @staticmethod
    def _last_ai_answer(messages: List[BaseMessage]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return RagAgent._chunk_text(message.content).strip()
        return ""

    async def query(self, question: str, session_id: str = "default") -> str:
        answer, _ = await self.query_with_report(question, session_id=session_id)
        return answer

    async def query_with_report(
        self,
        question: str,
        session_id: str = "default",
    ) -> tuple[str, Dict[str, Any]]:
        state = await self.graph.ainvoke(
            {
                "messages": [HumanMessage(content=question, id=str(uuid.uuid4()))],
            },
            config={"configurable": {"thread_id": session_id}},
        )
        answer = self._last_ai_answer(state.get("messages", []))
        summary_tokens = self.token_counter.count_text(state.get("summary", ""))
        recent_message_tokens = self.token_counter.count_messages(state.get("messages", []))
        memory_context_tokens = self.token_counter.count_text(state.get("memory_context", ""))
        summary_source_tokens = int(state.get("summary_source_tokens") or 0)
        history_raw_tokens = summary_source_tokens + recent_message_tokens
        history_compressed_tokens = summary_tokens + recent_message_tokens
        prompt_messages = self._build_prompt_messages(self.chat_system_prompt, state)
        final_report = {
            "history_raw_tokens": history_raw_tokens,
            "history_compressed_tokens": history_compressed_tokens,
            "history_saved_tokens": max(history_raw_tokens - history_compressed_tokens, 0),
            "summary_source_tokens": summary_source_tokens,
            "summary_tokens": summary_tokens,
            "recent_message_tokens": recent_message_tokens,
            "memory_context_tokens": memory_context_tokens,
            "prompt_with_system_tokens": self.token_counter.count_messages(prompt_messages),
            "last_route": state.get("last_route", state.get("route", "chat")),
            "last_prompt_tokens": int(state.get("last_prompt_tokens") or 0),
            "last_rag_context_tokens": int(state.get("last_rag_context_tokens") or 0),
            "recent_message_count": len(state.get("messages", [])),
            "recent_message_limit": self.recent_message_limit,
            "history_source": "langgraph_checkpoint",
            "memory_source": "langgraph_store" if self.store is not None else "none",
            "chat_model": self.config.chat_model,
            "summary_model": self.config.summary_model,
            "token_counter_provider": self.token_counter.provider,
            "token_counter_model": self.token_counter.model,
        }
        print(f"[token_report] session={session_id} {final_report}")
        return answer, final_report

    async def clear_history(self, session_id: str = "default") -> None:
        if self.checkpointer is not None and hasattr(self.checkpointer, "adelete_thread"):
            await self.checkpointer.adelete_thread(session_id)













