"""FastAPI RAG 后端对应的 Streamlit 前端。

运行方式：
    streamlit run frontend/streamlit_app.py
"""
import os
import uuid
from datetime import datetime

import httpx
import streamlit as st


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


PAGE_CSS = """
<style>
    :root {
        --page-bg: #f6f8fb;
        --panel-bg: rgba(255, 255, 255, 0.88);
        --ink: #172033;
        --muted: #667085;
        --line: #d9e2ef;
        --accent: #1f7a8c;
        --accent-dark: #125463;
        --accent-soft: #e6f5f8;
        --warn: #fff7ed;
    }

    .stApp {
        background:
            radial-gradient(circle at 18% 8%, rgba(31, 122, 140, 0.12), transparent 28%),
            radial-gradient(circle at 88% 4%, rgba(245, 158, 11, 0.10), transparent 24%),
            linear-gradient(180deg, #f8fbff 0%, var(--page-bg) 42%, #eef3f8 100%);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.1rem;
        padding-bottom: 7.5rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f2f7fb 100%);
        border-right: 1px solid var(--line);
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label {
        color: #344054;
    }

    .app-hero {
        position: relative;
        padding: 1.35rem 1.45rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(31, 122, 140, 0.18);
        border-radius: 8px;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.96), rgba(239,249,251,0.90)),
            linear-gradient(90deg, rgba(31,122,140,0.08), rgba(245,158,11,0.06));
        box-shadow: 0 18px 45px rgba(21, 47, 76, 0.10);
    }

    .app-kicker {
        margin: 0 0 0.35rem;
        color: var(--accent-dark);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
    }

    .app-title {
        margin: 0;
        color: var(--ink);
        font-size: 2rem;
        line-height: 1.18;
        font-weight: 750;
        letter-spacing: 0;
    }

    .app-subtitle {
        max-width: 720px;
        margin: 0.55rem 0 0;
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.65;
    }

    .status-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0 0 1.2rem;
    }

    .status-card {
        padding: 0.8rem 0.95rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--panel-bg);
        box-shadow: 0 10px 26px rgba(21, 47, 76, 0.07);
    }

    .status-label {
        color: var(--muted);
        font-size: 0.78rem;
        margin-bottom: 0.2rem;
    }

    .status-value {
        color: var(--ink);
        font-size: 1.02rem;
        font-weight: 700;
    }

    [data-testid="stChatMessage"] {
        padding: 0.9rem 1rem;
        margin-bottom: 0.85rem;
        border: 1px solid rgba(217, 226, 239, 0.95);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.90);
        box-shadow: 0 12px 28px rgba(21, 47, 76, 0.08);
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #eef9fb 0%, #ffffff 100%);
        border-color: rgba(31, 122, 140, 0.24);
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.72;
    }

    [data-testid="stCaptionContainer"] {
        color: #7a8699;
    }

    .stTextInput input {
        border-radius: 8px;
        border-color: var(--line);
        background: #ffffff;
    }

    .stTextInput input:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 2px rgba(31, 122, 140, 0.14);
    }

    .stButton button {
        border-radius: 8px;
        border-color: rgba(31, 122, 140, 0.26);
        background: #ffffff;
        color: var(--accent-dark);
        font-weight: 700;
        transition: all 140ms ease;
    }

    .stButton button:hover {
        border-color: var(--accent);
        background: var(--accent-soft);
        color: var(--accent-dark);
        transform: translateY(-1px);
    }

    [data-testid="stChatInput"] {
        border-top: 1px solid rgba(217, 226, 239, 0.72);
        background: rgba(246, 248, 251, 0.88);
        backdrop-filter: blur(10px);
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 8px;
        border: 1px solid var(--line);
        box-shadow: 0 12px 28px rgba(21, 47, 76, 0.08);
    }

    div[data-testid="stAlert"] {
        border-radius: 8px;
    }

    @media (max-width: 720px) {
        .block-container {
            padding-top: 1.25rem;
        }

        .app-hero {
            padding: 1.1rem;
        }

        .app-title {
            font-size: 1.55rem;
        }

        .status-strip {
            grid-template-columns: 1fr;
        }
    }
</style>
"""


st.set_page_config(
    page_title="扫地机器人智能客服",
    page_icon="🤖",
    layout="wide",
)


def init_state() -> None:
    defaults = {
        "session_id": uuid.uuid4().hex,
        "messages": [],
        "api_base_url": DEFAULT_API_BASE_URL,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value



def ask_backend(question: str) -> dict:
    url = st.session_state.api_base_url.rstrip("/") + "/chat"
    payload = {
        "question": question,
        "session_id": st.session_state.session_id,
    }
    with httpx.Client(timeout=120, trust_env=False) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def clear_backend_session() -> None:
    url = st.session_state.api_base_url.rstrip() + f"/sessions/{st.session_state.session_id}/clear"
    with httpx.Client(timeout=30, trust_env=False) as client:
        response = client.post(url)
        response.raise_for_status()


def format_token_report(report):
    if not report:
        return ""
    provider = report.get("token_counter_provider", "unknown")
    model = report.get("token_counter_model", "")
    counter = f"{provider}/{model}" if model else provider
    return (
        f"Token ({counter}): "
        f"history {report.get('history_raw_tokens', 0)} -> {report.get('history_compressed_tokens', 0)} "
        f"saved {report.get('history_saved_tokens', 0)}, "
        f"recent {report.get('recent_message_tokens', 0)}/{report.get('recent_message_count', 0)} msgs, "
        f"summary {report.get('summary_tokens', 0)}, "
        f"route {report.get('last_route', 'chat')}, "
        f"prompt {report.get('last_prompt_tokens', 0)}, "
        f"rag_ctx {report.get('last_rag_context_tokens', 0)}"
    )


def render_page_header() -> None:
    message_count = len(st.session_state.messages)
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="app-hero">
            <p class="app-kicker">RAG CUSTOMER ASSISTANT</p>
            <h1 class="app-title">扫地机器人智能客服</h1>
            <p class="app-subtitle">
                面向产品咨询、故障排查和维护保养的知识库问答助手，支持多轮会话上下文。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="status-strip">
            <div class="status-card">
                <div class="status-label">当前会话</div>
                <div class="status-value">{st.session_state.session_id[:12]}...</div>
            </div>
            <div class="status-card">
                <div class="status-label">历史消息</div>
                <div class="status-value">{message_count} 条</div>
            </div>
            <div class="status-card">
                <div class="status-label">Checkpoint 记忆</div>
                <div class="status-value">6轮+摘要</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    init_state()

    render_page_header()

    with st.sidebar:
        st.subheader("连接")
        st.text_input("API 地址", key="api_base_url")
        st.caption(f"会话 ID：{st.session_state.session_id[:12]}...")

        if st.button("新建会话", use_container_width=True):
            st.session_state.session_id = uuid.uuid4().hex
            st.session_state.messages = []
            st.rerun()

        if st.button("清空当前会话", use_container_width=True):
            try:
                clear_backend_session()
            except Exception as exc:
                st.warning(f"后端清空失败：{exc}")
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            st.caption(message["time"])

    question = st.chat_input("请输入你的问题")
    if not question:
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append(
        {"role": "user", "content": question, "time": now}
    )
    with st.chat_message("user"):
        st.markdown(question)
        st.caption(now)

    with st.chat_message("assistant"):
        answer = ""
        token_caption = ""
        try:
            with st.spinner("正在思考..."):
                result = ask_backend(question)
                answer = result.get("answer", "")
                token_caption = format_token_report(result.get("token_report"))
            st.markdown(answer)
        except Exception as exc:
            answer = f"请求后端失败：{exc}"
            st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": f"{answer}\n\n`{token_caption}`" if token_caption else answer,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    st.rerun()


if __name__ == "__main__":
    main()


