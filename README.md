# 扫地机器人智能客服 RAG Agent

一个基于 **FastAPI + Streamlit + LangGraph + PostgreSQL/ParadeDB** 构建的垂直领域智能客服 RAG Agent 项目，面向扫地机器人、拖地机器人等智能家居产品场景。

系统能够结合产品知识库、多轮对话上下文、用户长期偏好和工具调用能力，完成产品咨询、故障排查、维护保养、选购建议等问答任务。

## 项目简介

本项目使用 LangGraph 编排智能客服 Agent 工作流，通过 LLM 对用户问题进行意图判断：普通问题进入 Chat 分支，产品知识类问题进入 RAG 分支。RAG 分支会先进行查询改写，再从 PostgreSQL/ParadeDB 知识库中完成混合检索、结果重排和增强回答生成。

项目同时实现了会话记忆与长期用户记忆。会话状态持久化到 PostgreSQL，并通过“更早对话摘要 + 最近 6 轮原始上下文”的方式管理多轮对话；用户偏好、设备信息和长期问题模式会被抽取为长期记忆，并通过向量语义检索在后续对话中召回。

## 功能特点

- **Chat/RAG 智能分流**：根据用户意图自动选择普通聊天或知识库问答。
- **知识库增强问答**：支持 PDF/TXT 文档读取、文本切分、MD5 去重与增量入库。
- **混合检索与重排**：基于 BGE-M3 向量检索、BM25 检索、RRF 融合和 BGE reranker 精排。
- **多轮会话记忆**：使用 PostgreSQL 持久化 LangGraph 会话状态，实现摘要 + 最近 6 轮的滑动窗口上下文。
- **长期用户记忆**：自动抽取用户偏好、设备信息和长期问题模式，并基于 pgvector 进行语义召回。
- **工具调用能力**：集成高德地图 MCP 工具，以及故障诊断、保养计划、选购推荐等业务工具。
- **Web 交互界面**：FastAPI 提供后端问答接口，Streamlit 提供聊天前端。

## 技术栈

- **后端服务**：FastAPI, Uvicorn
- **前端界面**：Streamlit
- **Agent 编排**：LangGraph, LangChain
- **大模型服务**：DashScope / Qwen
- **数据存储**：PostgreSQL, ParadeDB, pgvector, pg_search
- **检索模型**：BGE-M3 embedding, BGE reranker
- **工具调用**：MCP, LangChain tools

## Agent 工作流

```text
用户问题
  -> 上下文准备
  -> LLM 意图路由
  -> Chat 回答 或 RAG 查询改写 + 知识库检索 + 重排 + 回答生成
  -> 长期记忆抽取
  -> 会话摘要维护
  -> 返回回答
```

## 快速启动

### 1. 创建环境并安装依赖

```powershell
conda create -n rag python=3.10 -y
conda activate rag
pip install -r requirements.txt
```

如果已经存在 `rag` 环境：

```powershell
conda activate rag
pip install -r requirements.txt
```

### 2. 配置环境变量

复制环境变量示例文件：

```powershell
Copy-Item .env.example .env
```

至少需要配置：

```env
DASHSCOPE_API_KEY=你的 DashScope API Key
POSTGRES_PASSWORD=1234
```

默认会从 HuggingFace 加载 BGE-M3 embedding 和 BGE reranker。如果已经把模型下载到本地，可以在 `.env` 中启用：

```env
EMBEDDING_MODEL=./models/bge-m3
RERANKER_MODEL=./models/bge-reranker-v2-m3
```

如果不需要 MCP 工具，可以关闭：

```env
MCP_ENABLED=false
```

### 3. 启动数据库

```powershell
docker compose up -d
```

### 4. 初始化数据库

```powershell
python scripts/init_postgres.py
```

### 5. 构建知识库

首次运行需要先将 `data/` 目录下的文档写入知识库：

```powershell
python scripts/build_knowledge_base.py
```

如果旧版本项目已经使用同一个 PostgreSQL 数据库构建过知识库，并且 embedding 维度一致，可以跳过这一步。

### 6. 启动后端

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

健康检查：

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
```

### 7. 启动前端

另开一个 PowerShell：

```powershell
conda activate rag
streamlit run frontend/streamlit_app.py
```

浏览器访问：

```text
http://localhost:8501
```

前端中的后端地址建议填写：

```text
http://127.0.0.1:8000
```

## 说明

- `.env` 中的 API Key 不要提交到 GitHub。
- `models/` 目录通常较大，建议本地单独准备，不直接上传到仓库。
- 首次启动后端时，LangGraph checkpoint 和长期记忆 store 相关表会自动创建。