"""从环境变量加载项目配置。"""
import os
import re
from dataclasses import dataclass, field

import httpx
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


@dataclass
class RAGConfig:
    # 路径
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "data"))
    md5_record_path: str = field(default_factory=lambda: os.getenv("MD5_RECORD_PATH", "./runtime/md5.text"))

    # 文本切分
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))
    separators: list[str] = field(default_factory=lambda: [
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        "；",
        "，",
        ".",
        "!",
        "?",
        ";",
        ",",
        " ",
        "",
    ])
    max_split_threshold: int = field(default_factory=lambda: int(os.getenv("MAX_SPLIT_THRESHOLD", "500")))

    # 检索
    retrieval_top_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "3")))
    retrieval_candidates: int = field(default_factory=lambda: int(os.getenv("RETRIEVAL_CANDIDATES", "10")))

    # 模型
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "qwen3-max"))
    summary_model: str = field(default_factory=lambda: os.getenv("SUMMARY_MODEL", "qwen-turbo"))


    # PostgreSQL/ParadeDB knowledge base
    knowledge_table: str = field(default_factory=lambda: os.getenv("KNOWLEDGE_TABLE", "knowledge_chunks"))
    dense_vector_dim: int = field(default_factory=lambda: int(os.getenv("DENSE_VECTOR_DIM", "1024")))
    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))

    # 高德地图 MCP 定位/天气工具
    mcp_enabled: bool = field(default_factory=lambda: os.getenv("MCP_ENABLED", "true").lower() == "true")
    mcp_transport: str = field(default_factory=lambda: os.getenv("MCP_TRANSPORT", "http"))
    mcp_url: str = field(
        default_factory=lambda: os.getenv(
            "MCP_URL",
            "https://mcp.amap.com/mcp",
        )
    )
    tool_domains: dict = field(default_factory=lambda: {
        "location": ["maps_ip_location"],
        "weather": ["maps_weather"],
    })

    # 用户网络上下文
    ip: str = field(default_factory=lambda: os.getenv("IP", ""))

    # PostgreSQL chat history
    postgres_host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "127.0.0.1"))
    postgres_port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    postgres_user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    postgres_password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "1234"))
    postgres_database: str = field(default_factory=lambda: os.getenv("POSTGRES_DATABASE", "rag_agent"))
    postgres_connect_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("POSTGRES_CONNECT_TIMEOUT_SECONDS", "5")))

    # Chat memory window and rolling summary
    history_recent_turns: int = field(default_factory=lambda: int(os.getenv("HISTORY_RECENT_TURNS", "6")))
    default_user_id: str = field(default_factory=lambda: os.getenv("DEFAULT_USER_ID", "default_user"))

    @property
    def api_key(self) -> str:
        return os.getenv("DASHSCOPE_API_KEY", "")

    @property
    def amap_maps_api_key(self) -> str:
        return os.getenv("AMAP_MAPS_API_KEY", "")

    @property
    def postgres_uri(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
            "?sslmode=disable"
        )

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("请在环境变量或 .env 文件中设置 DASHSCOPE_API_KEY。")
        if self.mcp_enabled and "mcp.amap.com" in self.mcp_url and not self.amap_maps_api_key:
            raise ValueError("Please set AMAP_MAPS_API_KEY in environment variables or .env when MCP_ENABLED=true.")

    @staticmethod
    async def get_public_ip() -> str:
        services = [
            ("https://myip.ipip.net", "text"),
            ("https://cip.cc", "text"),
            ("https://api.ipify.org", "text"),
            ("https://ifconfig.me/ip", "text"),
        ]
        async with httpx.AsyncClient(timeout=5) as client:
            for url, mode in services:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    text = resp.text.strip()
                    if mode == "text":
                        match = re.search(
                            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",
                            text,
                        )
                        if match:
                            ip = match.group(0)
                            print(f"[IP检测] {url} -> {ip}")
                            return ip
                except Exception:
                    continue
        return ""




