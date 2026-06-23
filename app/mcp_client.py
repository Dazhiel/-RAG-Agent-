"""高德地图 MCP 工具客户端管理器。"""
from typing import Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import RAGConfig


class McpClientManager:
    """按领域加载并缓存高德地图 MCP 工具。"""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._client: Optional[MultiServerMCPClient] = None
        self._tools_cache: dict[str, list[BaseTool]] = {}

    async def _get_client(self) -> MultiServerMCPClient:
        if self._client is None:
            mcp_url = self.config.mcp_url
            if "mcp.amap.com" in mcp_url and "key=" not in mcp_url:
                separator = "&" if "?" in mcp_url else "?"
                mcp_url = f"{mcp_url}{separator}key={self.config.amap_maps_api_key}"

            headers = {}
            if "dashscope.aliyuncs.com" in mcp_url:
                headers["Authorization"] = f"Bearer {self.config.api_key}"

            self._client = MultiServerMCPClient(
                {
                    "amap-server": {
                        "transport": self.config.mcp_transport,
                        "url": mcp_url,
                        "headers": headers,
                    }
                }
            )
        return self._client

    async def get_all_tools(self) -> list[BaseTool]:
        if "all" not in self._tools_cache:
            client = await self._get_client()
            self._tools_cache["all"] = await client.get_tools()
        return self._tools_cache["all"]

    async def get_tools_for(self, domain: str) -> list[BaseTool]:
        all_tools = await self.get_all_tools()
        target_names = set(self.config.tool_domains.get(domain, []))
        return [tool for tool in all_tools if tool.name in target_names]

    async def close(self) -> None:
        self._client = None
        self._tools_cache.clear()
