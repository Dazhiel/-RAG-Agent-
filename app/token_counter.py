"""Token counting helpers for prompt budgeting and diagnostics."""
import os
import re
from typing import Iterable


class TokenCounter:
    """Count tokens with DashScope when available, otherwise use a stable estimate."""

    def __init__(self) -> None:
        self.model = os.getenv("TOKEN_COUNTER_MODEL", "qwen-turbo")
        self.use_cloud = os.getenv("TOKEN_COUNTER_USE_CLOUD", "true").lower() == "true"
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self._cloud_available = self.use_cloud and bool(self.api_key)
        self._cache: dict[tuple[tuple[str, str], ...], int] = {}
        self._encoder = None
        if os.getenv("TOKEN_COUNTER_USE_TIKTOKEN", "false").lower() == "true":
            try:
                import tiktoken

                self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoder = None

    @property
    def provider(self) -> str:
        if self._cloud_available:
            return "dashscope"
        if self._encoder is not None:
            return "tiktoken"
        return "estimate"

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        cloud_count = self._count_dashscope_messages(
            [{"role": "user", "content": text}],
        )
        if cloud_count is not None:
            return cloud_count
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return self._estimate_text_tokens(text)

    def count_message(self, message) -> int:
        cloud_count = self._count_dashscope_messages([self._to_dashscope_message(message)])
        if cloud_count is not None:
            return cloud_count
        role_tokens = self.count_text(message.__class__.__name__)
        content = getattr(message, "content", "")
        if isinstance(content, str):
            content_tokens = self.count_text(content)
        else:
            content_tokens = self.count_text(str(content))
        return role_tokens + content_tokens + 4

    def count_messages(self, messages: Iterable) -> int:
        message_list = list(messages)
        cloud_count = self._count_dashscope_messages(
            [self._to_dashscope_message(message) for message in message_list],
        )
        if cloud_count is not None:
            return cloud_count
        return sum(self.count_message(message) for message in message_list)

    def _count_dashscope_messages(self, messages: list[dict[str, str]]) -> int | None:
        if not self._cloud_available or not messages:
            return None
        cache_key = tuple((message["role"], message["content"]) for message in messages)
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            import dashscope

            response = dashscope.Tokenization.call(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
            )
            token_count = self._extract_token_count(response)
            if token_count is None:
                self._cloud_available = False
            else:
                self._cache[cache_key] = token_count
            return token_count
        except Exception:
            self._cloud_available = False
            return None

    @classmethod
    def _extract_token_count(cls, response) -> int | None:
        candidates = [
            cls._lookup(response, "usage.input_tokens"),
            cls._lookup(response, "usage.total_tokens"),
            cls._lookup(response, "output.input_tokens"),
            cls._lookup(response, "output.token_count"),
            cls._lookup(response, "output.tokens"),
        ]
        for value in candidates:
            if isinstance(value, int):
                return value
            if isinstance(value, list):
                return len(value)
            if isinstance(value, str) and value.isdigit():
                return int(value)
        return None

    @staticmethod
    def _lookup(value, path: str):
        current = value
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
            if current is None:
                return None
        return current

    @classmethod
    def _to_dashscope_message(cls, message) -> dict[str, str]:
        return {
            "role": cls._message_role(message),
            "content": cls._content_to_text(getattr(message, "content", "")),
        }

    @staticmethod
    def _message_role(message) -> str:
        name = message.__class__.__name__.lower()
        if "human" in name:
            return "user"
        if "ai" in name:
            return "assistant"
        if "system" in name:
            return "system"
        return "user"

    @staticmethod
    def _content_to_text(content) -> str:
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

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        non_cjk = re.sub(r"[\u4e00-\u9fff]", " ", text)
        words = len(re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", non_cjk))
        return cjk_chars + words
