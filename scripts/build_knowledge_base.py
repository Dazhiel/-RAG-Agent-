"""
从 data/ 目录读取文档并构建 PostgreSQL/ParadeDB 知识库。

运行方式：
    python scripts/build_knowledge_base.py
"""
from app.config import RAGConfig
from app.knowledge_base import KnowledgeBaseBuilder


def main() -> None:
    config = RAGConfig()
    kb = KnowledgeBaseBuilder(config)
    count = kb.build_from_directory()
    print(f"已写入 {count} 个文本块。")


if __name__ == "__main__":
    main()
