"""Shared PostgreSQL schema helpers."""
from typing import Any, Optional

from psycopg import sql


def connect_kwargs(config: Any, database: Optional[str] = None) -> dict:
    return {
        "host": config.postgres_host,
        "port": config.postgres_port,
        "user": config.postgres_user,
        "password": config.postgres_password,
        "dbname": database or config.postgres_database,
        "connect_timeout": config.postgres_connect_timeout_seconds,
    }


def ensure_knowledge_base_schema(cursor, config: Any) -> None:
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    cursor.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {table} (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                chunk_uid TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                create_time TIMESTAMPTZ NOT NULL DEFAULT now(),
                embedding vector({dim}) NOT NULL
            )
            """
        ).format(
            table=sql.Identifier(config.knowledge_table),
            dim=sql.Literal(config.dense_vector_dim),
        )
    )
    cursor.execute(
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index}
            ON {table}
            USING hnsw (embedding vector_cosine_ops)
            """
        ).format(
            index=sql.Identifier(f"{config.knowledge_table}_embedding_hnsw_idx"),
            table=sql.Identifier(config.knowledge_table),
        )
    )
    cursor.execute(
        sql.SQL(
            """
            DROP INDEX IF EXISTS {old_index}
            """
        ).format(
            old_index=sql.Identifier(f"{config.knowledge_table}_bm25_idx"),
        )
    )
    cursor.execute(
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {index}
            ON {table}
            USING bm25 (id, (text::pdb.jieba), source)
            WITH (key_field='id')
            """
        ).format(
            index=sql.Identifier(f"{config.knowledge_table}_bm25_jieba_idx"),
            table=sql.Identifier(config.knowledge_table),
        )
    )


