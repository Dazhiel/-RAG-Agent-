"""Initialize PostgreSQL storage for the knowledge base.

Run:
    python scripts/init_postgres.py
"""

import os
import sys

from psycopg import sql

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import RAGConfig
from app.postgres_schema import connect_kwargs, ensure_knowledge_base_schema


def ensure_database(config: RAGConfig) -> None:
    import psycopg

    with psycopg.connect(**connect_kwargs(config, "postgres"), autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (config.postgres_database,),
            )
            if cursor.fetchone():
                return
            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(config.postgres_database)
                )
            )


def ensure_tables(config: RAGConfig) -> None:
    import psycopg

    with psycopg.connect(**connect_kwargs(config, config.postgres_database)) as conn:
        with conn.cursor() as cursor:
            ensure_knowledge_base_schema(cursor, config)
        conn.commit()


def main() -> None:
    config = RAGConfig()
    ensure_database(config)
    ensure_tables(config)
    print(f"PostgreSQL database initialized: {config.postgres_database}")


if __name__ == "__main__":
    main()


