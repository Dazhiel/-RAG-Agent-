"""
Knowledge base built on PostgreSQL/ParadeDB BM25 + dense vector retrieval.
"""
import hashlib
import os
import uuid
from datetime import datetime
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from psycopg import sql

from app.config import RAGConfig
from app.loaders import DocumentLoader
from app.postgres_schema import connect_kwargs, ensure_knowledge_base_schema


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_project_root(), path)


def _vector_literal(vector: List[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


class DedupService:
    """MD5 based content deduplication persisted in a local record file."""

    def __init__(self, record_path: str):
        self.record_path = record_path
        self._records: set = self._load_records()

    def _load_records(self) -> set:
        os.makedirs(os.path.dirname(self.record_path), exist_ok=True)
        if not os.path.exists(self.record_path):
            open(self.record_path, "w", encoding="utf-8").close()
            return set()
        with open(self.record_path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    @staticmethod
    def _md5(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def exists(self, content: str) -> bool:
        return self._md5(content) in self._records

    def mark(self, content: str) -> None:
        md5_hex = self._md5(content)
        self._records.add(md5_hex)
        with open(self.record_path, "a", encoding="utf-8") as f:
            f.write(md5_hex + "\n")


class TextSplitter:
    """Text splitter wrapper."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def split(self, text: str) -> List[str]:
        if len(text) <= self.config.max_split_threshold:
            return [text]
        return self._splitter.split_text(text)


class DenseEmbeddingService:
    """Dense embedding adapter for local BGE-style sentence-transformer models."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency sentence-transformers. Install it with: pip install sentence-transformers"
            ) from exc

        model_path = self._resolve_model_path(model_name)
        self.model = SentenceTransformer(model_path)

    @staticmethod
    def _resolve_model_path(model_name: str) -> str:
        if os.path.exists(model_name):
            return model_name

        project_model_path = _project_path(model_name)
        if os.path.exists(project_model_path):
            return project_model_path

        looks_local = (
            model_name.startswith(".")
            or model_name.startswith("/")
            or "\\" in model_name
            or model_name.lower().startswith("models/")
        )
        if looks_local:
            raise FileNotFoundError(
                f"Embedding model path does not exist: {model_name}. "
                "Download the model locally or update EMBEDDING_MODEL."
            )

        return model_name

    def encode(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]


class BgeReranker:
    """BGE reranker adapter. If unavailable, keeps original retrieval order."""

    def __init__(self, model_name: str):
        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            self.reranker = None
            return

        try:
            model_path = DenseEmbeddingService._resolve_model_path(model_name)
            self.reranker = FlagReranker(model_path, use_fp16=True)
        except Exception as exc:
            print(f"Warning: reranker unavailable, hybrid retrieval will continue without rerank: {exc}")
            self.reranker = None

    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Document]:
        if not docs or self.reranker is None:
            return docs[:top_k]

        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.compute_score(pairs, normalize=True)
        if not isinstance(scores, list):
            scores = [scores]

        ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
        results = []
        for doc, score in ranked[:top_k]:
            doc.metadata["rerank_score"] = float(score)
            results.append(doc)
        return results


class PostgresHybridRetriever:
    """LangChain-like retriever facade for Postgres hybrid search."""

    def __init__(self, kb: "KnowledgeBaseBuilder"):
        self.kb = kb

    def invoke(self, query: str) -> List[Document]:
        return self.kb.search(query)


class KnowledgeBaseBuilder:
    """Load, split, embed, index, and retrieve documents with Postgres."""

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.config.data_dir = _project_path(self.config.data_dir)
        self.config.md5_record_path = _project_path(self.config.md5_record_path)
        self.embedding = DenseEmbeddingService(self.config.embedding_model)
        self.reranker = BgeReranker(self.config.reranker_model)
        self.splitter = TextSplitter(self.config)
        self.dedup = DedupService(self.config.md5_record_path)
        self._ensure_storage()

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency psycopg. Install it with: pip install 'psycopg[binary]'"
            ) from exc

        return psycopg.connect(
            **connect_kwargs(self.config),
            row_factory=dict_row,
        )

    def _ensure_storage(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                ensure_knowledge_base_schema(cursor, self.config)
            conn.commit()

    def build_from_directory(self, directory: Optional[str] = None) -> int:
        directory = directory or self.config.data_dir
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory does not exist: {directory}")

        documents = DocumentLoader.load_directory(directory)
        total_chunks = 0

        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if self.dedup.exists(doc.page_content):
                print(f"[skip] {source} already indexed")
                continue

            total_chunks += self.add_text(
                content=doc.page_content,
                source=source,
                mark_dedup=False,
            )
            self.dedup.mark(doc.page_content)

        print(f"[done] indexed {total_chunks} chunks")
        return total_chunks

    def add_text(self, content: str, source: str = "manual", mark_dedup: bool = True):
        if mark_dedup and self.dedup.exists(content):
            return "[skip] content already exists"

        chunks = self.splitter.split(content)
        dense_vectors = self.embedding.encode(chunks)
        now = datetime.now()

        rows = [
            (
                uuid.uuid4().hex,
                chunk,
                source,
                now,
                _vector_literal(dense_vectors[index]),
            )
            for index, chunk in enumerate(chunks)
        ]

        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(
                    sql.SQL(
                        """
                        INSERT INTO {table}
                            (chunk_uid, text, source, create_time, embedding)
                        VALUES (%s, %s, %s, %s, %s::vector)
                        """
                    ).format(table=sql.Identifier(self.config.knowledge_table)),
                    rows,
                )
            conn.commit()

        if mark_dedup:
            self.dedup.mark(content)
            return f"[ok] {source}: indexed {len(chunks)} chunks"
        return len(chunks)

    def search(self, query: str) -> List[Document]:
        query_vector = _vector_literal(self.embedding.encode([query])[0])
        limit = self.config.retrieval_candidates
        rrf_k = self.config.rrf_k

        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        """
                        WITH bm25_ranked AS (
                            SELECT
                                id,
                                pdb.score(id) AS bm25_score,
                                row_number() OVER (ORDER BY pdb.score(id) DESC) AS bm25_rank
                            FROM {table}
                            WHERE text ||| %s
                            ORDER BY pdb.score(id) DESC
                            LIMIT %s
                        ),
                        dense_ranked AS (
                            SELECT
                                id,
                                1 - (embedding <=> %s::vector) AS dense_score,
                                row_number() OVER (ORDER BY embedding <=> %s::vector ASC) AS dense_rank
                            FROM {table}
                            ORDER BY embedding <=> %s::vector ASC
                            LIMIT %s
                        ),
                        fused AS (
                            SELECT
                                COALESCE(b.id, d.id) AS id,
                                b.bm25_score,
                                d.dense_score,
                                b.bm25_rank,
                                d.dense_rank,
                                COALESCE(1.0 / (%s + b.bm25_rank), 0.0)
                                    + COALESCE(1.0 / (%s + d.dense_rank), 0.0)
                                    AS retrieval_score
                            FROM bm25_ranked b
                            FULL OUTER JOIN dense_ranked d ON b.id = d.id
                        )
                        SELECT
                            k.id,
                            k.text,
                            k.source,
                            k.create_time,
                            fused.bm25_score,
                            fused.dense_score,
                            fused.bm25_rank,
                            fused.dense_rank,
                            fused.retrieval_score
                        FROM fused
                        JOIN {table} k ON k.id = fused.id
                        ORDER BY fused.retrieval_score DESC, fused.bm25_rank NULLS LAST, fused.dense_rank NULLS LAST
                        LIMIT %s
                        """
                    ).format(table=sql.Identifier(self.config.knowledge_table)),
                    (
                        query,
                        limit,
                        query_vector,
                        query_vector,
                        query_vector,
                        limit,
                        rrf_k,
                        rrf_k,
                        limit,
                    ),
                )
                rows = cursor.fetchall()

        docs = [
            Document(
                page_content=row["text"],
                metadata={
                    "source": row["source"],
                    "create_time": row["create_time"].isoformat(),
                    "retrieval_score": float(row["retrieval_score"]),
                    "bm25_score": float(row["bm25_score"]) if row["bm25_score"] is not None else None,
                    "dense_score": float(row["dense_score"]) if row["dense_score"] is not None else None,
                    "bm25_rank": int(row["bm25_rank"]) if row["bm25_rank"] is not None else None,
                    "dense_rank": int(row["dense_rank"]) if row["dense_rank"] is not None else None,
                },
            )
            for row in rows
        ]

        return self.reranker.rerank(query, docs, self.config.retrieval_top_k)
    def get_retriever(self):
        return PostgresHybridRetriever(self)
