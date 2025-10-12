import asyncpg
from pgvector.asyncpg import register_vector
import numpy as np
from typing import List, Dict, Tuple

from core.config import settings

class DatabaseManager:
    """Manages all asynchronous database interactions using asyncpg."""
    def __init__(self):
        self.pool = None

    async def _ensure_vector_extension_exists(self):
        conn = None
        try:
            conn = await asyncpg.connect(dsn=settings.DATABASE_URL)
            print("Ensuring pgvector extension exists...")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("pgvector extension is enabled.")
        finally:
            if conn:
                await conn.close()

    async def connect_pool(self):
        if self.pool:
            return
        await self._ensure_vector_extension_exists()
        print("Creating database connection pool...")
        self.pool = await asyncpg.create_pool(
            dsn=settings.DATABASE_URL,
            min_size=5,
            max_size=20,
            init=register_vector
        )
        print("Database connection pool created successfully.")

    async def close_pool(self):
        if self.pool:
            print("Closing database connection pool...")
            await self.pool.close()
            self.pool = None
            print("Database connection pool closed.")

    async def create_table(self):
        """ # 3. Otimizar a Tabela: Manter apenas o índice HNSW para cosseno."""
        print("Ensuring table 'image_embeddings' and HNSW index exist...")
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS image_embeddings (
                        id SERIAL PRIMARY KEY,
                        filename VARCHAR(255) NOT NULL,
                        filepath VARCHAR(4096) NOT NULL UNIQUE,
                        embedding vector({settings.EMBEDDING_DIM}) NOT NULL
                    );
                """)
                # Remover índices antigos/concorrentes para focar no HNSW com cosseno
                await conn.execute("DROP INDEX IF EXISTS embedding_hnsw_idx;")
                await conn.execute("DROP INDEX IF EXISTS idx_ivfflat_cosine;")
                # Criar o índice HNSW otimizado para Similaridade de Cosseno
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS embedding_cosine_idx
                    ON image_embeddings USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 32, ef_construction = 150);
                """)
        print("Table and HNSW Cosine index are ready.")

    async def insert_embeddings_batch(self, records: List[Tuple[str, str, np.ndarray]]):
        if not records:
            return
        async with self.pool.acquire() as conn:
            query = """
                INSERT INTO image_embeddings (filename, filepath, embedding)
                VALUES ($1, $2, $3)
                ON CONFLICT (filepath) DO NOTHING;
            """
            await conn.executemany(query, records)

    async def search_similar(self, embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """# 1. Simplificar a Query: Usar a busca HNSW direta e eficiente."""
        query = """
            SELECT id, filename, filepath, 1 - (embedding <=> $1) as similarity
            FROM image_embeddings
            ORDER BY embedding <=> $1
            LIMIT $2;
        """
        async with self.pool.acquire() as conn:
            # 2. Ajustar a Precisão: Definir o parâmetro de busca para HNSW.
            # Valores maiores = mais preciso, mais lento. Valores menores = mais rápido, menos preciso.
            await conn.execute("SET LOCAL hnsw.ef_search = 100;")
            rows = await conn.fetch(query, embedding, top_k)
            return [dict(row) for row in rows]

    async def list_records(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        async with self.pool.acquire() as conn:
            query = "SELECT id, filename, filepath FROM image_embeddings ORDER BY id LIMIT $1 OFFSET $2;"
            rows = await conn.fetch(query, limit, offset)
            return [dict(row) for row in rows]
