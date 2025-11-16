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
            await self.pool.close()
            self.pool = None
            print("Database connection pool closed.")

    async def create_table(self):
        """Creates or updates the table to the latest schema."""
        print("Ensuring table 'image_embeddings' and indexes exist...")
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # 1. Create the table if it doesn't exist (for the first run)
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS image_embeddings (
                        id SERIAL PRIMARY KEY,
                        filename VARCHAR(255) NOT NULL,
                        filepath VARCHAR(4096) NOT NULL,
                        embedding vector({settings.EMBEDDING_DIM}) NOT NULL,
                        embedding_type VARCHAR(20) NOT NULL,
                        embedding_hash VARCHAR(64) NOT NULL
                    );
                """)

                # 3. Remove the old, problematic unique constraint if it exists
                #await conn.execute("ALTER TABLE image_embeddings DROP CONSTRAINT IF EXISTS image_embeddings_filepath_embedding_key;")

                # 4. Add the new, robust unique constraint using the hash
                # We also drop it first to make this operation idempotent
                #await conn.execute("ALTER TABLE image_embeddings DROP CONSTRAINT IF EXISTS image_embeddings_filepath_hash_key;")
                #await conn.execute("ALTER TABLE image_embeddings ADD CONSTRAINT image_embeddings_filepath_hash_key UNIQUE (filepath, embedding_hash);")
                
                # 5. Create other necessary indexes
                await conn.execute("CREATE INDEX IF NOT EXISTS embedding_cosine_idx ON image_embeddings USING hnsw (embedding vector_cosine_ops);")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_embedding_type ON image_embeddings(embedding_type);")

        print("Table schema is up to date.")

    async def insert_embeddings_batch(self, records: List[Tuple[str, str, np.ndarray, str, str]]):
        """Asynchronously inserts a batch of embeddings with their hash and type."""
        if not records:
            return
        async with self.pool.acquire() as conn:
            query = """
                INSERT INTO image_embeddings (filename, filepath, embedding, embedding_type, embedding_hash)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (filepath, embedding_hash) DO NOTHING;
            """
            await conn.executemany(query, records)

    async def search_similar(self, embedding: np.ndarray, top_k: int = 5, scope: str = 'original_only') -> List[Dict]:
        """
        Asynchronously searches for similar images with a configurable scope.
        """
        base_query = "SELECT id, filename, filepath, 1 - (embedding <=> $1) as similarity FROM image_embeddings"
        
        where_clause = "WHERE embedding_type = 'original'" if scope == 'original_only' else ""
        full_query = f"{base_query} {where_clause} ORDER BY embedding <=> $1 LIMIT $2;"
        
        async with self.pool.acquire() as conn:
            await conn.execute("SET LOCAL hnsw.ef_search = 100;")
            rows = await conn.fetch(full_query, embedding, top_k)
            return [dict(row) for row in rows]

    async def get_embedding_by_filepath(self, filepath: str) -> np.ndarray | None:
        """Retrieves the 'original' embedding vector for a specific image filepath."""
        query = "SELECT embedding FROM image_embeddings WHERE filepath = $1 AND embedding_type = 'original' LIMIT 1;"
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, filepath)
            return row['embedding'] if row else None

    async def list_records(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        async with self.pool.acquire() as conn:
            query = "SELECT id, filename, filepath, embedding_type FROM image_embeddings ORDER BY id LIMIT $1 OFFSET $2;"
            rows = await conn.fetch(query, limit, offset)
            return [dict(row) for row in rows]
