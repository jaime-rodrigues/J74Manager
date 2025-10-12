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
        """Uses a single, temporary connection to ensure the vector extension is created."""
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
        """Creates the connection pool after ensuring the vector extension exists."""
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
        """Closes the connection pool. To be called on application shutdown."""
        if self.pool:
            print("Closing database connection pool...")
            await self.pool.close()
            self.pool = None
            print("Database connection pool closed.")

    async def create_table(self):
        """Creates the main table with a Cosine-based HNSW index."""
        print("Ensuring table 'image_embeddings' and indexes exist...")
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
                # Drop the old L2 index if it exists
                await conn.execute("DROP INDEX IF EXISTS embedding_hnsw_idx;")
                # Create an index optimized for Cosine Similarity
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS embedding_cosine_idx
                    ON image_embeddings USING hnsw (embedding vector_cosine_ops);
                """)
        print("Table and Cosine index are ready.")

    async def insert_embeddings_batch(self, records: List[Tuple[str, str, np.ndarray]]):
        if not records:
            return
        async with self.pool.acquire() as conn:
            query = """
                INSERT INTO image_embeddings (filename, filepath, embedding)
                VALUES ($1, $2, $3)
                ON CONFLICT (filepath) DO NOTHING;
            """
            processed_records = [(r[0], r[1], r[2]) for r in records]
            await conn.executemany(query, processed_records)

    async def search_similar(self, embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Asynchronously searches for similar images using Cosine Similarity."""
        async with self.pool.acquire() as conn:
            # Use the <=> operator for Cosine Distance
            # The result is Cosine Distance (0=identical, 1=orthogonal, 2=opposite)
            # We calculate `1 - distance` to get Cosine Similarity (-1 to 1)
            query = """
                SELECT id, filename, filepath, 1 - (embedding <=> $1) as similarity
                FROM image_embeddings
                ORDER BY embedding <=> $1
                LIMIT $2;
            """
            rows = await conn.fetch(query, embedding, top_k)
            # Convert asyncpg.Record to dict for Pydantic compatibility
            return [dict(row) for row in rows]

    async def list_records(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        async with self.pool.acquire() as conn:
            query = "SELECT id, filename, filepath FROM image_embeddings ORDER BY id LIMIT $1 OFFSET $2;"
            rows = await conn.fetch(query, limit, offset)
            return [dict(row) for row in rows]
