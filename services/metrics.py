import numpy as np
from scipy.spatial.distance import cosine, euclidean, cityblock

class MetricCalculator:
    """Calculates similarity and distance metrics between two embedding vectors."""

    def calculate_all_metrics(self, embedding1: np.ndarray, embedding2: np.ndarray) -> dict:
        """
        Calculates a set of metrics between two embedding vectors.

        Args:
            embedding1: The first embedding vector (e.g., from the query image).
            embedding2: The second embedding vector (e.g., from the database result).

        Returns:
            A dictionary containing all calculated metric scores.
        """
        # Ensure embeddings are numpy arrays
        vec1 = np.asarray(embedding1, dtype=np.float32)
        vec2 = np.asarray(embedding2, dtype=np.float32)

        # 1. Cosine Similarity: Measures the angle between two vectors.
        # The scipy function calculates cosine *distance* (1 - similarity).
        # So, we subtract from 1 to get the similarity score.
        # Score: 1.0 means identical direction, 0.0 means orthogonal, -1.0 means opposite.
        cosine_similarity = 1 - cosine(vec1, vec2)

        # 2. Euclidean Distance (L2 Norm): The straight-line distance between two points.
        # Score: Lower is better. 0.0 means identical.
        euclidean_distance = euclidean(vec1, vec2)

        # 3. Manhattan Distance (L1 Norm): The sum of the absolute differences of their coordinates.
        # Also known as "city block" distance.
        # Score: Lower is better. 0.0 means identical.
        manhattan_distance = cityblock(vec1, vec2)

        return {
            "cosine_similarity": {
                "score": float(cosine_similarity),
                "explanation": "Mede a similaridade de direção/semântica entre os vetores. 1.0 é semanticamente idêntico."
            },
            "euclidean_distance_l2": {
                "score": float(euclidean_distance),
                "explanation": "Mede a distância em linha reta entre os vetores no espaço. Quanto menor, mais similares."
            },
            "manhattan_distance_l1": {
                "score": float(manhattan_distance),
                "explanation": "Mede a distância somando as diferenças absolutas em cada dimensão. Quanto menor, mais similares."
            }
        }
