import numpy as np


class EmbeddingUtils:
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    @staticmethod
    def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return 1 - EmbeddingUtils.cosine_similarity(vec1, vec2)
