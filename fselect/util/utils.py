import numpy as np


def get_feature_scores_from_matrix_row_norm(
    W: np.ndarray,
    norm_order: float = 2,
) -> np.ndarray:
    scores = np.linalg.norm(W, ord=norm_order, axis=1)
    return scores
