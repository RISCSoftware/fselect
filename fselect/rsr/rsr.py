import numpy as np

from fselect.util.utils import get_feature_scores_from_matrix_row_norm


def rsr(
    X: np.ndarray,
    lambda_: float = 1.0,
    eps: float = 1e-6,
    max_iter: int = 10000,
    delta_stop: float = 1e-6,
    verbose=False,
) -> tuple[np.ndarray, np.ndarray]:
    """Unsupervised feature selection based on:

    Zhu, P., Zuo, W., Zhang, L., Hu, Q., & Shiu, S. C. (2015). Unsupervised feature selection by regularized self-representation. Pattern Recognition, 48(2), 438-446.

    Learn a matrix W to minimize $$\| X - X W \|_{2,1} + \lambda \|W\|_{2,1}$$.

    Args:
        X (np.ndarray): Data matrix, rows are observations.
        lambda_ (float, optional): Regularization parameter, higher means more regularization. Defaults to 1.0.
        eps (float, optional): Denominator is increased to at least eps for numerical stability. Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10000.
        delta_stop (float, optional): Iteration stops when the L2,1 norm of delta W is less than delta_stop. Defaults to 1e-6.
        verbose (bool, optional): Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]: Returns feature scores and W. The L2 norm of the rows of W are the feature selection scores. A higher score means a more important feature.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be an integer >= 1")

    n, d = X.shape

    # initialize
    GR = np.ones((d, 1))
    GL = np.ones((n, 1))

    # dim(W) = (d, d)

    for i in range(max_iter):
        H = ((1 / GR) * X.T) @ (GL * X)
        W = np.linalg.inv(H + lambda_ * np.eye(d)) @ H
        # compute L2 norm of rows
        GR_inv = 2 * np.linalg.norm(W, axis=1, keepdims=True)
        GR = 1 / GR_inv.clip(eps, np.inf)
        # compute residual
        E = X - X @ W
        # compute L2 norm of residual rows
        GL_inv = 2 * np.linalg.norm(E, axis=1, keepdims=True)
        GL = 1 / GL_inv.clip(eps, np.inf)

        if i > 0:
            delta = np.linalg.norm(W_old - W, axis=1).sum()
            if verbose:
                print(f"iteration {i}, delta = {delta}")
            if delta < delta_stop:
                break

        W_old = W

    feature_scores = get_feature_scores_from_matrix_row_norm(W, norm_order=2)
    return feature_scores, W
