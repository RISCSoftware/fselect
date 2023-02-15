import numpy as np
import pandas as pd

from fselect.rsr.rsr import rsr
from fselect.util.data import get_flat_standardized_noisy_mnist


def rsr_demo() -> None:
    X = get_flat_standardized_noisy_mnist()
    # replace the lower right pixel by the mean of all other pixels
    X[:, -1] = X[:, :-1].mean(1)
    feature_scores, W = rsr(X, lambda_=100, verbose=True)
    print(feature_scores.reshape(8, 8))


if __name__ == "__main__":
    rsr_demo()
