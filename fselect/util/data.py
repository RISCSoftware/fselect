import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


def get_flat_standardized_noisy_mnist() -> np.ndarray:
    df = datasets.load_digits()["data"]
    df = df.reshape(len(df), -1)
    df = df + 0.1 * np.random.randn(*df.shape)
    X = StandardScaler().fit_transform(df)
    return X
