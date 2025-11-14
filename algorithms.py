import numpy as np

class PCA:
    def __init__(self, k=None):
        self.k = k
        self.mean = None
        self.std = None
        self.components = None
        self.S = None

    def fit(self, X):
        X = X / 255

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0, ddof=1)

        self.std[self.std == 0] = 1.0

        X_norm = (X - self.mean) / self.std

        U, S, Vt = np.linalg.svd(X_norm, full_matrices=False)
        self.S = S

        if self.k is not None:
            self.components = Vt[:self.k]
        else:
            self.components = Vt

        return np.dot(X_norm, self.components.T)

    def transform(self, X_new):
        X_new = X_new / 255
        X_norm = (X_new - self.mean) / self.std
        return np.dot(X_norm, self.components.T)

    def inverse_transform(self, Z):
        X_norm_recon = np.dot(Z, self.components)
        return (X_norm_recon * self.std + self.mean) * 255