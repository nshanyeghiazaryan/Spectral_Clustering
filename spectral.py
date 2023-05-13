import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import normalize

class SpectralClustering:
    def __init__(self, n_clusters, gamma=1.0):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.labels_ = None

    def fit(self, X):
        laplacian_matrix = self._construct_laplacian(X, gamma=self.gamma)
        e_values, e_vectors = eigsh(laplacian_matrix, k=self.n_clusters + 1, which='SM')
        eigen_matrix = e_vectors[:, 1:self.n_clusters + 1]
        eigen_matrix_normed = normalize(eigen_matrix)

        kmeans = KMeans(n_clusters=self.n_clusters)
        self.labels_ = kmeans.fit_predict(eigen_matrix_normed)

    def _construct_laplacian(self, X, gamma):
        X = np.asarray(X)
        distance_matrix = np.square(np.linalg.norm(X[:, np.newaxis] - X, axis=2))
        affinity_matrix = np.exp(-gamma * distance_matrix)
        np.fill_diagonal(affinity_matrix, 0)
        degree_matrix = np.diag(affinity_matrix.sum(axis=1))
        laplacian_matrix = degree_matrix - affinity_matrix

        return laplacian_matrix

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def predict(self):
        return self.labels_