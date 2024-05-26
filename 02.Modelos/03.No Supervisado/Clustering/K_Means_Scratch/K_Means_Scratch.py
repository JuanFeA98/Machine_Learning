import pandas as pd
import numpy as np

class KMeans_Scratch:
    def __init__(self, n_clusters=3, max_iter=300, random_state=13):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initialize_centroids(self, X):
        # Inicializamos nuestra semilla
        np.random.seed(self.random_state)

        # Asignamos un valor aleatorio a las observaciones de nuestro dataframe
        random_idx = np.random.permutation(X.shape[0])

        # Traemos los primeras registros dependiendo de nuestra cantidad de clusters, estos seran los centroides
        centroids = X[random_idx[:self.n_clusters]]

        return centroids
    
    def compute_distances(self, X, centroids):
        # Calculamos la distancia entre cada observación contra los centroides
        distances = np.sqrt(
            ((X - centroids[:, np.newaxis]) ** 2).sum(axis=2)
        )

        return distances
    
    def update_centroids(self, X, labels):
        # Actualizamos el valor de los centroides con el promedio del grupo
        centroids = [X[labels == i].mean(axis=0) for i in range(self.n_clusters)]
        centroids = np.array(centroids)

        return centroids

    def compute_sse(self, X, labels):
        # Calculamos la distancia entre las observaciones y el centroide
        distances = np.sqrt(
            ((X - self.clusters_center[labels])**2).sum(axis=1)
        )

        sse = np.sum(distances**2)

        return sse
    
    def fit(self, X):
        # Inicializamos los centroides
        centroids = self.initialize_centroids(X)

        for i in range(self.max_iter):
            # Definimos el valor de los centroides anteriores
            old_centroids = centroids

            # Calculamos la distancia entre las observaciones y los centroides
            distances = self.compute_distances(X, centroids)

            # Definimos los labels de nuestras observaciones
            labels = np.argmin(distances, axis=0)

            # Actualizamos el valor de los centroides
            centroids = self.update_centroids(X, labels)

            # Si los centroides son igual que los de la iteración anterior salimos del ciclo
            if np.all(centroids == old_centroids):
                break

        self.clusters_center = centroids
        self.labels = labels
        self.inertia = self.compute_sse(X, labels)