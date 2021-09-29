import numpy as np

class k_NN:

    def __init__(self, k, X, y):
        self.k = k
        self.X = X
        self.y = y

    def CalcDistances(self, datapoint):
        """
        Calculates and returns a vector of distances from entries in X
        to the datapoint
        """
        distances = np.linalg.norm(self.X - datapoint)
        return distances

    def FindNearestNeighbors(self, distances):
        nnIDs = distances.argsort()[:self.k]
        return nnIDs