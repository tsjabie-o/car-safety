import numpy as np
from scipy import stats

class k_NN:

    def __init__(self, k, X, y):
        self.k = k
        self.X = X
        self.y = y
        print(self.X)

    def CalcDistances(self, datapoint):
        """
        Calculates and returns a vector of distances from entries in X
        to the datapoint
        """
        distances = np.linalg.norm(self.X - datapoint, axis=1)
        print(distances)
        return distances

    def FindNearestNeighbors(self, distances):
        nnIDs = distances.argsort()[:self.k]
        print(len(nnIDs))
        return nnIDs

    def Vote(self, nnIDs):
        neighbourSafety = self.y[nnIDs]
        print(neighbourSafety)
        winner = stats.mode(neighbourSafety)
        return winner

    def Classify(self, datapoint):
        distances = self.CalcDistances(datapoint)
        nnIDs = self.FindNearestNeighbors(distances)
        winner = self.Vote(nnIDs)
        return winner