import numpy as np
from scipy import stats

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
        distances = np.linalg.norm(self.X - datapoint, axis=1)
        return distances

    def FindNearestNeighbors(self, distances):
        nnIDs = distances.argsort()[:self.k]
        return nnIDs

    def Vote(self, nnIDs):
        neighbourSafety = self.y[nnIDs]
        winner = stats.mode(neighbourSafety)
        return winner

    def Classify(self, datapoint):
        distances = self.CalcDistances(datapoint)
        nnIDs = self.FindNearestNeighbors(distances)
        winner = self.Vote(nnIDs)
        return winner

    def Test(self, X_test, y_test):
        amt_corr = 0
        for i in range(len(X_test)):
            prediction = self.Classify(X_test[i])
            actual = y_test[i]
            if actual == int(str(prediction)[23]):
                amt_corr += 1
        acc = amt_corr / len(X_test)
        return acc