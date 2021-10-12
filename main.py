import data_processing
from k_NN import k_NN
import numpy as np

X, y = data_processing.ExtractData()
k = 100


classifier = k_NN(k, X, y)

testDatapoint = np.array([1, 1, 1, 2, 0, 3])

testSafety = classifier.Classify(testDatapoint)

print(testSafety)