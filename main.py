import data_processing
from k_NN import k_NN
import numpy as np

X, y = data_processing.ExtractData()
k = 100


classifier = k_NN(k, X, y)

testDatapoint = np.array([3, 3, 0, 0, 0, 0])

testSafety = classifier.Classify(testDatapoint)

print(testSafety)