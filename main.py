import data_processing
from k_NN import k_NN

# Variables
trainingsize = int(input())
k = int(input())

((X_train, y_train), (X_test, y_test)) = data_processing.ExtractData(trainingsize)


classifier = k_NN(k, X_train, y_train)

print(classifier.Test(X_test, y_test))