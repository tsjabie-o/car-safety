# Car-Safety

A small self-developed project in which I build a k-NN model to classify the safety ratings of cars.

In `data_preprocessing.py`, the dataset is loaded, processed and cleaned. `k_NN.py` contains the k_NN model class, which implements this model and all of the internal features. In `main.py` the previous two modules are imported to load the data, seperate the train and test set and create an instance of the k_NN model, which is trained. Then classification is performed on the test set.
