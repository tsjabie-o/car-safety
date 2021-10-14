import numpy as np
import pandas as pd

def ExtractData(trainingsize):
    """
    Takes the input from the specified file and processes it to
    return a pandas Dataframe object
    """
    cars = list()

    with open("./data/car.data") as data:
        for line in data:
            features = line.strip().split(",")
            for i in range(len(features)):
                features[i] = ToNumerical(features[i], i)
            cars.append(features)
    cars = np.array(cars)
    cars = pd.DataFrame(
        cars, 
        columns=[
                "buying", "maint", "doors", "persons", "lug_boot", "safety", "class"
            ]
        )
    return SplitData(cars, trainingsize)

def SplitData(cars, trainingsize):
    """
    Takes the Dataframe object and splits it into independent variables and a dependent variable
    """
    cars_train = cars.iloc[0:trainingsize - 1, 0:]
    cars_test = cars.iloc[trainingsize:1727, 0:]

    # make training set

    X_train = cars_train.drop("safety", axis=1)
    X_train = X_train.values
    y_train = cars_train["safety"]
    y_train = y_train.values

    # Make test set

    X_test = cars_test.drop("safety", axis=1)
    X_test = X_test.values
    y_test = cars_test["safety"]
    y_test = y_test.values

    return ((X_train, y_train), (X_test, y_test))

def ToNumerical(text, i):
    """
    Uses text and an index to change strings from the data to 
    a numerical value. This makes it easier to compute distances
    """
    database = {
        0:{
            "low":0,
            "med":1,
            "high": 2,
            "vhigh": 3
        },
        1:{
            "low":0,
            "med":1,
            "high": 2,
            "vhigh": 3
        },
        2:{
            "2":0,
            "3":1,
            "4":2,
            "5more":3
        },
        3:{
            "2":0,
            "4":1,
            "more":2
        },
        4:{
            "small":0,
            "med":1,
            "big":2
        },
        5:{
            "low":0,
            "med":1,
            "high": 2,
        },
        6:{
            "unacc":0,
            "acc":1,
            "good":2,
            "vgood":3
        }
    }
    return database[i][text]