import numpy as np
import pandas as pd

def ExtractData():
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
    return cars

def SplitData(cars):
    """
    Takes the Dataframe object and splits it into independent variables and a dependent variable
    """
    X = cars.drop("safety", axis=1)
    X = X.values
    y = cars["safety"]
    y = y.values
    return (X,y)

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