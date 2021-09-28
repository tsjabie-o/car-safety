def ExtractData():
    cars = list()

    with open("./data/car.data") as data:
        for line in data:
            features = line.strip().split(",")
            for i in range(len(features)):
                features[i] = ToNumerical(features[i], i)
            cars.append(features)
    
    return cars
            
def ToNumerical(text, i):
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