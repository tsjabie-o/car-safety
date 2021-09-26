def ExtractData():
    cars = list()

    with open("./data/car.data") as data:
        for line in data:
            features = line.split(",")
            