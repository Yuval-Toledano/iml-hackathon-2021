from preprocessing import *
import pickle
import random
import datetime

CLASSIFIER_PATH = "classifier.weights"
CLUSTER_PATH = "cluster_centers.pkl"


def predict(csv_path: str) -> int:
    X = load_and_clean(csv_path)
    print(X)
    with open(CLASSIFIER_PATH, "rb") as fh:
        classifier = pickle.load(fh)

    return classifier.predict(X)


def send_police_cars(X):
    with open(CLUSTER_PATH, "rb") as fh:
        locations = pickle.load(fh)

    result = []

    for x in X:
        try:
            date_time = datetime.datetime.strptime(str(x), "%m/%d/%Y %H:%M:%S %p")
        except ValueError:
            try:
                date_time = datetime.datetime.strptime(str(x), "%m/%d/%Y %H:%M")
            except:
                date_time = datetime.datetime.now()

        entry = []
        for l in locations:
            hour = random.choice([23, 0])
            new_datetime = datetime.datetime(
                date_time.year, date_time.month,
                date_time.day, hour, 0).strftime("%m/%d/%Y %H:%M:%S %p")
            entry.append((l[0], l[1], new_datetime))

        result.append(entry)

    return result
