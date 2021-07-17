import datetime
import random
import pickle

import pandas as pd
from sklearn.cluster import KMeans


def clustering(df):
    kmeans = KMeans(init= "k-means++",n_clusters= 30)
    kmeans.fit(df.values)
    return kmeans.cluster_centers_


def loaddata():
    df = pd.read_csv(r'./Data/Dataset_crimes.csv',usecols=["X Coordinate", "Y Coordinate"])
    df = df.dropna()
    return df


def send_police_cars(X):
    with open("./Data/cluster_centers.pkl", "rb") as fh:
        locations = pickle.load(fh)
    date_time = datetime.datetime.strptime(X, "%m/%d/%Y %H:%M:%S %p")

    result = []
    for l in locations:
        hour = random.choice([23, 0])
        new_datetime = datetime.datetime(
            date_time.year, date_time.month,
            date_time.day, hour, 0).strftime("%m/%d/%Y %H:%M:%S %p")
        result.append((l[0], l[1], new_datetime))

    return result

if __name__ == '__main__':
    result = send_police_cars("01/07/2021 05:32:00 AM")
    print(result)
    
