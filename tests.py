import pickle
import classifier
import numpy as np
import pandas as pd

csv_path = "Data/test_dataset.csv"


def load_test():
    with open("Data/test_dataset.pkl", 'rb') as fh:
        test_x, test_y = pickle.load(fh)

    csv = test_x.to_csv(index=False)

    with open(csv_path, 'w') as fw:
        fw.write(csv)

    return test_x, csv_path, test_y


def test_classifier():
    _, csv_path, test_y = load_test()
    prediction = classifier.predict(csv_path)
    loss = np.array(prediction != test_y)
    miss = np.count_nonzero(loss)
    print(miss)
    print(test_y.shape)


def test_cluster():
    test_police = pd.read_csv("Data/test_police.csv")
    dates = test_police["Date"]
    ret = classifier.send_police_cars(list(dates))
    print(ret)


if __name__ == "__main__":
    test_classifier()
    # test_cluster()
