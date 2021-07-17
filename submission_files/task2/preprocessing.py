import pickle

import pandas as pd
import numpy as np
from pandas import Series
from sklearn.model_selection import train_test_split

CRIME_DICT = {'BATTERY': 0, 'THEFT': 1, 'CRIMINAL DAMAGE': 2,
              'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}

DROPPED_FEATURES = ["FBI Code", "ID", "IUCR", "Description",
                    "Case Number", "Year", "Updated On", "Date"]

DROPPED_LOCATION_FEATURES = ["Block", "District", "Ward", "Community Area", "Location Description", "Beat",
                             "X Coordinate", "Y Coordinate", "Latitude", "Longitude", "Location"]

HOT_ENCODINGS = ["Day", "Weekday", "Hour"]

LOCATIONS_DESCRIPTIONS = ["RESIDENCE - YARD(FRONT / BACK)", "GAS STATION", "VEHICLE NON-COMMERCIAL",
                          "GROCERY FOOD STORE", "RESTAURANT", "COMMERCIAL / BUSINESS OFFICE",
                          "ALLEY", "DEPARTMENT STORE", "RESIDENCE - PORCH / HALLWAY", "SMALL RETAIL STORE",
                          "PARKING LOT / GARAGE (NON RESIDENTIAL)", "SIDEWALK", "STREET", "RESIDENCE", "APARTMENT"]

BOOLEAN_FEATURES = ["Arrest", "Domestic"]

FEATURES_TESTS = {
    lambda df: df['Primary Type'] > 0,
}


def load_data(csv_path: str) -> pd.DataFrame:
    """
    df = 
    :param csv_path: 
    :return: 
    """
    return pd.read_csv(csv_path, parse_dates=["Date"],
                       dtype={'Location Description': 'string', 'Block': 'string', 'Description': 'string'})



def drop_invalid_features(df: pd.DataFrame) -> pd.DataFrame:
    for test in FEATURES_TESTS:
        df = df.loc[test]

    return df


def parse_date(df: pd.DataFrame) -> pd.DataFrame:
    df['Day'] = df['Date'].dt.day
    df['Hour'] = df['Date'].dt.hour
    df['Weekday'] = df['Date'].dt.weekday

    return df


def categorials_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    for he in HOT_ENCODINGS:
        df = pd.get_dummies(df, prefix=he, columns=[he])

    # Day
    rows, _ = df.shape
    zeros = np.zeros(rows)
    for i in range(24):
        name = f'Hour_{i}'
        if name not in df.columns:
            df[name] = zeros

    for i in range(7):
        name = f'Weekday_{i}'
        if name not in df.columns:
            df[name] = zeros

    for i in range(32):
        name = f'Day_{i}'
        if name not in df.columns:
            df[name] = zeros

    return df


def location_description_encoding(df: pd.DataFrame) -> pd.DataFrame:
    location_description_map = {}
    for i, location in enumerate(LOCATIONS_DESCRIPTIONS):
        location_description_map[location] = i

    df['LocationDescriptionLabel'] = df['Location Description'].map(location_description_map)
    df = pd.get_dummies(df, prefix="loc_dec", dtype=int, columns=['LocationDescriptionLabel'])

    return df


def drop_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(DROPPED_FEATURES, axis=1)


def drop_not_numbers(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def convert_boolean(df: pd.DataFrame) -> pd.DataFrame:
    for bf in BOOLEAN_FEATURES:
        df[bf] = df[bf].astype(int)

    return df


def drop_location_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(DROPPED_LOCATION_FEATURES, axis=1)


def clean_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    :return:
    """
    raw_data = raw_data.iloc[:, 1:]
    df = parse_date(raw_data)
    df = drop_features(df)
    df = categorials_hot_encoding(df)
    df = convert_boolean(df)
    df = drop_location_features(df)
    # df = location_description_encoding(df)
    # df = drop_not_numbers(df)

    return df


def extract_labels(df: pd.DataFrame) -> Series:
    label = df.pop('Primary Type')
    label = label.map(CRIME_DICT)
    return label


def split_train_validate_test(X, y):
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
    return train_x, train_y, test_x, test_y


def save_train_test(X, y, filename):
    with open(f"Data/{filename}_dataset.pkl", "wb") as fh:
        pickle.dump((X, y), fh)


def load_and_clean(csv_path: str) -> pd.DataFrame:
    raw_data = load_data(csv_path)
    print(raw_data)
    df = clean_data(raw_data)
    return df


if __name__ == "__main__":
    raw_data = load_data("Data/Dataset_crimes.csv")
    labels = extract_labels(raw_data)
    train_x, train_y, test_x, test_y = split_train_validate_test(raw_data, labels)

    save_train_test(train_x, train_y, 'train')
    # save_train_test(cleaned_validate_x, validate_y, 'validate')
    save_train_test(test_x, test_y, 'test')
