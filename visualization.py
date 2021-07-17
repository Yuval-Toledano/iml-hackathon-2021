import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}

DATA_PATH = r"Data/Dataset_crimes.csv"

def load_data(path):
    df = pd.read_csv(path, index_col=0)
    return df

def plot_crime_distro(df):
    df_count_crimes = df.groupby(["Primary Type"]).size().reset_index(name="Count")
    plt.bar(df_count_crimes["Primary Type"], df_count_crimes["Count"])
    plt.show()

def plot_date_distro(df):
    # needs to have a month column
    df_count_crimes = df.groupby(["Month"]).size().reset_index(name="Count")
    plt.bar(df_count_crimes["Month"], df_count_crimes["Count"])
    plt.show()

def plot_block_distro(df):
    df_count_crimes = df.groupby(["Block"]).size().reset_index(name="Count")
    plt.bar(df_count_crimes["Block"], df_count_crimes["Count"])
    plt.show()

def plot_unique_values(df):
    fig, axs = plt.subplots(1, 2)
    count_uniqes = df.apply(lambda col: col.nunique())
    axs[0].bar(list(df.columns.values), count_uniqes)
    axs[1].set_title("linear scale")
    axs[1].bar(list(df.columns.values), count_uniqes)
    axs[1].set_yscale('log')
    axs[1].set_title("log scale")
    plt.show()

def plot_unique_values(df):
    fig, axs = plt.subplots(1, 2)
    count_uniqes = df.apply(lambda col: col.nunique())
    axs[0].bar(list(df.columns.values), count_uniqes)
    axs[1].set_title("linear scale")
    axs[1].bar(list(df.columns.values), count_uniqes)
    axs[1].set_yscale('log')
    axs[1].set_title("log scale")
    plt.show()




def main():
    df = load_data(r"Data/Dataset_crimes.csv")
    # plot_crime_distro(df)
    # plot_block_distro(df)
    plot_unique_values(df)




if __name__ == "__main__":
    main()