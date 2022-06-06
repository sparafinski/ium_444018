import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def drop_relevant_columns():
    imbd_data.drop(columns=["Poster_Link"], inplace=True)
    imbd_data.drop(columns=["Overview"], inplace=True)


def lowercase_columns_names():
    imbd_data["Series_Title"] = imbd_data["Series_Title"].str.lower()
    imbd_data["Genre"] = imbd_data["Genre"].str.lower()
    imbd_data["Director"] = imbd_data["Director"].str.lower()
    imbd_data["Star1"] = imbd_data["Star1"].str.lower()
    imbd_data["Star2"] = imbd_data["Star2"].str.lower()
    imbd_data["Star3"] = imbd_data["Star3"].str.lower()
    imbd_data["Star4"] = imbd_data["Star4"].str.lower()


def gross_to_numeric():
    global imbd_data
    imbd_data = imbd_data.replace(np.nan, '', regex=True)
    imbd_data["Gross"] = imbd_data["Gross"].str.replace(',', '')
    imbd_data["Gross"] = pd.to_numeric(imbd_data["Gross"], errors='coerce')


def create_train_dev_test():
    data_train, data_test = train_test_split(imbd_data, test_size=230, random_state=1)
    data_test, data_dev = train_test_split(data_test, test_size=115, random_state=1)
    print("Dataset successfully divided into test/dev/train sets\n")
    data_test.to_csv("data_test.csv", encoding="utf-8", index=False)
    data_dev.to_csv("data_dev.csv", encoding="utf-8", index=False)
    data_train.to_csv("data_train.csv", encoding="utf-8", index=False)

    print("Data train description: ")
    print(data_train.describe(include="all"))
    print("\nData test description: ")
    print(data_test.describe(include="all"))
    print("\nData dev description: ")


imbd_data = pd.read_csv('imdb_top_1000.csv')

drop_relevant_columns()

lowercase_columns_names()

imbd_data = imbd_data.dropna()

gross_to_numeric()

create_train_dev_test()
