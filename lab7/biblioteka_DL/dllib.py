import sys

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sacred.observers import MongoObserver, FileStorageObserver
from sacred import Experiment


ex = Experiment(save_git_info=False)
ex.observers.append(MongoObserver(url='mongodb://admin:IUM_2021@172.17.0.1:27017',
                                  db_name='sacred'))

ex.observers.append(FileStorageObserver('s444018_sacred_FileObserver'))

@ex.config
def my_config():
    epochs = "1000"


def drop_relevant_columns(imbd_data):
    imbd_data.drop(columns=["Poster_Link"], inplace=True)
    imbd_data.drop(columns=["Overview"], inplace=True)
    imbd_data.drop(columns=["Certificate"], inplace=True)
    return imbd_data


def lowercase_columns_names(imbd_data):
    imbd_data["Series_Title"] = imbd_data["Series_Title"].str.lower()
    imbd_data["Genre"] = imbd_data["Genre"].str.lower()
    imbd_data["Director"] = imbd_data["Director"].str.lower()
    imbd_data["Star1"] = imbd_data["Star1"].str.lower()
    imbd_data["Star2"] = imbd_data["Star2"].str.lower()
    imbd_data["Star3"] = imbd_data["Star3"].str.lower()
    imbd_data["Star4"] = imbd_data["Star4"].str.lower()
    return imbd_data


def data_to_numeric(imbd_data):
    imbd_data = imbd_data.replace(np.nan, '', regex=True)
    imbd_data["Gross"] = imbd_data["Gross"].str.replace(',', '')
    imbd_data["Gross"] = pd.to_numeric(imbd_data["Gross"], errors='coerce')
    imbd_data["Runtime"] = imbd_data["Runtime"].str.replace(' min', '')
    imbd_data["Runtime"] = pd.to_numeric(imbd_data["Runtime"], errors='coerce')
    imbd_data["IMDB_Rating"] = pd.to_numeric(imbd_data["IMDB_Rating"], errors='coerce')
    imbd_data["Meta_score"] = pd.to_numeric(imbd_data["Meta_score"], errors='coerce')
    imbd_data["Released_Year"] = pd.to_numeric(imbd_data["Released_Year"], errors='coerce')
    imbd_data = imbd_data.dropna()
    imbd_data = imbd_data.reset_index()
    imbd_data.drop(columns=["index"], inplace=True)
    return imbd_data


def create_train_dev_test(imbd_data):
    data_train, data_test = train_test_split(imbd_data, test_size=230, random_state=1, shuffle=True)
    data_test, data_dev = train_test_split(data_test, test_size=115, random_state=1, shuffle=True)
    data_test.to_csv("data_test.csv", encoding="utf-8", index=False)
    data_dev.to_csv("data_dev.csv", encoding="utf-8", index=False)
    data_train.to_csv("data_train.csv", encoding="utf-8", index=False)


def normalize_gross(imbd_data):
    imbd_data[["Gross"]] = imbd_data[["Gross"]] / 10000000
    return imbd_data


def prepare_dataset():
    df = pd.read_csv('biblioteka_DL/imdb_top_1000.csv')
    df = drop_relevant_columns(df)
    df_lowercase = lowercase_columns_names(df)
    df = data_to_numeric(df_lowercase)
    df = normalize_gross(df)
    return df


class LinearRegressionModel(torch.nn.Module):

    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


@ex.automain
def my_main(epochs, _run):
    # num_epochs = 1000
    # num_epochs = int(sys.argv[1])

    # number of epochs is parametrized
    try:
        num_epochs = int(epochs)
    except Exception as e:
        print(e)
        print("Setting default epochs value to 1000.")
        num_epochs = 1000

    df = prepare_dataset()
    data_train, data_test = train_test_split(df, random_state=1, shuffle=True)
    X_train = pd.DataFrame(data_train["Meta_score"], dtype=np.float64)
    X_train = X_train.to_numpy()
    y_train = pd.DataFrame(data_train["Gross"], dtype=np.float64)
    y_train = y_train.to_numpy()
    X_train = X_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    X_train = torch.from_numpy(X_train.astype(np.float32)).view(-1, 1)
    y_train = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
    input_size = 1
    output_size = 1
    model = nn.Linear(input_size, output_size)
    learning_rate = 0.0001
    l = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # forward feed
        y_pred = model(X_train.requires_grad_())

        # calculate the loss
        loss = l(y_pred, y_train)

        # backward propagation: calculate gradients
        loss.backward()

        # update the weights
        optimizer.step()

        # clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()

        if epoch % 100 == 0:
            print('epoch {}, loss {}'.format(epoch, loss.item()))

    X_test = pd.DataFrame(data_test["Meta_score"], dtype=np.float64)
    X_test = X_test.to_numpy()
    X_test = X_test.reshape(-1, 1)
    X_test = torch.from_numpy(X_test.astype(np.float32)).view(-1, 1)

    predictedSet = model(X_test).detach().numpy()

    gross_test_g = pd.DataFrame(data_test["Gross"], dtype=np.float64)
    gross_test_g = gross_test_g.to_numpy()
    gross_test_g = gross_test_g.reshape(-1, 1)

    pred = pd.DataFrame(predictedSet)
    pred.to_csv('result.csv')
    # save model
    torch.save(model, "model.pkl")

    predicted = []
    expected = []

    for i in range(0, len(X_test)):
        predicted.append(np.argmax(model(X_test[i]).detach().numpy(), axis=0))
        expected.append(gross_test_g[i])

    for i in range(0, len(expected)):
        expected[i] = expected[i][0]

    rmse = mean_squared_error(gross_test_g, pred, squared=False)
    mse = mean_squared_error(gross_test_g, pred)

    _run.log_scalar("RMSE", rmse)
    _run.log_scalar("MSE", mse)
    _run.info['epochs'] = epochs

# ex.run()
ex.add_artifact("model.pkl")

