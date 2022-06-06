import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, explained_variance_score, \
    mean_squared_error, mean_absolute_error


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
    df = pd.read_csv('data/imdb_top_1000.csv')
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

model = torch.load("model.pkl")

X_test = pd.DataFrame(data_test["Meta_score"], dtype=np.float64)
X_test = X_test.to_numpy()
X_test = X_test.reshape(-1, 1)
X_test = torch.from_numpy(X_test.astype(np.float32)).view(-1, 1)

predicted = model(X_test).detach().numpy()

gross_test_g = pd.DataFrame(data_test["Gross"], dtype=np.float64)
gross_test_g = gross_test_g.to_numpy()
gross_test_g = gross_test_g.reshape(-1, 1)

pred = pd.DataFrame(predicted)

predicted = []
expected = []

for i in range(0, len(X_test)):
    predicted.append(np.argmax(model(X_test[i]).detach().numpy(), axis=0))
    expected.append(gross_test_g[i])

for i in range(0, len(expected)):
    expected[i] = expected[i][0]

rmse = mean_squared_error(gross_test_g, pred, squared=False)
mse = mean_squared_error(gross_test_g, pred)
evr = explained_variance_score(gross_test_g, pred)
mae = mean_absolute_error(gross_test_g, pred)

res = f"Explained variance regression score: {evr}, RMSE: {rmse}, MSE: {mse}, MAE: {mae}"

with open('mae.txt', 'a+') as f:
    f.write(str(mae) + '\n')

with open('rmse.txt', 'a+') as f:
    f.write(str(rmse) + '\n')

with open('mse.txt', 'a+') as f:
    f.write(str(mse) + '\n')

with open('evr.txt', 'a+') as f:
    f.write(str(evr) + '\n')

with open('mae.txt') as f:
    mae_val = [float(line) for line in f if line]
    builds = list(range(1, len(mae_val) + 1))

with open('rmse.txt') as f:
    rmse_val = [float(line) for line in f if line]

with open('mse.txt') as f:
    mse_val = [float(line) for line in f if line]

with open('evr.txt') as f:
    evr_val = [float(line) for line in f if line]


ax = plt.gca()
ax.set_title('Build')

mae_line = ax.plot(mae_val, color='blue', label="MAE")
rmse_line = ax.plot(rmse_val, color='green', label="RMSE")
mse_line = ax.plot(mse_val, color='red', label="MSE")
evr_line = ax.plot(evr_val, color='orange', label="EVR")
ax.legend(bbox_to_anchor=(0., 1.01, 1.0, .1), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.show()
plt.savefig('metrics.png')
