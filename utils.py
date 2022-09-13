import pandas as pd
import numpy as np

from expand_basis import expand_basis


def read_data():
    df_train = pd.read_csv("traindata.txt", sep="   ", names=range(9), engine="python")

    df_train = df_train.sample(len(df_train))

    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values.reshape(-1, 1)

    X_test = pd.read_csv("testinputs.txt", sep="   ", names=range(8), engine="python").values
    return X_train, y_train, X_test


def model_fit(Z, y):
    w = np.linalg.inv(Z.T @ Z) @ (Z.T @ y)
    # w = np.linalg.lstsq(Z.T @ Z, (Z.T @ y), rcond=None)
    return w


def all_train_fit(Xtrain, ytrain, basis):
    Ztrain = expand_basis(Xtrain, *basis)
    w = model_fit(Ztrain, ytrain)
    return w


def model_predict(Xtest, w_ls, basis):
    Ztest = expand_basis(Xtest, *basis)
    ytest_preds = Ztest @ w_ls
    return ytest_preds

