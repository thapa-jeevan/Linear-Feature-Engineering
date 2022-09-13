import pandas as pd
import numpy as np

from expand_basis import expand_basis


def read_data():
    """ Reads the data for training and test

    It extracts training data inputs and labels from `traindata.txt` and test data inputs
    from `testinputs.txt`.

    Returns:
        np.ndarray: Input training data of shape (N, D).
        np.ndarray: Training data labels of shape (N, 1).
        np.ndarray: Input test data of shape (N', D).

    """
    df_train = pd.read_csv("traindata.txt", sep="   ", names=range(9), engine="python")

    df_train = df_train.sample(len(df_train))

    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values.reshape(-1, 1)

    X_test = pd.read_csv("testinputs.txt", sep="   ", names=range(8), engine="python").values
    return X_train, y_train, X_test


def model_fit(Z, y):
    """ Fits Linear Regression model on the data

    Args:
        Z (np.ndarray): Feature engineered inputs of shape (N, D').
        y (np.ndarray): Corresponding data labels of shape (N, 1).

    Returns:
        np.ndarray: Weight for fitted linear regression model of shape (D', 1).

    """
    w = np.linalg.inv(Z.T @ Z) @ (Z.T @ y)
    # w = np.linalg.lstsq(Z.T @ Z, (Z.T @ y), rcond=None)
    return w


def all_train_fit(Xtrain, ytrain, basis):
    """ Trains Linear Regression model of the basis on the whole training data

    Args:
        Xtrain: Input training data of shape (N, D).
        ytrain: Corresponding data labels of shape (N, 1).
        basis: Basis with the least cross validation MSE loss with values (poly_degree, include_sin, include_log).

    Returns:
        np.ndarray: Weight for fitted linear regression model of shape (D', 1).

    """
    Ztrain = expand_basis(Xtrain, *basis)
    w = model_fit(Ztrain, ytrain)
    return w


def model_predict(Xtest, w_ls, basis):
    """ Runs model prediction on test data with the fitted linear regression weight

    Args:
        Xtest: Input test data of shape (N, D).
        w_ls: Weight for fitted linear regression model of shape (D', 1).
        basis: Basis with the least cross validation MSE loss with values (poly_degree, include_sin, include_log).

    Returns:
        np.ndarray: Labels predicted by the linear regerssion model on test data.

    """
    Ztest = expand_basis(Xtest, *basis)
    ytest_preds = Ztest @ w_ls
    return ytest_preds


def mse(y_true, y_pred):
    """ (float) Computes MSE loss between true and prediction values """
    return ((y_true - y_pred) ** 2).mean()
