import numpy as np

from utils import model_fit, mse


def cross_validation(Z, y, K):
    """ Applies cross validation to the data

    This module applies K-fold cross validation on the input data to avoid over-fitting on the data.
    It includes a parameter `K`, which is the number of folds to experiment on.

    Args:
        Z (np.ndarray): Feature engineered data of shape (N, D').
        y (np.ndarray): Regression labels array of shape (N, 1) .
        K (int): Number of folds to run cross validation on.

    Returns:
        float: mean MSE error for cross validation.
    """
    chunk_length = len(Z) // K

    sum_cross_val_loss = 0

    for k in range(K):
        test_start = k * chunk_length
        test_stop = (k + 1) * chunk_length

        Z_test = Z[test_start: test_stop, :]
        y_test = y[test_start: test_stop, :]

        Z_train = np.vstack((Z[: test_start, :], Z[test_stop:, :]))
        y_train = np.vstack((y[: test_start, :], y[test_stop:, :]))

        w = model_fit(Z_train, y_train)

        y_test_preds = Z_test @ w
        mse_test = mse(y_test, y_test_preds)
        sum_cross_val_loss += mse_test

    mean_cross_val_loss = sum_cross_val_loss / K
    return mean_cross_val_loss
