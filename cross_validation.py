import numpy as np

from utils import model_fit


def cross_validation(Z, y, K):
    """ Applies cross validation to the data

    This module applies K-fold cross validation on the input data to avoid over-fitting on the data.
    It includes a parameter `K`, which is the number of folds to experiment on.

    Args:
        Z (np.ndarray): Feature engineered data of shape (N, D).
        y (np.ndarray): Regression labels array of shape (N, 1) .
        K (int): Number of folds to run cross validation on.

    Returns:
        float: mean MSE error for cross validation.
    """
    chunk_length = len(Z) // K

    R_cross_eval = 0

    for k in range(K):
        test_start = k * chunk_length
        test_stop = (k + 1) * chunk_length

        Z_test = Z[test_start: test_stop, :]
        y_test = y[test_start: test_stop, :]

        Z_train = np.vstack((Z[: test_start, :], Z[test_stop:, :]))
        y_train = np.vstack((y[: test_start, :], y[test_stop:, :]))

        w = model_fit(Z_train, y_train)

        R_test = ((Z_test @ w - y_test) ** 2).mean()
        R_cross_eval += R_test

    mean_R = R_cross_eval / K
    return mean_R
