import itertools

import numpy as np

from expand_basis import expand_basis
from cross_validation import cross_validation


def basis_expansion_chooser(X, y):
    """ Chooses basis on the based of cross validation error

    Args:
        X (np.ndarray): Input training data of shape (N, D).
        y (np.ndarray): Training data labels of shape (N, 1).

    Returns:
        tuple: Basis with the least cross validation MSE loss with values
        (poly_degree, include_sin, include_log).
    """
    least_R = np.inf
    basis = 0

    K = 5

    poly_deg_ls = range(1, 5)
    include_sin_ls = [True, False]
    include_log_ls = [True, False]

    for _basis in itertools.product(include_log_ls, include_sin_ls, poly_deg_ls):  # only has polynomial expansion
        Z = expand_basis(X, *_basis[::-1])
        mean_R = cross_validation(Z, y, K)

        if mean_R < least_R:
            least_R = mean_R
            basis = _basis

        print(_basis, "MSE: ", mean_R)
    print(f"Minimal MSE basis: {basis} Least MSE Loss: {least_R}")
    return basis


def get_expanded_data(X, comb):
    include_sin_ls = [comb[1]]
    include_log_ls = [comb[0]]
    poly_deg_ls = range(comb[2], comb[2] + 1)

    Z = None

    for _basis in itertools.product(include_log_ls, include_sin_ls, poly_deg_ls):  # only has polynomial expansion
        Z = expand_basis(X, *_basis[::-1])
        print(type(Z), Z.shape)

    return Z