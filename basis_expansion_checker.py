import itertools

import numpy as np

from expand_basis import expand_basis
from cross_validation import cross_validation
from utils import visualize_cross_validation_mses


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

    cv_results = []

    poly_deg_ls = range(1, 8)
    include_sin_ls = [True, False]
    include_log_ls = [True, False]

    for _basis in itertools.product(poly_deg_ls, include_sin_ls, include_log_ls):
        Z = expand_basis(X, *_basis)
        mean_R = cross_validation(Z, y, K)

        if mean_R < least_R:
            least_R = mean_R
            basis = _basis

        cv_results.append([*_basis, mean_R])
    print(f"Minimal MSE basis: {basis} Least MSE Loss: {least_R}")
    visualize_cross_validation_mses(cv_results)
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