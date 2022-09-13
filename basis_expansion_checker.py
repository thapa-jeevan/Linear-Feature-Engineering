import itertools

import numpy as np

from expand_basis import expand_basis
from cross_validation import cross_validation


def basis_expansion_chooser(X, y):
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
