import numpy as np

from sklearn.preprocessing import PolynomialFeatures


def expand_basis(X, poly_deg, include_sin, include_log):
    # Z = expand_poly(X, p)
    poly_expansion = PolynomialFeatures(degree=poly_deg)

    Z_ls = [poly_expansion.fit_transform(X)]

    if include_sin:
        Z_ls.append(np.sin(3 * X))

    if include_log:
        Z_ls.append(1 + np.log(np.where(X < 0, 0, X) + 1))

    Z = np.hstack(Z_ls)
    return Z


def expand_poly(X, p):
    N, dim = X.shape
    Z_ = np.zeros((N, (p + 1) * dim))

    for i in range(p + 1):
        Z_[:, i * dim: (i + 1) * dim] = X ** (i + 1)

    Z = np.hstack([np.ones((N, 1)), Z_])
    return Z
