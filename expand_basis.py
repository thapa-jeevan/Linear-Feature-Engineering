import numpy as np

from sklearn.preprocessing import PolynomialFeatures


def expand_basis(X, poly_deg, include_sin, include_log):
    """ Expands basis with polynomial, sine and logarithm functions

    Args:
        X (np.ndarray):
        poly_deg (int): Degree for polynomial expansion.
        include_sin  (bool): Includes sine features if True else not.
        include_log (bool): Includes log features if True else not.

    Returns:
        np.ndarray: Expanded feature vector of shape (N, D').
    """
    # poly_expansion = PolynomialFeatures(degree=poly_deg)
    # Z_ls = [poly_expansion.fit_transform(X)]

    Z_ls = [expand_poly(X, p)]

    if include_sin:
        Z_ls.append(np.sin(3 * X))

    if include_log:
        Z_ls.append(1 + np.log(np.where(X < 0, 0, X) + 1))

    Z = np.hstack(Z_ls)
    return Z


def expand_poly(X, p):
    """ Expands the polynomial basis of the input data

    Args:
        X (np.ndarray): Input data of shape (N, D).
        p (int): Polynomial Degree

    Returns:
        (np.ndarray): Expanded polynomial features.
    """
    N, dim = X.shape
    Z = np.hstack([np.ones((N, 1)), *[X ** i for i in range(1, p + 1)]])
    return Z

