import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import ElasticNet
import pandas as pd

def prox_ENbt(A, Xt, Om, gamma, d, x0, lam, L, eta, maxits, tol):
    """
    Python translation of the R function prox_ENbt.
    Uses backtracking line search for Elastic Net–style proximal gradient steps.
    """

    # Keep original L for possible resets (R code leaves this but doesn't reset every iteration)
    origL = L
    lam = float(lam)

    # Initialization
    x = x0.copy()
    n = x.shape[0]

    oneMat = np.ones((n, 1))
    zeroMat = np.zeros((n, 1))

    for k in range(maxits + 1):
        # Gradient of differentiable part: 0.5 * x' A x - d' x
        df = A @ x - d

        # Initialize error vector and cardinality
        err = np.zeros((n, 1))
        card = 0

        # For each i, update if in the support
        for i in range(n):
            if abs(x[i, 0]) > 1e-12:
                card += 1
                err[i, 0] = -df[i, 0] - lam * np.sign(x[i, 0])

        # Stopping criteria: -df(x) in subdiff g(x)
        if max(np.linalg.norm(df, np.inf) - lam,
               np.linalg.norm(err, np.inf)) < tol * n:
            return {
                "x": x,
                "k": k,
                "L": L
            }

        # Backtracking line search
        alpha = 1.0 / L
        pL = np.sign(x - alpha * df) * np.maximum(
            np.abs(x - alpha * df) - lam * alpha * oneMat,
            zeroMat
        )
        pTilde = pL - x
        gap = 0.5 * (L * np.linalg.norm(pTilde, 2)**2 -
                     (pTilde.T @ A @ pTilde)[0, 0])

        while gap < -tol:
            L *= eta
            alpha = 1.0 / L
            pL = np.sign(x - alpha * df) * np.maximum(
                np.abs(x - alpha * df) - lam * alpha * oneMat,
                zeroMat
            )
            pTilde = pL - x
            gap = 0.5 * (L * np.linalg.norm(pTilde, 2)**2 -
                         (pTilde.T @ A @ pTilde)[0, 0])

        x = pL

    return {
        "x": x,
        "k": maxits,
        "L": L
    }


# Load dataset from R
# A = pd.read_csv("../A.csv").to_numpy()
# d = pd.read_csv("../d.csv").to_numpy()
# x0 = pd.read_csv("../x0.csv").to_numpy()

# # Parameters
# lam = 0.1
# L = 1
# eta = 1.1
# maxits = 1000
# tol = 1e-6

# Xt = A
# Om = np.eye(A.shape[0])
# gamma = 0.1

# # Run Python function
# res_py = prox_ENbt(A=A, Xt=Xt, Om=Om, gamma=gamma,
#                    d=d, x0=x0, lam=lam, L=L, eta=eta,
#                    maxits=maxits, tol=tol)

# print("Python results:")
# print(res_py["x"])
# print(f"Iterations: {res_py['k']}")
# print(f"Final L: {res_py['L']}")

# done