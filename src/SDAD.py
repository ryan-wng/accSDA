import numpy as np
from scipy.linalg import cholesky, solve_triangular, norm
import ADMM_EN2, ADMM_EN_SMW

def SDAD(Xt, Yt, Om, gam, lam, mu, q, PGsteps, PGtol, maxits, tol,
                 selector=None, initTheta=None):
    """
    Python translation of SDAD.default from R.
    Xt: n x p feature matrix
    Yt: n x K class indicator matrix
    Om: p x p Omega matrix
    gam, lam, mu: regularization parameters
    q: number of components
    PGsteps, PGtol: proximal gradient parameters
    maxits, tol: iteration limits and tolerances
    selector: binary selector vector for features
    initTheta: optional initial theta vector
    """

    nt, p = Xt.shape
    _, K = Yt.shape
    if selector is None:
        selector = np.ones(p)

    subits = 0
    totalits = np.full(q, maxits, dtype=int)

    # Check if Omega is diagonal (Frobenius norm < 1e-15)
    if norm(np.diag(np.diag(Om)) - Om, ord='fro') < 1e-15 and np.sum(selector) == len(selector):
        SMW = True
        M = mu + 2*gam*np.diag(Om)
        Minv = 1.0 / M
        # RS = chol(I + 2*Xt %*% (Minv/nt) * t(Xt))
        RS = cholesky(np.eye(nt) + 2 * Xt @ (np.diag(Minv) / nt) @ Xt.T, lower=True)
    else:
        SMW = False
        A = mu * np.eye(p) + 2 * (Xt.T @ Xt) / nt + 2 * gam * Om
        R2 = cholesky(A, lower=True)

    D = (1/nt) * (Yt.T @ Yt)
    R = cholesky(D, lower=True)

    Q = np.ones((K, q))
    B = np.zeros((p, q))

    for j in range(q):
        Qj = Q[:, :j+1]  # K x (j+1)

        def Mj(u):
            return u - Qj @ (Qj.T @ (D @ u))

        # Initialize theta
        theta = np.random.uniform(size=(K, 1))
        theta = Mj(theta)
        if j == 0 and initTheta is not None:
            theta = initTheta.reshape(-1,1)
        theta /= np.sqrt((theta.T @ D @ theta).item())

        d = 2 * (Xt.T @ (Yt @ (theta / nt)))

        # Initialize beta
        if SMW:
            btmp = Xt @ (Minv * d) / nt
            # Solve with forward/backsolve using RS (lower triangular)
            tmp = solve_triangular(RS, btmp, lower=True)
            tmp = solve_triangular(RS.T, tmp, lower=False)
            beta = (Minv * d) - 2 * Minv * (Xt.T @ tmp)
        else:
            # Solve R2 * R2.T * beta = d
            tmp = solve_triangular(R2, d, lower=True)
            beta = solve_triangular(R2.T, tmp, lower=False)

        for its in range(maxits):
            b_old = beta.copy()
            if SMW:
                betaObj = ADMM_EN_SMW(Minv, Xt, RS, d, beta, lam, mu, PGsteps, PGtol, True, selector)
                beta = betaObj['y']
                subits += betaObj['k']
            else:
                betaObj = ADMM_EN2(R2, d, beta, lam, mu, PGsteps, PGtol, True, selector)
                beta = betaObj['y']
                subits += betaObj['k']

            if norm(beta, 2) > 1e-15:
                b = Yt.T @ (Xt @ beta)
                y = solve_triangular(R.T, b, lower=False)
                z = solve_triangular(R, y, lower=True)
                tt = Mj(z)
                t_old = theta.copy()
                theta = tt / np.sqrt((tt.T @ D @ tt).item())

                db = norm(beta - b_old, 2) / norm(beta, 2)
                dt = norm(theta - t_old, 2) / norm(theta, 2)
            else:
                beta = np.zeros_like(beta)
                theta = np.zeros_like(theta)
                db, dt = 0, 0

            if max(db, dt) < tol:
                totalits[j] = its + 1
                break

        if theta[0] < 0:
            theta = -theta
            beta = -beta

        Q[:, j] = theta.ravel()
        B[:, j] = beta.ravel()

    return {
        'B': B,
        'Q': Q,
        'subits': subits,
        'totalits': totalits.sum()
    }
