import numpy as np
from numpy.linalg import norm, solve, cholesky
from scipy.linalg import solve_triangular
import prox_EN, prox_ENbt

def SDAP(Xt, Yt, Om, gam, lam, q, PGsteps, PGtol, maxits, tol, initTheta=None, bt=False, L=None, eta=None):
    """
    Elastic net sparse discriminant analysis with alternating direction method.

    Args:
        Xt: numpy array (n x p) input features
        Yt: numpy array (n x K) class indicator matrix
        Om: numpy array (p x p) regularization matrix
        gam: float regularization parameter
        lam: float sparsity parameter (lambda)
        q: int number of discriminant directions
        PGsteps: int max proximal gradient steps
        PGtol: float proximal gradient tolerance
        maxits: int max ADMM iterations
        tol: float convergence tolerance
        initTheta: optional numpy array (K x 1) initial theta vector
        bt: bool use backtracking or not in proximal gradient
        L: float initial Lipschitz constant for backtracking (required if bt=True)
        eta: float backtracking scale factor (required if bt=True)

    Returns:
        dict with keys:
          B: (p x q) matrix of beta coefficients
          Q: (K x q) matrix of discriminant vectors (theta)
          subits: int total proximal gradient iterations
          totalits: int total alternating iterations
    """

    n, p = Xt.shape
    _, K = Yt.shape

    subits = 0
    totalits = np.full(q, maxits)

    # Precompute matrix A
    A = 2 * (Xt.T @ Xt) / n + 2 * gam * Om
    alpha = 1 / norm(A, 'fro')

    # Compute L for backtracking if not provided
    if L is None:
        L = gam * norm(np.diag(np.diag(Om)), ord=np.inf) + (norm(Xt, 'fro') ** 2) / n
    origL = L

    D = (Yt.T @ Yt) / n
    R = cholesky(D)

    Q = np.ones((K, q))
    B = np.zeros((p, q))

    for j in range(q):
        L = origL
        Qj = Q[:, :j+1]

        def Mj(u):
            # u: (K, 1)
            # returns u - Qj (Qj' D u)
            QjTDu = Qj.T @ (D @ u)
            return u - Qj @ QjTDu

        # Initialize theta
        theta = np.random.rand(K, 1)
        theta = Mj(theta)
        if j == 0 and initTheta is not None:
            theta = initTheta / 10
        theta = theta / np.sqrt(float(theta.T @ D @ theta))

        # Initialize beta
        beta = np.zeros((p, 1))
        if norm(np.diag(np.diag(Om)) - Om, ord='fro') < 1e-15:
            ominv = 1 / np.diag(Om)
            rhs0 = Xt.T @ (Yt @ (theta / n))
            rhs = Xt @ ((ominv / n) * rhs0)
            tmp_partial = solve(np.eye(n) + Xt @ ((ominv / (gam * n))[:, None] * Xt.T), rhs)
            beta = (ominv / gam)[:, None] * rhs0 - (1 / gam ** 2) * (ominv[:, None] * (Xt.T @ tmp_partial))

        for its in range(maxits):
            d = 2 * Xt.T @ (Yt @ (theta / n))

            b_old = beta.copy()
            if not bt:
                betaObj = prox_EN(A, d, beta, lam, alpha, PGsteps, PGtol)
                beta = betaObj['x']
                k_iter = betaObj['k']
            else:
                betaObj = prox_ENbt(A, Xt, Om, gam, d, beta, lam, L, eta, PGsteps, PGtol)
                beta = betaObj['x']
                L = betaObj['L']
                k_iter = betaObj['k']

            subits += k_iter

            if norm(beta, 2) > 1e-12:
                b = Yt.T @ (Xt @ beta)
                y = solve_triangular(R.T, b, lower=True)
                z = solve_triangular(R, y, lower=False)
                tt = Mj(z)
                t_old = theta.copy()
                theta = tt / np.sqrt(float(tt.T @ D @ tt))

                db = norm(beta - b_old) / norm(beta, 2)
                dt = norm(theta - t_old) / norm(theta, 2)
            else:
                beta = np.zeros_like(beta)
                theta = np.zeros_like(theta)
                db = 0
                dt = 0

            if max(db, dt) < tol:
                totalits[j] = its + 1
                break

        # Make first element of theta positive for reproducibility
        if theta[0, 0] < 0:
            theta = -theta
            beta = -beta

        Q[:, j:j+1] = theta
        B[:, j:j+1] = beta

    total_iters_sum = np.sum(totalits)

    return {
        'B': B,
        'Q': Q,
        'subits': subits,
        'totalits': total_iters_sum
    }
