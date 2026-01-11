import numpy as np
from numpy.linalg import norm, cholesky, solve
from scipy.linalg import solve_triangular
import SDAP, normalize, normalizetest, prox_EN, prox_ENbt

def SDAPcv(X, Y, folds, Om, gam, lams, q, PGsteps, PGtol, maxits, tol, feat, quiet, initTheta=None, bt=False, L=None, eta=None):
    n, p_orig = X.shape
    _, K = Y.shape

    # Pad if n not divisible by folds
    pad = 0
    if n % folds != 0:
        pad = int(np.ceil(n / folds) * folds - n)
        X = np.vstack([X, X[:pad, :]])
        Y = np.vstack([Y, Y[:pad, :]])
    
    n = X.shape[0]

    # Permute rows randomly
    perm = np.random.permutation(n)
    X = X[perm, :]
    Y = Y[perm, :]

    # Sort lambdas ascending
    lams = np.sort(lams)

    # Initialize indices for cross-validation splits
    nv = n // folds  # Number validation samples per fold
    vinds = np.arange(nv)
    tinds = np.arange(nv, n)

    nlam = len(lams)

    # Initialize validation scores and misclassification matrix
    scores = np.ones((folds, nlam)) * q * p_orig
    mc = np.zeros((folds, nlam))

    # Helper normalize functions assumed implemented elsewhere:
    # normalize(X) returns dict with keys: 'Xc', 'mx', 'vx', 'Id'
    # normalizetest(Xtst, Xn) normalizes test data Xtst given training norm Xn

    for f in range(folds):
        # Training and validation splits
        Xt = X[tinds, :]
        Yt = Y[tinds, :]
        Xv = X[vinds, :]
        Yv = Y[vinds, :]

        # Normalize training and validation
        Xt_norm = normalize(Xt)  # should return dict
        Xt = Xt_norm['Xc']
        Xv = normalizetest(Xv, Xt_norm)

        nt, p = Xt.shape
        if Om.shape[0] != p:
            if not quiet:
                print("Warning: Columns dropped in normalization to a total of p, setting Om to diag(p)")
            Om = np.eye(p)

        # Centroid matrix of training data
        D_inv = np.diag(1 / np.diag(Yt.T @ Yt))
        C = D_inv @ Yt.T @ Xt

        # Precompute matrices
        A = 2 * ((Xt.T @ Xt) / nt + gam * Om)
        alpha = 1 / norm(A, 'fro')
        L = 1 / alpha
        L = 2 * norm(np.diag(np.diag(Om * gam)), ord=np.inf) + 2 * norm(Xt, 'fro') ** 2
        origL = L
        D = (1 / nt) * (Yt.T @ Yt)
        R = cholesky(D)

        if not quiet:
            print("-------------------------------------------")
            print(f"Fold number: {f+1}")
            print("-------------------------------------------")

        B = np.zeros((p_orig, q, nlam))

        for ll in range(nlam):
            Q = np.ones((K, q))

            for j in range(q):
                L = origL

                Qj = Q[:, :j+1]

                # Mj function
                def Mj(u):
                    return u - Qj @ (Qj.T @ (D @ u))

                theta = np.random.uniform(size=(K, 1))
                theta = Mj(theta)
                if j == 0 and initTheta is not None:
                    theta = initTheta
                theta = theta / np.sqrt((theta.T @ D @ theta)[0, 0])

                if ll == 0:
                    if norm(np.diag(np.diag(Om)) - Om, 'fro') < 1e-15:
                        ominv = 1 / np.diag(Om)
                        rhs0 = Xt.T @ (Yt @ (theta / nt))
                        rhs = Xt @ ((ominv / nt) * rhs0)

                        # Solve partial system
                        tmp_partial = solve(np.eye(nt) + Xt @ ((ominv / (gam * nt))[:, None] * Xt.T), rhs)
                        beta = (ominv / gam) * rhs0 - (1 / gam ** 2) * ominv * (Xt.T @ tmp_partial)
                        beta = beta.reshape((-1, 1))
                    else:
                        beta = np.zeros((p, 1))
                else:
                    beta = B[:, j, ll - 1].reshape((p_orig, 1))
                    beta = beta[Xt_norm['Id'], :]

                for its in range(maxits):
                    d = 2 * Xt.T @ (Yt @ (theta / nt))

                    b_old = beta.copy()
                    if not bt:
                        beta_obj = prox_EN(A, d, beta, lams[ll], alpha, PGsteps, PGtol)
                        beta = beta_obj['x']
                    else:
                        beta_obj = prox_ENbt(A, Xt, Om, gam, d, beta, lams[ll], L, eta, PGsteps, PGtol)
                        beta = beta_obj['x']

                    if norm(beta, 2) > 1e-12:
                        b = Yt.T @ (Xt @ beta)
                        y = solve_triangular(R.T, b, lower=True)
                        z = solve_triangular(R, y, lower=False)
                        tt = Mj(z)
                        t_old = theta.copy()
                        theta = tt / np.sqrt((tt.T @ D @ tt)[0, 0])

                        db = norm(beta - b_old) / norm(beta, 2)
                        dt = norm(theta - t_old) / norm(theta, 2)
                    else:
                        beta = beta * 0
                        theta = theta * 0
                        db = 0
                        dt = 0

                    if max(db, dt) < tol:
                        break

                if theta[0] < 0:
                    theta = -theta
                    beta = -beta

                Q[:, j] = theta.flatten()
                B[Xt_norm['Id'], j, ll] = beta.flatten()

            PXtest = Xv @ B[Xt_norm['Id'], :, ll]
            PC = C @ B[Xt_norm['Id'], :, ll]

            dists = np.zeros((nv, K))
            for i in range(nv):
                for jj in range(K):
                    dists[i, jj] = norm(PXtest[i, :] - PC[jj, :])

            predicted_labels = np.argmin(dists, axis=1)

            Ypred = np.zeros((nv, K))
            Ypred[np.arange(nv), predicted_labels] = 1

            mc[f, ll] = (0.5 * norm(Yv - Ypred, 'fro') ** 2) / nv

            B_loc = B[:, :, ll]
            sum_B_loc_nnz = np.sum(B_loc != 0)

            if 1 <= sum_B_loc_nnz <= q * p_orig * feat:
                scores[f, ll] = mc[f, ll]
            elif sum_B_loc_nnz > q * p_orig * feat:
                scores[f, ll] = sum_B_loc_nnz

            if not quiet:
                print(f"f: {f+1} | ll: {ll+1} | lam: {lams[ll]} | feat: {sum_B_loc_nnz/(q*p_orig):.4f} | mc: {mc[f,ll]:.4f} | score: {scores[f,ll]}")

        tmp = tinds[:nv]

        if nv + 1 > nt:
            tinds = vinds
            vinds = tmp
        else:
            tinds = np.concatenate([tinds[nv:], vinds])
            vinds = tmp

    avg_score = np.mean(scores, axis=0)
    lbest = np.argmin(avg_score)
    lambest = lams[lbest]

    print(f"Finished Training: lam = {lambest}")

    Xt = X[:n - pad, :]
    Yt = Y[:n - pad, :]
    Xt_norm = normalize(Xt)
    Xt = Xt_norm['Xc']

    if p_orig != p:
        Om = np.eye(Xt.shape[1])

    # Assuming SDAP is implemented
    resBest = SDAP(Xt, Yt, Om, gam, lams[lbest], q, PGsteps, PGtol, maxits, tol, bt=bt)

    retOb = {
        'call': 'SDAPcv',
        'B': resBest['B'],
        'Q': resBest['Q'],
        'lbest': lbest,
        'lambest': lambest,
        'scores': scores
    }

    return retOb
