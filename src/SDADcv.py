import numpy as np
from numpy.linalg import norm, cholesky, solve
from scipy.linalg import solve_triangular
import SDAD, normalize, normalizetest, ADMM_EN2, ADMM_EN_SMW

def SDADcv(X, Y, folds, Om, gam, lams, mu, q, PGsteps, PGtol, maxits, tol, feat, quiet, initTheta=None):
    n, p = X.shape
    _, K = Y.shape

    # Pad so that n divisible by folds
    pad = 0
    if n % folds != 0:
        pad = int(np.ceil(n / folds) * folds - n)
        X = np.vstack([X, X[:pad, :]])
        Y = np.vstack([Y, Y[:pad, :]]) 
    n = X.shape[0]

    # Random permutation
    perm = np.random.permutation(n)
    X = X[perm, :]
    Y = Y[perm, :]

    # Sort lambdas ascending
    lams = np.sort(lams)

    nv = n // folds  # Number validation samples per fold
    vinds = np.arange(nv)
    tinds = np.arange(nv, n)

    nlam = len(lams)
    scores = np.ones((folds, nlam)) * q * p
    mc = np.zeros((folds, nlam))

    for f in range(folds):
        Xt = X[tinds, :]
        Yt = Y[tinds, :]
        Xv = X[vinds, :]
        Yv = Y[vinds, :]

        # Normalize
        Xt_norm = normalize(Xt)
        Xt = Xt_norm['Xc']
        Xv = normalizetest(Xv, Xt_norm)

        nt, p = Xt.shape

        C = np.diag(1 / np.diag(Yt.T @ Yt)) @ (Yt.T @ Xt)

        # Check if Om is diagonal for Sherman-Morrison-Woodbury
        if norm(np.diag(np.diag(Om)) - Om, 'fro') < 1e-15:
            if Om.shape[0] != p:
                if not quiet:
                    print("Warning: Columns dropped in normalization to a total of p, setting Om to diag(p)")
                Om = np.eye(p)
            SMW = True

            M = mu + 2 * gam * np.diag(Om)
            Minv = 1 / M

            RS = cholesky(np.eye(nt) + 2 * Xt @ ((Minv / nt) * Xt.T))
        else:
            SMW = False
            A = mu * np.eye(p) + 2 * (Xt.T @ Xt / nt + gam * Om)
            R2 = cholesky(A)

        D = (1 / nt) * (Yt.T @ Yt)
        R = cholesky(D)

        if not quiet:
            print("-------------------------------------------")
            print(f"Fold number: {f+1}")
            print("-------------------------------------------")

        B = np.zeros((p, q, nlam))

        for ll in range(nlam):
            Q = np.ones((K, q))

            for j in range(q):
                Qj = Q[:, :j+1]

                def Mj(u):
                    return u - Qj @ (Qj.T @ (D @ u))

                theta = np.random.uniform(size=(K, 1))
                theta = Mj(theta)
                if j == 0 and initTheta is not None:
                    theta = initTheta
                theta /= np.sqrt((theta.T @ D @ theta)[0, 0])

                d = 2 * Xt.T @ (Yt @ (theta / nt))

                if SMW:
                    btmp = Xt @ (Minv * d) / nt
                    temp1 = solve(RS, btmp)
                    temp2 = solve(RS.T, temp1)
                    beta = (Minv * d) - 2 * Minv * (Xt.T @ temp2)
                else:
                    temp1 = solve(R2.T, d)
                    beta = solve(R2, temp1)

                for its in range(maxits):
                    b_old = beta.copy()

                    if SMW:
                        betaOb = ADMM_EN_SMW(Minv, Xt, RS, d, beta, lams[ll], mu, PGsteps, PGtol, True, np.ones(p))
                        beta = betaOb['y']
                    else:
                        betaOb = ADMM_EN2(R2, d, beta, lams[ll], mu, PGsteps, PGtol, True, np.ones(p))
                        beta = betaOb['y']

                    if norm(beta, 2) > 1e-12:
                        b = Yt.T @ (Xt @ beta)
                        y = solve_triangular(R.T, b, lower=True)
                        z = solve_triangular(R, y, lower=False)
                        tt = Mj(z)
                        t_old = theta.copy()
                        theta = tt / np.sqrt((tt.T @ D @ tt)[0, 0])

                        db = norm(beta - b_old, 2) / norm(beta, 2)
                        dt = norm(theta - t_old, 2) / norm(theta, 2)
                    else:
                        beta = np.zeros_like(beta)
                        theta = np.zeros_like(theta)
                        db = 0
                        dt = 0

                    if max(db, dt) < tol:
                        break

                if theta[0] < 0:
                    theta = -theta
                    beta = -beta

                Q[:, j] = theta.flatten()
                B[:, j, ll] = beta.flatten()

            PXtest = Xv @ B[:, :, ll]
            PC = C @ B[:, :, ll]

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

            if 1 <= sum_B_loc_nnz <= q * p * feat:
                scores[f, ll] = mc[f, ll]
            elif sum_B_loc_nnz > q * p * feat:
                scores[f, ll] = sum_B_loc_nnz

            if not quiet:
                print(f"f: {f+1} | ll: {ll+1} | lam: {lams[ll]} | feat: {sum_B_loc_nnz / (q * p):.4f} | mc: {mc[f,ll]:.4f} | score: {scores[f,ll]:.4f}")

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

    # Assuming SDAD is implemented in Python
    resBest = SDAD(Xt, Yt, Om, gam, lambest, mu, q, PGsteps, PGtol, maxits, tol)

    retOb = {
        'call': 'SDADcv',
        'B': resBest['B'],
        'Q': resBest['Q'],
        'lbest': lbest,
        'lambest': lambest,
        'scores': scores
    }

    return retOb
