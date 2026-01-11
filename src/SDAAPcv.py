import numpy as np
from scipy.linalg import cholesky, solve_triangular
import normalize, normalizetest, SDAAP, APG_EN2, APG_EN2bt

def SDAAPcv_default(X, Y, folds, Om, gam, lams, q, PGsteps, PGtol, maxits, tol, feat, quiet, initTheta=None, bt=False, L=None, eta=None):
    n, p = X.shape
    _, K = Y.shape

    # Pad so n divisible by folds
    pad = 0
    if n % folds > 0:
        pad = int(np.ceil(n / folds) * folds - n)
        X = np.vstack([X, X[:pad]])
        Y = np.vstack([Y, Y[:pad]])
    n = X.shape[0]

    # Random permutation
    perm = np.random.permutation(n)
    X = X[perm, :]
    Y = Y[perm, :]

    lams = np.sort(lams)  # ascending order

    nv = n // folds
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

        # Normalize training data
        Xt_norm = normalize(Xt)  # Implement this, returns dict with 'Xc'
        Xt = Xt_norm['Xc']
        Xv = normalizetest(Xv, Xt_norm)  # Implement this

        nt = Xt.shape[0]
        p = Xt.shape[1]

        C = np.diag(1 / np.diag(Yt.T @ Yt)) @ (Yt.T @ Xt)  # centroid matrix

        A = {}
        if np.linalg.norm(np.diag(np.diag(Om)) - Om, ord='fro') < 1e-15:
            if Om.shape[0] != p:
                print("Warning: Columns dropped in normalization to total p, setting Om to diag(p)")
                Om = np.eye(p)
            A['flag'] = 1
            A['gom'] = gam * np.diag(Om)
            A['X'] = Xt
            A['n'] = nt
            A['A'] = 2 * (Xt.T @ Xt / nt + gam * Om)
            alpha = 1 / (2 * (np.linalg.norm(Xt, ord=1) * np.linalg.norm(Xt, ord=np.inf) / nt + np.linalg.norm(np.diag(A['gom']), ord=np.inf)))
        else:
            A['flag'] = 0
            A['A'] = 2 * (Xt.T @ Xt / nt + gam * Om)
            alpha = 1 / np.linalg.norm(A['A'], ord='fro')

        L_val = 1 / alpha
        L_val = np.linalg.norm(np.diag(np.diag(Om * gam)), ord=np.inf) + (np.linalg.norm(Xt, ord='fro') ** 2)
        origL = L_val
        D = (1 / n) * (Yt.T @ Yt)
        R = cholesky(D, lower=False)

        if not quiet:
            print("-------------------------------------------")
            print(f"Fold number: {f+1}")
            print("-------------------------------------------")

        B = np.zeros((p, q, nlam))

        for ll in range(nlam):
            Q = np.ones((K, q))

            def Mj(u):
                return u - Q[:, :ll+1] @ (Q[:, :ll+1].T @ (D @ u))

            for j in range(q):
                L_val = origL

                Qj = Q[:, :j+1]

                def Mj(u):
                    return u - Qj @ (Qj.T @ (D @ u))

                theta = np.random.rand(K, 1)
                theta = Mj(theta)
                if j == 0 and initTheta is not None:
                    theta = initTheta
                theta = theta / np.sqrt(float(theta.T @ D @ theta))

                if ll == 0:
                    if np.linalg.norm(np.diag(np.diag(Om)) - Om, ord='fro') < 1e-15:
                        ominv = 1 / np.diag(Om)
                        rhs0 = Xt.T @ (Yt @ (theta / nt))
                        rhs = Xt @ ((ominv / nt) * rhs0)
                        tmp_partial = np.linalg.solve(np.eye(nt) + Xt @ ((ominv / (gam * nt)) * Xt.T), rhs)
                        beta = (ominv / gam) * rhs0 - (1 / gam ** 2) * ominv * (Xt.T @ tmp_partial)
                    else:
                        beta = np.zeros((p, 1))
                else:
                    beta = B[:, j, ll - 1].reshape(p, 1)

                for its in range(maxits):
                    d = 2 * Xt.T @ (Yt @ (theta / nt))
                    b_old = beta.copy()

                    if not bt:
                        betaOb = APG_EN2(A, d, beta, lams[ll], alpha, PGsteps, PGtol)
                        beta = betaOb['x']
                    else:
                        betaOb = APG_EN2bt(A, Xt, Om, gam, d, beta, lams[ll], L_val, eta, PGsteps, PGtol)
                        # L_val = betaOb['L']
                        beta = betaOb['x']

                    if np.linalg.norm(beta, ord=2) > 1e-12:
                        b = Yt.T @ (Xt @ beta)
                        y = solve_triangular(R.T, b, lower=True)
                        z = solve_triangular(R, y, lower=False)
                        tt = Mj(z)
                        t_old = theta.copy()
                        theta = tt / np.sqrt(float(tt.T @ D @ tt))

                        db = np.linalg.norm(beta - b_old) / np.linalg.norm(beta, ord=2)
                        dt = np.linalg.norm(theta - t_old) / np.linalg.norm(theta, ord=2)
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

                Q[:, j] = theta.ravel()
                B[:, j, ll] = beta.ravel()

            PXtest = Xv @ B[:, :, ll]
            PC = C @ B[:, :, ll]

            dists = np.zeros((nv, K))
            for i in range(nv):
                for j_ in range(K):
                    dists[i, j_] = np.linalg.norm(PXtest[i, :] - PC[j_, :])

            predicted_labels = np.argmin(dists, axis=1)

            Ypred = np.zeros((nv, K))
            for i in range(nv):
                Ypred[i, predicted_labels[i]] = 1

            mc[f, ll] = (0.5 * np.linalg.norm(Yv - Ypred, ord='fro') ** 2) / nv

            B_loc = B[:, :, ll]
            sum_B_loc_nnz = np.sum(B_loc != 0)

            if 1 <= sum_B_loc_nnz <= q * p * feat:
                scores[f, ll] = mc[f, ll]
            elif sum_B_loc_nnz > q * p * feat:
                scores[f, ll] = sum_B_loc_nnz

            if not quiet:
                print(f"f: {f+1} | ll: {ll+1} | lam: {lams[ll]} | feat: {sum_B_loc_nnz / (q * p)} | mc: {mc[f, ll]} | score: {scores[f, ll]}")

        tmp = tinds[:nv]

        if nv + 1 > nt:
            tinds = vinds
            vinds = tmp
        else:
            tinds = np.concatenate((tinds[nv:], vinds))
            vinds = tmp

    avg_score = np.mean(scores, axis=0)
    lbest = np.argmin(avg_score)
    lambest = lams[lbest]

    print(f"Finished Training: lam = {lambest}")

    Xt = X[:(n - pad), :]
    Yt = Y[:(n - pad), :]

    Xt_norm = normalize(Xt)
    Xt = Xt_norm['Xc']
    if Om.shape[0] != Xt.shape[1]:
        print("Warning: Columns dropped in normalization to total p, setting Om to diag(p)")
        Om = np.eye(Xt.shape[1])

    resBest = SDAAP(Xt, Yt, Om, gam, lams[lbest], q, PGsteps, PGtol, maxits, tol, bt=bt)

    return {
        'B': resBest['B'],
        'Q': resBest['Q'],
        'lbest': lbest,
        'lambest': lambest,
        'scores': scores
    }
