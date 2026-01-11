import numpy as np

def normalize(X):
    n, p = X.shape
    mx = np.mean(X, axis=0)           # mean of each column (shape: (p,))
    X_centered = X - mx               # broadcasting subtract mean from each row

    vx = np.sqrt(np.sum(X_centered ** 2, axis=0))  # L2 norm of each column (shape: (p,))
    Id = vx != 0                     # boolean mask for nonzero norm columns

    # Normalize only columns with nonzero norm
    X_normalized = X_centered[:, Id] / vx[Id]

    return {
        'Xc': X_normalized,
        'mx': mx,
        'vx': vx,
        'Id': Id
    }

# --- Test the normalize function ---
# Data
# X = np.random.choice([1, 2, 3], size=(3, 4), replace=True)

# # Normalize data
# Nm = normalize(X)

# # Print normalized matrix
# print("Normalized Xc:\n", Nm["Xc"])

# # Check if any variables were removed (columns with zero variance)
# removed_vars = np.where(~Nm["Id"])[0]
# print("Removed variable indices (0-based):", removed_vars)