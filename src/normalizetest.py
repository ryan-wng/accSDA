import numpy as np
from normalize import normalize

def normalizetest(Xtst, Xn):
    ntst, p = Xtst.shape
    
    # Subtract training mean from test data
    Xtst_centered = Xtst - Xn['mx']  # broadcast subtraction
    
    # Normalize only columns where norm != 0
    Xtst_normalized = Xtst_centered[:, Xn['Id']] / Xn['vx'][Xn['Id']]
    
    return Xtst_normalized

# Training data
# Xtr = np.random.choice(np.arange(1, 4), size=(3, 4), replace=True)

# # Test data
# Xtst = np.random.choice(np.arange(1, 4), size=(3, 4), replace=True)

# # Normalize training data
# Nm = normalize(Xtr)

# # Normalize test data
# Xtst_norm = normalizetest(Xtst, Nm)

# print("Training data:\n", Xtr)
# print("Test data before normalization:\n", Xtst)
# print("Normalized test data:\n", Xtst_norm)