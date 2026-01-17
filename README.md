# RE-sLDA: Resampling-Enhanced Sparse LDA for Ordinal Outcomes
This repository contains the implementation of RE-sLDA, a framework designed to enhance feature selection stability and accuracy when dealing with high-dimensional data and ordinal outcomes. By integrating resampling techniques with Sparse Linear Discriminant Analysis (sLDA), this method identifies robust biomarker signatures that standard sparse models often miss due to selection instability.

## Key Features
Ordinal Outcome Optimization: Specifically tuned for categorical outcomes with a natural ordering (e.g., disease severity, treatment response grades).

Resampling-Based Stability: Utilizes bootstrap-based resampling to calculate Variable Inclusion Probabilities (VIP), ensuring the selected features are not artifacts of a single data split.

Parallel Computing Support: Fully integrated with doParallel and foreach for high-performance execution on multi-core clusters.


## Testing

- Individual solvers validated against R outputs
- Shape and sparsity consistency confirmed
- Floating-point differences expected at ~1e−6 level
- Full large-scale benchmarks ongoing
---

## Basic Usage

Clone the repository and install dependencies.
#### To use **Binary or Multiclass SDA**:
```bash
from ASDA import ASDA
import numpy as np

X = np.random.randn(100, 500)     # n x p data
y = np.random.choice([0, 1, 2], size=100)

res = ASDA(
    Xt=X,
    Yt=y,
    gam=1e-3,
    lam=1e-4,
    method="SDAAP"
)

B = res["beta"]       # sparse coefficients
lda = res["fit"]      # trained LDA model
```
#### To use **Ordinal SDA**:
```bash
from ordASDA import ordASDA

res = ordASDA(
    Xt=X,
    Yt=y,      # ordinal labels
    s=1,
    gam=0,
    lam=0.05
)

selected_features = res["n_selected"]
```
#### To use **Cross-Validation for λ**:
```bash
res = ASDA(
    Xt=X,
    Yt=y,
    gam=1e-3,
    lam=np.logspace(-5, -1, 20),
    method="SDAAP",
    control={"CV": True, "folds": 5}
)

best_lambda = res["lambda"]
```
## Reference

If you use this code in academic work, please cite the original accSDA paper/package:
> Clemmensen, L., Hastie, T., Witten, D., & Ersbøll, B. (2011).  
> Sparse Discriminant Analysis. Technometrics.

---

## Disclaimer

This is a research-grade implementation, not a drop-in replacement for scikit-learn classifiers.  
It is intended for:
- Method development
- Reproducibility
- High-dimensional statistical learning research
