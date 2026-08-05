# Vignette 2: Logical Polishing and Rule Compression

The experiment driver toggles several “logical polish” options to compress the
rectified L1 model into a sparse m-of-k voting rule.  This vignette shows how to
trigger the same behaviour directly via the `CutlassClassifier`.

```python
import numpy as np
import pandas as pd
from cutlass import CutlassClassifier
from cutlass.metrics import calculate_youden_j

rng = np.random.default_rng(7)
n = 2000
timepoints = [f"T{i}" for i in range(6)]
X = pd.DataFrame({f"feat_{t}": rng.normal(size=n) for t in timepoints})

# Positive class concentrates on two “critical range” windows.
y = (
    (X["feat_T1"].between(0.5, 1.0))
    & (X["feat_T4"].between(-0.6, -0.1))
).astype(int)
df = X.assign(INDC=y)

clf = CutlassClassifier(
    rectify=True,
    Cs=25,
    solver="cd",
    backend="cpu",
    cv=5,
    logic_polish=True,
    logic_polish_stage="expanded",  # polish after restoring duplicate columns
    logic_scale=12.0,
    logic_target=0.95,          # adopt rule when Youden's J >= 0.95
    logic_k_policy="premin",    # mirror experiment driver option
    logic_intercept="mofk",     # use m-of-k voting threshold
    logic_m_frac=0.6,           # vote threshold at 60% of active features
    logic_plot=False,           # set True + install matplotlib for figure export
)
clf.fit(df.drop(columns=["INDC"]), df["INDC"])

prob = clf.predict_proba(df.drop(columns=["INDC"]))[:, 1]
pred = (prob >= 0.5).astype(int)
print("Adopted logical rule:", clf.logic_diag_["adopted"])
print("Top-k selected:", clf.logic_diag_["k_chosen"])
print("Intercept policy:", clf.logic_diag_["intercept_policy"])
print("Youden's J:", calculate_youden_j(df["INDC"], pred))

# Diagnostics captured during polishing mirror the experiment driver output.
diag = clf.logic_diag_
print("Baseline J0:", diag["J0"])
print("Adopt threshold:", diag["adopt_threshold"])
```

The `logic_*` parameters map one-to-one with the original script, and
`logic_diag_` captures the same dictionary returned by the driver (including the
J-versus-k curve).  Set `logic_plot=True` and `logic_plot_dir="./figs"` to
reproduce the diagnostic PNGs.

Logical polishing always executes on CPU, even if the preceding model fit uses
`solver="fista"` or `solver="hybrid"` with `backend="cuda"`. For fit-axis
polishing its duration is reported as `logical_polish_cpu` in `fit_timings_`.
The default `logic_polish_stage="expanded"` restores consolidated duplicate
columns before constructing the rule; use `"fit"` to polish only the reduced
fit axis.
