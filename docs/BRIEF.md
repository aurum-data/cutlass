# CUTLASS Brief

## Purpose
CUTLASS (Critical-range rectified LASSO) is a lightweight Python package that implements the research workflow for
rectified L1-penalized logistic regression and optional rule compression. It targets interpretable, sparse binary
classifiers by (1) rectifying numeric features into {-1, +1} indicators using class-conditional critical ranges,
(2) fitting an L1 logistic model with cross-validation, and (3) optionally "polishing" the model into a fixed-magnitude
logical rule optimized for Youden's J.

The library is intentionally minimal (NumPy + pandas, optional matplotlib for plots) and mirrors the research scripts
so experiment drivers can be migrated with minimal changes.

## Core Workflow (High Level)
- Rectify: infer per-feature critical ranges from the positive class, then binarize features into {-1, +1}.
- Scale: optionally standardize features when not already binary.
- Fit: cross-validate an L1 logistic model across a C grid with warm starts and parallelized folds.
- Polish (optional): compress to a top-k rule with fixed magnitude K and intercept policies, adopting if Youden's J
  stays within a user-defined tolerance.

## Package Structure (src/cutlass)
- __init__.py: package exports (CutlassClassifier, CutlassLogisticCV, Rectifier, StandardScaler, metrics).
- model.py: CutlassClassifier end-to-end estimator; handles rectification, scaling, fitting, and predictions.
- linear_model.py: CutlassLogisticCV (L1 logistic CV, parallel fold evaluation, optional logical polish).
- _solvers.py: low-level solvers (_CDLogistic coordinate descent, _FISTALogistic proximal gradient).
- preprocessing.py: Rectifier and a minimal StandardScaler; grouping heuristic by feature name prefix.
- metrics.py: Youden's J, ROC AUC, and precision-recall curve implementations.
- serialization.py: persistence helpers (rectifier limits JSON and fitted model NPZ).
- pipeline.py: minimal Pipeline used by experiment scripts (fit + predict_proba).
- _math.py: numerical helpers (sigmoid, softplus, log loss, soft-threshold).

## Notable Top-Level Scripts and Data
- experiment_driver_v5.py: main experiment driver for synthetic datasets and logical polish evaluation.
- experiment_driver_csv.py: variant that runs experiments on CSV datasets.
- case_1_simple_script_scikit_fast_v6.py / case_1_simple_script_gd_v9.py: research scripts used by the drivers.
- sensor_generate - commented.py: synthetic data generator referenced by the experiment driver.
- examples/quickstart.py: minimal usage example for the library.
- docs/vignettes/: step-by-step guides (rectified workflow, logical polish, batch experiments).
- scripts/prepare_leukemia_csv.py: data preparation helper for leukemia CSVs.
- sample_data/: example leukemia datasets.
- runs_*/ runs_csv*/ runs_new*/ runs_leuk/: experiment outputs (generated artifacts).

## Reference Papers (papers/)
- 20250829_IEEE_Big_Data_Efficient_Longitudinal_Feature_Selection_via_Binarized_Transformation- Theory_and_Case_Studies_V5.pdf
- 27_Paper_for_BDAA_2025.pdf
- LASSO_Logic_Engine_20220819_IEEE-Big_Data.pdf

## Architecture & Design Principles
- **Standalone Implementation**: CUTLASS avoids a scikit-learn dependency to maintain tight control over its specialized data transformations (critical range rectification) and optimization path (L1 coordinate descent). It closely mimics the `fit(X, y)` and `predict_proba(X)` API to stay intuitive.
- **Performance**: Numeric helpers and solvers (in `_math.py` and `_solvers.py`) use heavily optimized NumPy operations.
- **Dependencies**: Standard PEP 621 packaging (`pyproject.toml`). Base dependencies are strictly `numpy` and `pandas`, with `matplotlib` available as an optional `[plots]` extra.

## Agent / Developer Modification Guide
This section provides a direct mapping of developer intentions to files and concepts, allowing an AI agent or human to modify the codebase exactly where necessary:

- **Modifying Optimization / Solvers**: To tweak coordinate descent or FISTA optimization steps, edit `src/cutlass/_solvers.py`.
- **Modifying Logical Rules & Compression**: To alter how models are "polished" into logical rules or rounded to top-k elements, modify `src/cutlass/linear_model.py` (`CutlassLogisticCV`).
- **Modifying Feature Binarization (Rectification)**: Changes to how critical ranges are computed from the positive class or how continuous variables are binarized must safely go in `src/cutlass/preprocessing.py` (`Rectifier`).
- **Modifying Overall Pipeline / Wrappers**: To change arguments exposed to users or how the rectifier and model are chained together, look at `src/cutlass/model.py` (`CutlassClassifier`).
- **Modifying Metrics / Validation**: New evaluators or changes to Youden's J, AUC, or other metrics should be done in `src/cutlass/metrics.py`.
- **Validating Changes**: The experiment drivers (`experiment_driver_v5.py`, `experiment_driver_csv.py`) mirror the reference papers and serve as robust integration tests. Ensure they run successfully when making algorithmic changes.
- **Build / Packaging**: The library uses standard tools (`python -m build`). Update `pyproject.toml` if modifying dependencies or metadata.
