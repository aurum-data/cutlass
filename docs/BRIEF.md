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

## Notes for Contributors
- If you modify the modeling behavior, start in linear_model.py (CV + logical polish) and _solvers.py (optimization).
- If you change how features are rectified or grouped, update preprocessing.py and review experiment drivers that
  assume specific feature orders.
- The experiment drivers mirror the papers and are the best integration tests; run them when making algorithmic changes.
