# Breast Cancer Diagnosis: End-to-End ML Pipeline (pandas, SQL, scikit-learn, TensorFlow/Keras)

A single runnable notebook that takes the UCI Breast Cancer Wisconsin (Diagnostic) data from raw CSV to a validated, explainable model that can score new batches safely. The goal is to flag likely malignant tumors for priority pathologist review. The model prioritizes cases; it does not make the diagnosis.

**Notebook:** [`breast_cancer_ml_pipeline.ipynb`](breast_cancer_ml_pipeline.ipynb)

## Results (held-out test set, 114 tumors)

| Model | Threshold | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic regression | 0.500 | 0.976 | 0.953 | 0.965 | 0.995 |
| Random forest | 0.337 | 1.000 | 0.933 | 0.966 | 0.998 |
| **Keras MLP + Dropout (selected)** | 0.500 | **0.976** | **0.976** | **0.976** | **0.995** |

- Selected model caught 41 of 42 malignant cases, with 1 benign tumor sent for extra review.
- 95% bootstrap CIs: recall 0.921–1.000, precision 0.919–1.000, ROC-AUC 0.983–1.000.
- The model and threshold were chosen on validation by a rule fixed before modeling: catch at least 97% of malignant tumors, then take the highest precision, with ties going to the simpler model. The test set was evaluated once.
- Brier score 0.017, against 0.233 for always guessing the base rate.

## What the pipeline covers

| # | Section | Highlights |
|---|---|---|
| 0 | Requirements & methodology | Business question, error costs, success criteria and selection rule set up front |
| 1–3 | Setup, load, first-look EDA | `read_csv` vs `np.genfromtxt`, `info` / `describe` / `value_counts` |
| 4 | Missing values | Find nulls, compare median / mean / KNN imputation against hidden true values |
| 5 | Quality checks & manipulation | Label encoding, duplicate/range checks, multicollinearity |
| 6 | EDA visuals | Class balance, distributions, box/violin plots, correlation heatmap, pairplot |
| 7 | SQL in the notebook (`sqlite3`) | `ROW_NUMBER`, `RANK`, `DENSE_RANK`, de-duplication, each paired with its pandas equivalent |
| 8–9 | Stratified 60/20/20 split, leak-free preprocessing | `Pipeline` (impute → scale) fit on train only |
| 10 | Baselines | Logistic regression, decision tree, random forest with leak-free stratified CV |
| 11 | Keras network | Overfitting model vs Dropout model, early stopping |
| 12 | Validation | Per-model threshold tuning on validation, one-time test evaluation, bootstrap CIs |
| 13 | Calibration | Reliability curves, Brier score |
| 14 | Drivers | Grouped permutation importance (handles correlated size features) |
| 15 | Batch scoring | `score_batch` with schema, type, null and out-of-range checks that fail loudly |
| 16 | Stakeholder summary | Generated from the computed numbers, so it stays correct on rerun |
| 17–19 | Recap, self-checking practice problems, Q&A | |
| A | Same network in PyTorch | Run in a separate process to avoid TF/PyTorch library conflicts |
| B | Practice solutions | Re-runs every checker |

## Run it

**Google Colab:** upload the notebook and run all cells. Colab already has the dependencies.

**Local:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab breast_cancer_ml_pipeline.ipynb
```

The notebook writes `breast_cancer_data.csv` on first run. It downloads the file from UCI, and if that fails it falls back to scikit-learn's bundled copy of the same dataset, so no manual download is needed. PyTorch is only used in Appendix A.

Tested with Python 3.13, pandas 3.0.5, NumPy 2.5.3, TensorFlow 2.21.0.

## Data

Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The dataset has 569 tumors and 30 numeric features computed from digitized fine-needle aspirate (FNA) images. The label is M = malignant, B = benign. It is public, de-identified research data with no PHI. All IDs in the practice problems are synthetic.
