# SC course 2026 Group 1 repository

## Project: Modeling CRISPR perturbations in melanoma cells under different experimental conditions

This project uses the CRISPR perturbation dataset from Frangieh et al., 2021. It contains
scRNA-seq data of ca. 218,000 melanoma cells split between three experimental conditions
and treated with a CRISPR library targeting 248 genes. Single-cell surface protein
measurements are also available.

The three experimental conditions (`perturbation_2`) are:
- **Control**: maintained in culture medium
- **IFNγ**: treated with interferon-γ
- **Co-culture**: co-cultured with tumor-infiltrating lymphocytes (TILs) for 48h

The genetic perturbation of a cell is given by `perturbation`, where `control` marks
unperturbed cells.

This project is done by a group of 2 students (Tim Auer, Levente Temesvári-Nagy).

## Tasks

### Task 1: Condition Classification
Can a classifier identify which treatment condition a cell came from using its gene
expression profile? Which genes drive the separation? For a group of two, two types of
models are trained, and feature importance is computed for each.

### Task 2: Clustering
Which genetic perturbations show similar effects, if any? Which clustering method(s) best
capture(s) the underlying biology? Different clustering methods are applied, visualized and
compared, and the results are interpreted biologically.

### Task 3: Perturbation Prediction
Can a model predict transcriptome changes for a gene that was not knocked out in the
training data? A subset of 50 perturbations is selected for modeling, with some held out
for testing only. The target variable is the mean log2 fold change per condition and
perturbation target. For a group of two, three types of models are trained, at least one of
which is deliberately simplistic.

## Task 1 approaches

Two model types are used for the condition classification task, both evaluated with the
same 5-fold stratified cross-validation over all three conditions.

### Tree ensemble (XGBoost)

Exploratory implementation: [notebooks/xgb_classifier_test.ipynb](notebooks/xgb_classifier_test.ipynb)
— QC filtering, normalization and log1p, 1000 highly variable genes plus paper marker
genes, PCA/neighbors/UMAP, XGBoost with 5-fold stratified CV, SHAP feature importance
shown on UMAPs.

Cross-validated version: [notebooks/task1_feature_importance_cv.ipynb](notebooks/task1_feature_importance_cv.ipynb).
Two capacities are compared (6 rounds and 400 rounds) to show how much of the accuracy is
reachable with a very small model. Gene selection is redone **inside each fold** so the HVG
step cannot see the held-out cells; the size of that leakage is quantified separately by
`tools/hvg_inside_fold.py` → `data/task1_hvg_leakage_fold0.json`.

### Neural network (MLP)

[notebooks/mlp_classifier_task1.ipynb](notebooks/mlp_classifier_task1.ipynb) — PyTorch MLP
on the same fold structure and the same feature space as the tree ensembles, so the three
models are directly comparable. Feature importance is permutation-based and computed per
fold, which gives an across-fold error bar on every gene rather than a single ranking.

### Task 1 outputs

- [task1_observations.md](task1_observations.md) — measurements and open questions
- `data/task1_accuracy_table.csv`, `data/task1_cv_results.json` — accuracy per model and fold
- `data/task1_gene_importance.csv`, `data/task1_fold_importance_long.csv` — importance,
  aggregated and per fold
- `data/task1_top20_annotated.csv` — top genes annotated against Reactome and the paper
- `figures/task1_fig*.png` — importance with error bars, cross-model concordance,
  confusion matrices, accuracy vs baselines

## Task 2 approach

[notebooks/task2_perturbation_clustering.ipynb](notebooks/task2_perturbation_clustering.ipynb)
clusters perturbations in all three conditions. Every QC-passing cell is first
assigned by its raw `obs["perturbation"]` value, then expression is averaged within each
condition and perturbation. Seven methods are compared on those aggregate signatures:
Ward/Euclidean, average- and complete-linkage on correlation distance, k-means, Leiden,
DBSCAN and HDBSCAN. DBSCAN's `eps` is chosen from a k-distance plot rather than by hand.

Methods are scored against the pathway modules discussed in the paper (ARI/AMI, module
recovery) and by bootstrap stability, so the choice of a recommended method is made on
stated criteria rather than by eye.

Outputs: [task2_observations.md](task2_observations.md), `data/clustering_comparison.csv`,
`data/module_recovery.csv`, `data/perturbation_clusters.parquet`,
`data/task2_perturbation_aggregation.csv`, `data/task2_aggregation_audit.csv`,
`figures/task2_*.png`.

## Task 3 approach

[notebooks/task3_perturbation_prediction.ipynb](notebooks/task3_perturbation_prediction.ipynb).
Target: mean log2 fold change per condition and perturbation target.

**Selection.** 50 perturbations from the 210 with at least 50 single-perturbation cells in
every condition, ranked by the number of genes with |log2FC| > 3×SE. 13 are mandatory so
that a whole-module holdout is possible (the IFN-γ/JAK-STAT core and the eligible MHC-I
members); the rest are the strongest remaining signals.

**Three holdouts**, none used in training or hyperparameter selection: 10 random
perturbations, the 5-member IFN-γ/JAK-STAT module, and the 8-member MHC-I module. Holding
out whole modules asks the harder question — can the model predict a gene when nothing
functionally similar was in training?

**Three model types** (feature engineering + learning algorithm):

1. multi-task neural network on engineered target-gene features, predicting the RNA and
   surface-protein responses jointly
2. training-mean floor — predict the mean training response, ignoring the target's identity
3. zero floor — predict no change at all (the deliberately simplistic model)

Uncertainty is bootstrap CIs over held-out perturbation rows, including the **paired**
network-minus-floor difference, which is the comparison that decides whether the network
is doing anything.

Outputs: [task3_observations.md](task3_observations.md), `data/task3_metrics.csv`,
`data/task3_per_perturbation.csv`, `data/task3_splits.json`, `figures/task3_*.png`.

## Repository structure

```
.
├── data/                          # AnnData inputs (untracked) + derived tables
│   ├── FOUNDATION_VERIFICATION.md # correctness checks on the pseudobulk foundation
│   └── CACHE_PROVENANCE.json      # what produced each cached table, and when
├── src/
│   ├── preprocessing.py           # preprocess()
│   ├── prepare_data.py             # builds the shared cache and Parquet inputs
│   ├── models.py                  # fit_xgb()
│   ├── pseudobulk.py              # streamed per-group statistics, log2FC, HVG
│   ├── task1_cv.py                # cross-validation driver for task 1
│   ├── task1_figures.py           # task 1 figure builders
│   ├── task3_features.py          # target-gene feature engineering
│   ├── task3_model.py             # multi-task network
│   ├── task3_run.py               # training / evaluation driver
│   └── figstyle.py                # shared figure style
├── tools/
│   ├── task1_figures.py           # renders figures/task1_fig*.png
│   └── hvg_inside_fold.py         # quantifies HVG selection leakage
├── notebooks/
│   ├── raw_data_overview.ipynb            # exploratory
│   ├── protein_data_overview.ipynb        # exploratory
│   ├── xgb_classifier_test.ipynb          # task 1, exploratory
│   ├── perturbation_clustering.ipynb      # task 2, exploratory
│   ├── task1_feature_importance_cv.ipynb  # task 1, tree ensembles
│   ├── mlp_classifier_task1.ipynb         # task 1, neural network
│   ├── task2_perturbation_clustering.ipynb
│   └── task3_perturbation_prediction.ipynb
├── figures/
├── task1_observations.md
├── task2_observations.md
├── task3_observations.md
├── environment.yml
└── README.md
```

The four `task*` notebooks are the ones that produce the reported results. The others are
exploratory and are kept for the record.

## Results

[RESULTS.md](RESULTS.md) collects the measurements from all three tasks in one place. It is
**generated** by `python tools/build_report.py`, which reads every number directly from the
tables in `data/` — so the report cannot drift from the results. Re-run it after re-running
any notebook.

## A note on the observations files

`task1_observations.md`, `task2_observations.md` and `task3_observations.md` record
**measurements and open questions only** — every number in them is traceable to a table in
`data/`. The biological interpretation is written by us in the report and the slides, in
line with the course policy on AI assistance.

## Data

The raw dataset is not tracked in git. Place these two files exactly here:

```
data/frangieh/rna.h5ad
data/frangieh/protein.h5ad
```

Build every shared cache and pseudo-bulk table with one command from the
repository root:

```
python src/prepare_data.py
```

This creates `data/dataset_hv.h5ad`, used by Tasks 1 and 2, plus the `rna_*.parquet`
and `adt_*.parquet` tables used by Task 3. The script streams the
gene-major RNA matrix, so it does not load the full 5.5 GB matrix into memory.
The two small `data/pathway_labels*` reference files are tracked with the code;
they do not need to be downloaded or regenerated.

### Clean run order

After activating the environment and running `prepare_data.py`, run the four
canonical notebooks in this order:

1. `notebooks/mlp_classifier_task1.ipynb`
2. `notebooks/task1_feature_importance_cv.ipynb`
3. `notebooks/task2_perturbation_clustering.ipynb`
4. `notebooks/task3_perturbation_prediction.ipynb`

The notebooks resolve the repository root themselves, so they work when
Jupyter is started from either the repository root or `notebooks/`. Task 2
creates `data/task2_bootstrap_labels.pkl` on its first clean run and reuses it
on later runs.

## Setup

Create the conda environment and activate it:

```
conda env create -f environment.yml
conda activate sc-course-2026
```

## Course description

This course will teach approaches to machine learning, applied to single-cell multiomics
data analysis tasks such as batch integration, clustering, cell type annotation,
differential gene expression, and analysis of gene regulatory mechanisms. Machine learning
methods covered will include both classical approaches (linear models, tree ensembles) as
well as deep learning with PyTorch. The focus will be on practical aspects critical to
rigorous training and evaluation of models.
