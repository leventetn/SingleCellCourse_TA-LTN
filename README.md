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

[notebooks/task1_feature_importance_cv.ipynb](notebooks/task1_feature_importance_cv.ipynb)
trains XGBoost at two capacities (6 rounds and 400 rounds) to show how much of the accuracy is
reachable with a very small model. The notebook uses the cached 1,011-gene feature set for
every fold; because those genes were selected on all cells, the held-out fold is visible to
the HVG step. That leakage is quantified separately by `tools/hvg_inside_fold.py`, which
redoes the selection **inside fold 0** and compares the accuracy →
`data/task1_hvg_leakage_fold0.json` (the optimism is negative, so the cached set is safe to
use for the reported numbers).

### Neural network (MLP)

The same notebook, [notebooks/task1_feature_importance_cv.ipynb](notebooks/task1_feature_importance_cv.ipynb),
also trains a PyTorch MLP (1011 → 128 → 64 → 3, dropout 0.2, Adam, 8 epochs) on the same
folds and the same feature space as the tree ensembles, so all three models are directly
comparable. Feature importance is mean |gradient × input| on each fold's held-out cells,
computed per fold, which gives an across-fold error bar on every gene rather than a single
ranking.

### Task 1 outputs

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

Outputs: `data/clustering_comparison.csv`,
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

Outputs: `data/task3_metrics.csv`,
`data/task3_per_perturbation.csv`, `data/task3_splits.json`, `figures/task3_*.png`.

## Repository structure

```
.
├── data/                          # AnnData inputs (untracked) + derived tables
├── src/
│   ├── prepare_data.py            # builds the shared cache and Parquet inputs
│   ├── pseudobulk.py              # streamed per-group statistics, log2FC, HVG
│   ├── task1_cv.py                # cross-validation driver for task 1
│   ├── task3_features.py          # target-gene feature engineering
│   ├── task3_model.py             # multi-task network
│   ├── task3_run.py               # training / evaluation driver
│   └── figstyle.py                # shared figure style
├── tools/
│   ├── task1_figures.py           # renders figures/task1_fig*.png
│   └── hvg_inside_fold.py         # quantifies HVG selection leakage
├── notebooks/
│   ├── task1_feature_importance_cv.ipynb  # task 1, tree ensembles + MLP
│   ├── task2_perturbation_clustering.ipynb
│   └── task3_perturbation_prediction.ipynb
├── figures/
├── environment.yml
└── README.md
```

These three notebooks produce all the reported results.

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

## Setup

Create the conda environment and activate it:

```
conda env create -f environment.yml
conda activate sc-course-2026
```

### Clean run order

After activating the environment and running `prepare_data.py`, run the three
canonical notebooks in this order:

1. `notebooks/task1_feature_importance_cv.ipynb`
2. `notebooks/task2_perturbation_clustering.ipynb`
3. `notebooks/task3_perturbation_prediction.ipynb`

The notebooks resolve the repository root themselves, so they work when
Jupyter is started from either the repository root or `notebooks/`. Task 2
creates `data/task2_bootstrap_labels.pkl` on its first clean run and reuses it
on later runs.

## Course description

This course will teach approaches to machine learning, applied to single-cell multiomics
data analysis tasks such as batch integration, clustering, cell type annotation,
differential gene expression, and analysis of gene regulatory mechanisms. Machine learning
methods covered will include both classical approaches (linear models, tree ensembles) as
well as deep learning with PyTorch. The focus will be on practical aspects critical to
rigorous training and evaluation of models.
