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

Two model types are used for the condition classification task.

### Tree ensemble (XGBoost)

The current implementation is in [notebooks/xgb_classifier_test.ipynb](notebooks/xgb_classifier_test.ipynb):

- QC filtering (minimum genes per cell, minimum cells per gene, mitochondrial percentage,
  total counts), followed by normalization and log1p transformation.
- Feature selection: 1000 highly variable genes together with a list of differentially
  expressed marker genes from the paper.
- PCA, neighbors and UMAP.
- Label encoding of the condition.
- XGBoost classifier evaluated with 5-fold stratified cross-validation using balanced
  accuracy.
- Feature importance computed with SHAP; the most important features are shown on UMAPs.

### Neural network — planned

<!-- TODO: fill in the details of the neural network approach -->
- Architecture: _placeholder_
- Input features: _placeholder_
- Training setup: _placeholder_
- Evaluation: _placeholder_
- Feature importance: _placeholder_

## Repository structure

```
.
├── data/            
├── src/             
│   ├── preprocessing.py   # preprocess()
│   └── models.py          # fit_xgb()
├── notebooks/       
│   └── xgb_classifier_test.ipynb
├── environment.yml
└── README.md
```

## Data

The dataset is not tracked in git. Place the AnnData files in the `data/`
directory. The notebooks read from `../data/frangieh`.

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
