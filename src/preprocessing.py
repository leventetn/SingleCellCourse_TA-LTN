import os
import tempfile
from pathlib import Path

cache_dir = Path(tempfile.gettempdir()) / "sc-course-2026-cache"
(cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
(cache_dir / "numba").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import scanpy as sc


def preprocess(adata, min_genes, min_cells, total_cutoff, mt_cutoff, plot=True):
    # The Frangieh AnnData already contains these QC columns. Reusing them avoids
    # a slow full-matrix QC pass over the 5.5 GB RNA file.
    existing_qc = {'ngenes', 'ncounts', 'percent_mito'}.issubset(adata.obs.columns) and 'ncells' in adata.var.columns

    if existing_qc:
        adata.obs['n_genes_by_counts'] = adata.obs['ngenes']
        adata.obs['total_counts'] = adata.obs['ncounts']
        adata.obs['pct_counts_mt'] = adata.obs['percent_mito']
        adata.var['n_cells_by_counts'] = adata.var['ncells']
    else:
        # mitochondrial genes, 'MT-' for human, 'Mt-' for mouse
        adata.var['mt'] = adata.var_names.str.startswith(('MT-', 'Mt-'))
        # ribosomal genes
        adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
        # hemoglobin genes, excluding pseudogenes such as HBP*
        adata.var['hb'] = adata.var_names.str.contains(r'^HB(?!P)')
        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=['mt', 'ribo', 'hb'],
            percent_top=None,
            inplace=True,
            log1p=True,
        )

    if plot:
        sc.pl.violin(
            adata,
            ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
            jitter=0.4,
            multi_panel=True,
        )

    # Build masks from QC metrics and copy only once.
    cell_mask = (
        (adata.obs['n_genes_by_counts'] >= min_genes)
        & (adata.obs['pct_counts_mt'] < mt_cutoff)
        & (adata.obs['total_counts'] < total_cutoff)
    )
    gene_mask = adata.var['n_cells_by_counts'] >= min_cells
    adata = adata[cell_mask, gene_mask].copy()

    if plot:
        sc.pl.violin(
            adata,
            ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
            jitter=0.4,
            multi_panel=True,
        )

        sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', color='pct_counts_mt')

    return adata
