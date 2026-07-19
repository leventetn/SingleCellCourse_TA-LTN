import scanpy as sc


def preprocess(adata, min_genes, min_cells, total_cutoff, mt_cutoff):
    # mitochondrial genes, 'MT-' for human, 'Mt-' for mouse
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    # ribosomal genes
    adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
    # hemoglobin genes
    adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]')

    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo', 'hb'], inplace=True, log1p=True)

    sc.pl.violin(
        adata,
        ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
        jitter=0.4,
        multi_panel=True,
    )
    
    # filter data using parameters
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata = adata[adata.obs.pct_counts_mt < mt_cutoff]
    adata = adata[adata.obs.total_counts < total_cutoff]
    adata = adata.copy()

    sc.pl.violin(
        adata,
        ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
        jitter=0.4,
        multi_panel=True,
    )

    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', color='pct_counts_mt')
    
    return adata

