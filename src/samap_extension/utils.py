# src/labcore/scrnaseq/utils.py

import pandas as pd
import numpy as np
import scipy.sparse as sp
from anndata import AnnData 
import scanpy as sc

def describe_adata(adata, cell_index: int = 0, n_features: int = 100) -> dict:
    """Print a concise structural summary of an AnnData object.

    Displays the object's shape, the first few rows of cell and gene
    metadata, and the expression profile of a single cell of interest.

    Args:
        adata: An AnnData object to inspect.
        cell_index: Integer position of the cell to profile. Defaults to 0
            (the first cell).
        n_features: Number of gene-expression pairs to print for the
            selected cell. Defaults to 100.

    Returns:
        A dict mapping gene names to expression values for the selected
        cell, so the caller can use the profile programmatically if needed.

    Example:
        >>> import scanpy as sc
        >>> from labcore import scrnaseq
        >>> adata = sc.read_h5ad("path/to/object.h5ad")
        >>> profile = scrnaseq.describe_adata(adata, cell_index=0, n_features=50)
    """
    # --- Shape ---
    print(f"Shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes\n")

    # --- Metadata ---
    print("Cell metadata (obs):")
    print(adata.obs.head(), "\n")

    print("Gene metadata (var):")
    print(adata.var.head(), "\n")

    # --- Single-cell expression profile ---
    cell_id = adata.obs_names[cell_index]
    cell_features = adata[cell_index, :].X

    if not isinstance(cell_features, np.ndarray):
        cell_features = cell_features.toarray()

    cell_profile = dict(zip(adata.var_names, cell_features.flatten()))

    print(f"Cell ID: {cell_id}")
    print(f"First {n_features} features:")
    print(list(cell_profile.items())[:n_features])

    return cell_profile

def adata_status(adata, counts_layer="counts"):
    """Print a concise status report for an AnnData object."""
    def _x_summary(X):
        vals = X.data if sp.issparse(X) else np.asarray(X).ravel()
        if vals.size == 0: return {"dtype": str(vals.dtype), "nvals": 0}
        return {"dtype": str(vals.dtype), "min": float(vals.min()), "max": float(vals.max()), "mean": float(vals.mean())}

    print("=== AnnData Status Report ===")
    print(f"n_obs: {adata.n_obs:,}, n_vars: {adata.n_vars:,}")
    print(f"obs_names unique: {adata.obs_names.is_unique}, var_names unique: {adata.var_names.is_unique}")
    print(f"var columns: {list(adata.var.columns)}")
    print(f"X summary: {_x_summary(adata.X)}")
    print(f"layers: {list(adata.layers.keys())}")
    if counts_layer in adata.layers:
        print(f"  -> '{counts_layer}' summary: {_x_summary(adata.layers[counts_layer])}")
    print(f"obsm keys: {list(adata.obsm.keys())}")
    print(f"obsp keys: {list(adata.obsp.keys())}")
    if adata.raw: print(f"raw: present, with {adata.raw.n_vars:,} vars")

