# samap-extension

SAMap cross-species analysis and plotting tools for **CDBL**.

This package provides plotting and analysis utilities that wrap
[SAMap](https://github.com/atarashansky/SAMap) outputs. It is a
companion to [`lab-core-python`](https://github.com/[your-github-username]/lab-core-python)
and is kept as a separate package because SAMap requires a tightly
pinned Python 3.9 environment that is incompatible with `labcore`'s
general-purpose dependency stack.

## Requirements

This package has **no declared pip dependencies** and must be installed
into a pre-configured SAMap conda environment. See
`environment_samap.yml` for the full environment specification.

**Both** this package and `lab-core-python` must be installed into the
SAMap environment using `--no-deps`:

```bash
# 1. Create and activate the SAMap environment
conda env create -f environment_samap.yml
conda activate samap

# 2. Install lab-core-python (no deps — scanpy etc. already in env)
pip install --no-deps git+https://github.com/[your-github-username]/lab-core-python.git
# or in editable mode:
pip install --no-deps -e /path/to/lab-core-python

# 3. Install this package
pip install --no-deps git+https://github.com/[your-github-username]/samap-extension.git
# or in editable mode:
pip install --no-deps -e /path/to/samap-extension
```

## Package structure

```
labcore_integration/
├── __init__.py
└── plotting.py     # Sankey, square scatter, joint UMAP, expression overlap
```

## Quick start

```python
from samap_extension import plotting as splt

# Joint UMAP coloured by species
fig, ax = splt.square_scatter(sm, COLORS={"ml": "#377EB8", "sm": "#E41A1C"})

# Joint UMAP highlighting one species' cell-type labels
fig, ax = splt.plot_joint_umap(sm, species="ml", label_col="neural_cluster")

# Sankey diagram (2 or 3 species)
sankey = splt.sankey_plot_3(MappingTable, species_order=["sm", "ml", "sc"])

# Expression overlap for a gene pair
splt.save_expression_overlap_plot(
    sm=sm,
    genes={"ml": "Mlig455-019997", "sm": "dd-Smed-v4-15930-0-1"},
    colors={"ml": "#377EB8", "sm": "#E41A1C"},
    output_prefix="results/overlap_macpiwi",
)
```

## Contributing

Follow the same conventions as `lab-core-python`: Google-style
docstrings, one function per logical task, lazy imports for optional
heavy dependencies (holoviews, bokeh).
