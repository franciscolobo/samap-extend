from __future__ import annotations

"""
Plotting utilities for SAMap cross-species integration results.

Holoviews and Bokeh are imported lazily — only required when the
Sankey functions are called.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# SAMap — Sankey
# ---------------------------------------------------------------------------

def sankey_plot_3(M, species_order, align_thr=0.1,
                  figsize=(16, 10), dpi=150,
                  node_width=0.015, node_pad=None,
                  flow_alpha=0.45, label_fontsize=8,
                  palette=None,
                  palette_file=None,
                  palette_cluster_col="cluster_id",
                  palette_label_col="label",
                  palette_color_col="color",
                  save_to=None, show=True):
    """Three-column Sankey diagram for a three-species SAMap analysis.

    Replaces the HoloViews-based implementation with a pure matplotlib
    approach that enforces explicit column positions (left = species[0],
    center = species[1], right = species[2]) regardless of graph topology.
    This prevents center-species nodes that only connect to one outer
    species from drifting to the wrong column.

    Args:
        M: SAMap mapping table (``pandas.DataFrame``) with index/columns
            labelled as ``"sp_Celltype"`` (e.g. ``"sm_Neuron"``).
        species_order: Exactly 3 species keys in left-to-right order,
            e.g. ``["sm", "ml", "sc"]``.
        align_thr: Drop edges whose value is below this threshold.
            Defaults to 0.1.
        figsize: Figure dimensions ``(width, height)`` in inches.
            Defaults to ``(16, 10)``.
        dpi: Figure resolution. Defaults to 150.
        node_width: Width of node bars in axes units (0–1).
            Defaults to 0.015.
        node_pad: Vertical padding between nodes within a column in
            axes units. Computed adaptively when ``None``.
        flow_alpha: Transparency of flow ribbons. Defaults to 0.45.
        label_fontsize: Font size for node labels. Defaults to 8.
        palette: Optional ``{node_label: color}`` dict mapping node labels
            to colours. Labels may be full (``"ml_neural_1"``) or
            cluster-only (``"neural_1"``); full matches take precedence.
            Takes precedence over ``palette_file``. When ``None`` and no
            ``palette_file`` is given, the built-in 20-colour palette is
            used.
        palette_file: Path to a TSV file with three columns: original
            cluster IDs, display labels, and colour values. Uses the same
            format as ``plot_joint_umap``. Ignored when ``palette`` is
            provided.
        palette_cluster_col: Column containing the original cluster IDs.
            Defaults to ``"cluster_id"``.
        palette_label_col: Column containing the display labels shown on
            the Sankey bars. Defaults to ``"label"``.
        palette_color_col: Column containing colour values (hex or named).
            Defaults to ``"color"``.
        show: Call ``plt.show()`` after drawing. Defaults to ``True``.

    Returns:
        A ``(fig, ax)`` tuple.

    Raises:
        ValueError: If ``species_order`` does not contain exactly 3 entries,
            or if no edges survive ``align_thr`` filtering.
    """
    from natsort import natsorted
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    if len(species_order) != 3:
        raise ValueError("species_order must contain exactly 3 species.")
    left_sp, center_sp, right_sp = species_order

    def _sp(arr):
        return np.array([s.split('_', 1)[0] for s in arr], dtype=str)

    def _prep(mat_df, src_sp, tgt_sp):
        rows = np.array([s.strip() for s in mat_df.index.astype(str)])
        cols = np.array([s.strip() for s in mat_df.columns.astype(str)])
        A = np.asarray(mat_df.values, dtype=float)
        A[~np.isfinite(A)] = 0.0
        A[A < align_thr] = 0.0
        ri, ci = A.nonzero()
        if ri.size == 0:
            return pd.DataFrame(columns=["source", "target", "value"])
        df = pd.DataFrame({"source": rows[ri].astype(str),
                           "target": cols[ci].astype(str),
                           "value":  A[ri, ci].astype(float)})
        sl = _sp(df.source.values) == src_sp
        tr = _sp(df.target.values) == tgt_sp
        sr = _sp(df.source.values) == tgt_sp
        tl = _sp(df.target.values) == src_sp
        klr, krl = sl & tr, sr & tl
        if krl.any():
            df.loc[krl, ["source", "target"]] = df.loc[krl, ["target", "source"]].values
        df = df.loc[klr | krl].copy()
        return df.groupby(["source", "target"], as_index=False)["value"].sum() \
            if not df.empty else df

    e_lc = _prep(M, left_sp, center_sp)
    e_cr = _prep(M, center_sp, right_sp)

    if e_lc.empty and e_cr.empty:
        raise ValueError("No edges survive align_thr filtering.")

    _default_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]

    # Build node→color lookup: palette dict > palette_file > built-in
    _node_colors: dict = {}
    if palette is not None:
        _node_colors = {str(k).strip(): str(v).strip() for k, v in palette.items()}
    elif palette_file is not None:
        pal_df = pd.read_csv(palette_file, sep="\t", dtype=str)
        missing = {palette_cluster_col, palette_color_col} - set(pal_df.columns)
        if missing:
            raise ValueError(
                f"palette_file is missing columns: {missing}. "
                f"Available: {list(pal_df.columns)}"
            )
        _node_colors = dict(zip(
            pal_df[palette_cluster_col].str.strip(),
            pal_df[palette_color_col].str.strip(),
        ))

    def _node_color(label: str, fallback_index: int) -> str:
        """Resolve colour for a node label.

        Lookup order:
          1. Exact match on full label (e.g. 'ml_neural_1').
          2. Match on cluster portion only — label with species prefix
             stripped (e.g. 'neural_1' from 'ml_neural_1').
          3. Built-in palette by position.
        """
        if label in _node_colors:
            return _node_colors[label]
        # strip species prefix (everything up to and including the first '_')
        cluster_only = label.split("_", 1)[-1] if "_" in label else label
        if cluster_only in _node_colors:
            return _node_colors[cluster_only]
        return _default_palette[fallback_index % len(_default_palette)]

    # Node sets (natsorted so cluster_1 < cluster_2 < cluster_10)
    left_nodes   = natsorted(set(e_lc["source"].tolist()) if not e_lc.empty else [])
    center_nodes = natsorted(
        (set(e_lc["target"].tolist()) if not e_lc.empty else set()) |
        (set(e_cr["source"].tolist()) if not e_cr.empty else set())
    )
    right_nodes  = natsorted(set(e_cr["target"].tolist()) if not e_cr.empty else [])

    # Node heights: max of incoming and outgoing flow
    def _sum(df, col, node):
        return float(df.loc[df[col] == node, "value"].sum()) if not df.empty else 0.0

    w_l = {n: _sum(e_lc, "source", n) for n in left_nodes}
    w_c = {n: max(_sum(e_lc, "target", n), _sum(e_cr, "source", n)) for n in center_nodes}
    w_r = {n: _sum(e_cr, "target", n) for n in right_nodes}

    # Column layout: normalize to fill [0,1] minus inter-node padding
    def lay(nodes, weights, pad):
        raw = sum(weights[n] for n in nodes) or 1.0
        usable = 1.0 - max(len(nodes) - 1, 0) * pad
        scale = usable / raw
        pos, cur = {}, 0.0
        for n in nodes:
            h = weights[n] * scale
            pos[n] = (cur, h)
            cur += h + pad
        return pos, scale

    n_max = max(len(left_nodes), len(center_nodes), len(right_nodes), 1)
    _pad = node_pad if node_pad is not None else min(0.025, 0.6 / n_max)

    pos_l, sc_l = lay(left_nodes,   w_l, _pad)
    pos_c, sc_c = lay(center_nodes, w_c, _pad)
    pos_r, sc_r = lay(right_nodes,  w_r, _pad)

    # Colors
    _pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
            "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"]
    col_l = {n: _node_color(n, i) for i, n in enumerate(left_nodes)}
    col_c = {n: _node_color(n, i) for i, n in enumerate(center_nodes)}
    col_r = {n: _node_color(n, i) for i, n in enumerate(right_nodes)}

    NW = node_width
    x_l, x_c, x_r = 0.12, 0.50, 0.88

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Draw node bars and labels
    for n, (y0, h) in pos_l.items():
        ax.add_patch(plt.Rectangle((x_l, y0), NW, h, fc=col_l[n], ec="none", zorder=3))
        ax.text(x_l - 0.005, y0 + h / 2, _node_label(n), ha="right", va="center",
                fontsize=label_fontsize, clip_on=False)

    for n, (y0, h) in pos_c.items():
        ax.add_patch(plt.Rectangle((x_c - NW / 2, y0), NW, h, fc=col_c[n], ec="none", zorder=3))
        ax.text(x_c + NW / 2 + 0.005, y0 + h / 2, _node_label(n), ha="left", va="center",
                fontsize=label_fontsize, clip_on=False)

    for n, (y0, h) in pos_r.items():
        ax.add_patch(plt.Rectangle((x_r - NW, y0), NW, h, fc=col_r[n], ec="none", zorder=3))
        ax.text(x_r + 0.005, y0 + h / 2, _node_label(n), ha="left", va="center",
                fontsize=label_fontsize, clip_on=False)

    # Column titles
    for x, lbl in [(x_l + NW / 2, left_sp),
                   (x_c,           center_sp),
                   (x_r - NW / 2,  right_sp)]:
        ax.text(x, 1.02, lbl, ha="center", va="bottom",
                fontsize=label_fontsize + 2, fontweight="bold", clip_on=False)

    # Bezier flow ribbon
    def draw_flow(x0, y0, h0, x1, y1, h1, color):
        dx = x1 - x0
        cx0, cx1 = x0 + dx * 0.45, x0 + dx * 0.55
        verts = [
            (x0, y0 + h0), (cx0, y0 + h0), (cx1, y1 + h1), (x1, y1 + h1),
            (x1, y1),      (cx1, y1),      (cx0, y0),       (x0, y0),
            (x0, y0 + h0),
        ]
        codes = [Path.MOVETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.LINETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes),
                               fc=color, ec="none", alpha=flow_alpha, zorder=1))

    # Flow offset trackers (how far up each node's bar we've consumed)
    off_l_r = {n: pos_l[n][0] for n in left_nodes}
    off_c_l = {n: pos_c[n][0] for n in center_nodes}
    off_c_r = {n: pos_c[n][0] for n in center_nodes}
    off_r_l = {n: pos_r[n][0] for n in right_nodes}

    # sm → ml flows
    for _, row in e_lc.sort_values("value", ascending=False).iterrows():
        src, tgt, val = row["source"], row["target"], float(row["value"])
        if src not in pos_l or tgt not in pos_c:
            continue
        h_l, h_c = val * sc_l, val * sc_c
        draw_flow(x_l + NW, off_l_r[src], h_l,
                  x_c - NW / 2, off_c_l[tgt], h_c, col_c[tgt])
        off_l_r[src] += h_l
        off_c_l[tgt] += h_c

    # ml → sc flows
    for _, row in e_cr.sort_values("value", ascending=False).iterrows():
        src, tgt, val = row["source"], row["target"], float(row["value"])
        if src not in pos_c or tgt not in pos_r:
            continue
        h_c, h_r = val * sc_c, val * sc_r
        draw_flow(x_c + NW / 2, off_c_r[src], h_c,
                  x_r - NW, off_r_l[tgt], h_r, col_c[src])
        off_c_r[src] += h_c
        off_r_l[tgt] += h_r

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    if save_to:
        fig.savefig(save_to, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


# ---------------------------------------------------------------------------
# SAMap — UMAP scatter
# ---------------------------------------------------------------------------

def square_scatter(
    sm,
    figsize=(8, 8),
    dpi=600,
    COLORS=None,
    species_labels=None,
    legend=True,
    legend_title=None,
    legend_loc="center left",
    legend_bbox=(1.02, 0.5),
    legend_markersize=10,
    s=None,
    ss=None,
    axes=None,
    **kwargs,
):
    """Render a SAMap UMAP scatter on a square, publication-ready Axes.

    Wraps ``sm.scatter()``, enforces equal aspect ratio and a centred
    view window, strips any extra axes that SAMap creates internally,
    and optionally adds a clean legend.

    Args:
        sm: A ``samap.SAMAP`` instance.
        figsize: Figure dimensions ``(width, height)`` in inches.
            Defaults to ``(8, 8)``.
        dpi: Figure resolution. Defaults to 600.
        COLORS: ``{species_id: color}`` mapping for point and legend
            colours. When ``None`` SAMap's defaults apply.
        species_labels: ``{species_id: display_label}`` mapping used
            only in the legend. Falls back to ``species_id`` when
            ``None``.
        legend: Draw a legend when ``True``. Requires ``COLORS``.
            Defaults to ``True``.
        legend_title: Optional legend title string.
        legend_loc: Matplotlib legend location string.
            Defaults to ``"center left"``.
        legend_bbox: ``bbox_to_anchor`` tuple for the legend.
            Defaults to ``(1.02, 0.5)``.
        legend_markersize: Marker size in the legend. Defaults to 10.
        s: Uniform marker size for all species. Mutually exclusive with
            ``ss``; ignored when ``ss`` is provided.
        ss: ``{species_id: size}`` per-species marker sizes. Takes
            precedence over ``s``.
        axes: Existing ``matplotlib.Axes`` to draw onto. A new figure
            and axes are created when ``None``.
        **kwargs: Forwarded to ``sm.scatter()``. The keys
            ``legend_loc``, ``legend_loc_bounds``, and
            ``legend_fontsize`` are stripped before forwarding to avoid
            conflicts.

    Returns:
        A ``(fig, ax)`` tuple.
    """
    for bad in ("legend_loc", "legend_loc_bounds", "legend_fontsize"):
        kwargs.pop(bad, None)

    if ss is None and s is not None:
        ss = {sid: float(s) for sid in sm.ids}

    if axes is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        ax = axes
        fig = ax.figure
        fig.set_size_inches(*figsize)
        fig.set_dpi(dpi)

    sm.scatter(axes=ax, COLORS=(COLORS or {}), ss=(ss or {}), **kwargs)

    for extra in list(fig.axes):
        if extra is not ax:
            fig.delaxes(extra)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0) / 2.0
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_position([0, 0, 1, 1])

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    if legend and COLORS is not None:
        handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                label=(species_labels or {}).get(sid, sid),
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=legend_markersize,
            )
            for sid, color in COLORS.items()
        ]
        ax.legend(
            handles=handles,
            title=legend_title,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox,
            frameon=False,
        )

    return fig, ax


# ---------------------------------------------------------------------------
# SAMap — expression overlap plot
# ---------------------------------------------------------------------------

def save_expression_overlap_plot(
    sm,
    genes,
    colors,
    species_labels=None,
    overlap_color="#FFEE8C",
    figsize=(10, 10),
    dpi=600,
    output_prefix="expression_overlap",
    legend_title=None,
    markersize=10,
    legend_loc="center left",
    legend_bbox=(1.02, 0.5),
    margin=0.02,
    point_size=5,
    save_pdf=False,
    show=True,
):
    """Plot and save a SAMap expression-overlap UMAP.

    Wraps ``sm.plot_expression_overlap()``, enforces a square axis,
    adds a clean legend (including an overlap entry), and saves the
    result as PNG and optionally PDF.

    Args:
        sm: A ``samap.SAMAP`` instance.
        genes: ``{species_id: gene_name}`` dict passed directly to
            ``sm.plot_expression_overlap()``.
        colors: ``{species_id: color}`` dict for per-species point
            colours.
        species_labels: ``{species_id: display_label}`` mapping for the
            legend. Falls back to ``species_id`` when ``None``.
        overlap_color: Colour for cells co-expressing genes from both
            species. Defaults to ``"#FFEE8C"``.
        figsize: Figure dimensions ``(width, height)`` in inches.
            Defaults to ``(10, 10)``.
        dpi: Figure resolution. Defaults to 600.
        output_prefix: File path prefix (without extension) for saved
            outputs. Defaults to ``"expression_overlap"``.
        legend_title: Optional legend title string.
        markersize: Marker size in the legend. Defaults to 10.
        legend_loc: Matplotlib legend location string.
            Defaults to ``"center left"``.
        legend_bbox: ``bbox_to_anchor`` tuple for the legend.
            Defaults to ``(1.02, 0.5)``.
        margin: Fractional axis margin added after enforcing square
            limits. Defaults to 0.02.
        point_size: Uniform scatter point size forwarded to
            ``sm.plot_expression_overlap()`` for all species.
            Defaults to 5.
        save_pdf: Also save a PDF alongside the PNG when ``True``.
            Defaults to ``False``.
        show: Call ``plt.show()`` after saving when ``True``, otherwise
            close the figure. Defaults to ``True``.

    Returns:
        A ``(fig, ax)`` tuple.
    """
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    plt.close("all")

    ss = {sid: point_size for sid in genes}
    sm.plot_expression_overlap(genes, COLORS=colors, COLORC=overlap_color, ss=ss)

    fig = plt.gcf()
    ax = plt.gca()
    fig.set_size_inches(*figsize)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0
    half = max(xmax - xmin, ymax - ymin) / 2.0
    ax.set_xlim(xmid - half, xmid + half)
    ax.set_ylim(ymid - half, ymid + half)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(margin)

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            label=(species_labels or {}).get(sid, sid),
            markerfacecolor=colors[sid],
            markeredgecolor="none",
            markersize=markersize,
        )
        for sid in genes
    ]
    handles.append(
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            label="Overlap",
            markerfacecolor=overlap_color,
            markeredgecolor="none",
            markersize=markersize,
        )
    )
    ax.legend(
        handles=handles,
        title=legend_title,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
        frameon=False,
    )

    fig.savefig(f"{output_prefix}.png", dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


# ---------------------------------------------------------------------------
# SAMap — joint UMAP with per-species label overlay
# ---------------------------------------------------------------------------

def plot_joint_umap(
    sm,
    species,
    label_col,
    embedding_key="X_umap_samap",
    figsize=(8, 8),
    dpi=300,
    s_other=6,
    s_target=60,
    alpha_other=0.6,
    other_color="lightgray",
    palette=None,
    palette_file=None,
    palette_cluster_col="cluster_id",
    palette_label_col="label",
    palette_color_col="color",
    save_to=None,
):
    """Plot a SAMap joint UMAP highlighting cell-type labels for one species.

    Renders two layers: background points from all other species in gray,
    and foreground points from ``species`` coloured by ``label_col``.

    Args:
        sm: A ``samap.SAMAP`` instance whose ``sams`` dict holds per-species
            SAM objects.
        species: Species key whose labels to highlight (e.g. ``"ml"``).
        label_col: Column in ``adata.obs`` containing the cell-type labels
            for ``species``.
        embedding_key: Key in ``.obsm`` for the joint UMAP coordinates.
            Defaults to ``"X_umap_samap"``.
        figsize: Figure dimensions ``(width, height)`` in inches.
            Defaults to ``(8, 8)``.
        dpi: Figure resolution. Defaults to 300.
        s_other: Marker size for background (non-target) cells.
            Defaults to 6.
        s_target: Marker size for the highlighted species' cells.
            Defaults to 60.
        alpha_other: Opacity for background cells. Defaults to 0.6.
        other_color: Colour for background cells. Defaults to
            ``"lightgray"``.
        palette: Optional list of hex/named colours for the target
            species' categories. Ignored when ``palette_file`` is
            provided. Falls back to a built-in 20-colour palette when
            both ``palette`` and ``palette_file`` are ``None``.
        palette_file: Path to a TSV file mapping cluster IDs to colours.
            When provided, takes precedence over ``palette``. Any
            cluster not present in the file falls back to the built-in
            palette. Defaults to ``None``.
        palette_cluster_col: Column name in ``palette_file`` that
            contains cluster IDs. Defaults to ``"cluster_id"``.
        palette_color_col: Column name in ``palette_file`` that contains
            colour values (hex or named). Defaults to ``"color"``.
        save_to: File path to save the figure (e.g. ``"umap.png"``).
            Skipped when ``None``.

    Returns:
        A ``(fig, ax)`` tuple.

    Raises:
        KeyError: If ``embedding_key`` is absent from any per-species
            ``.obsm``.
        ValueError: If ``label_col`` is not found in the target species'
            ``.obs``.
    """
    import anndata as ad

    adatas = {k: sm.sams[k].adata for k in sm.sams}

    target_adata = adatas[species]
    if label_col not in target_adata.obs.columns:
        raise ValueError(
            f"'{label_col}' not found in obs for species '{species}'. "
            f"Available columns: {list(target_adata.obs.columns)}"
        )
    for k, a in adatas.items():
        if embedding_key not in a.obsm:
            raise KeyError(
                f"Embedding key '{embedding_key}' not found in obsm for "
                f"species '{k}'. Available keys: {list(a.obsm.keys())}"
            )

    combined = ad.concat(adatas, label="species", join="outer", index_unique=None)
    combined.obsm[embedding_key] = np.vstack(
        [adatas[k].obsm[embedding_key] for k in adatas]
    )

    mask = combined.obs["species"].astype(str).eq(species)
    combined.obs["_plot_labels"] = np.where(
        mask,
        combined.obs[label_col].astype(str),
        "other",
    )
    combined.obs["_plot_labels"] = combined.obs["_plot_labels"].astype("category")

    from natsort import natsorted
    cats = list(combined.obs["_plot_labels"].cat.categories)
    target_cats = natsorted([c for c in cats if c != "other"])
    _default_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]

    # Build color map: palette_file > palette > built-in default
    if palette_file is not None:
        pal_df = pd.read_csv(palette_file, sep="\t", dtype=str)
        missing = {palette_cluster_col, palette_color_col} - set(pal_df.columns)
        if missing:
            raise ValueError(
                f"palette_file is missing expected columns: {missing}. "
                f"Available columns: {list(pal_df.columns)}"
            )
        file_palette = dict(zip(
            pal_df[palette_cluster_col].str.strip(),
            pal_df[palette_color_col].str.strip(),
        ))
        # Fill any clusters missing from file with the built-in palette
        color_map = {"other": other_color}
        color_map.update({
            c: file_palette.get(c, _default_palette[i % len(_default_palette)])
            for i, c in enumerate(target_cats)
        })
    else:
        _palette = palette or _default_palette
        color_map = {"other": other_color}
        color_map.update({c: _palette[i % len(_palette)] for i, c in enumerate(target_cats)})
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    xy = combined.obsm[embedding_key]
    is_target = combined.obs["species"].values == species
    labels = combined.obs["_plot_labels"].values

    # Layer 1 — background: all non-target species cells in other_color
    ax.scatter(
        xy[~is_target, 0], xy[~is_target, 1],
        c=other_color,
        s=s_other,
        alpha=alpha_other,
        linewidths=0,
        rasterized=True,
    )

    # Layer 2 — foreground: target species cells coloured by cluster label
    if target_cats:
        target_colors = [color_map[lbl] for lbl in labels[is_target]]
        ax.scatter(
            xy[is_target, 0], xy[is_target, 1],
            c=target_colors,
            s=s_target,
            linewidths=0,
            rasterized=True,
        )

        # Legend — one entry per target cluster
        handles = [
            Line2D([0], [0], marker="o", linestyle="None",
                   label=cat, markerfacecolor=color_map[cat],
                   markeredgecolor="none", markersize=8)
            for cat in target_cats
        ]
        ax.legend(handles=handles, loc="right", bbox_to_anchor=(1.3, 0.5),
                  frameon=False, fontsize=8)

    ax.set_aspect("equal")
    ax.axis("off")

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    if save_to:
        fig.savefig(save_to, dpi=dpi, bbox_inches="tight")

    return fig, ax
