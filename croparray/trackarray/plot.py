from __future__ import annotations
"""
croparray.plot
==============

Notebook-friendly plotting helpers for TrackArray / CropArray workflows.

This module intentionally keeps dependencies light (matplotlib + xarray; seaborn is
used only inside `plot_track_signal_traces`).

Key conventions
--------------
1) Track selection is always `track_ids` (plural).
   - Backward-compatible aliases (e.g., `track_id`) are accepted but deprecated.

2) RGB merging (compositing) is explicit and works even for 2-channel datasets.
   - Controlled by `show_merge` and `show_merge_chs`.

RGB merge semantics
-------------------
`show_merge_chs` is interpreted as a *source-channel mapping* for the RGB planes:

    show_merge_chs = (r_src, g_src, b_src)

Each entry is a channel index in the source data (the `ch` dimension). Duplicates
are allowed and are the standard way to create magenta/yellow/cyan-style overlays:

- (0, 1, 0): R=ch0, G=ch1, B=ch0  -> ch0 appears magenta, ch1 appears green
- (1, 1, 0): R=ch1, G=ch1, B=ch0  -> ch1 appears yellow,  ch0 appears blue
- (0, 0, 0): grayscale display of ch0 replicated across RGB

This is intentionally different from “weights”; these are *source indices*.

Functions
---------
- plot_trackarray_crops: image crops over time, optional RGB merge + per-channel grayscale
- plot_track_signal_traces: per-track time traces for a selected variable (optionally channelled)
"""

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

__all__ = ["plot_trackarray_crops", "plot_track_signal_traces", "trajectories_xy"]


# -----------------------
# Small shared utilities
# -----------------------

_EPS = 1e-6


def _as_list_of_ints(x: Union[int, Sequence[int], np.ndarray]) -> List[int]:
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) for v in list(x)]
    return [int(x)]


def _is_binary_like(a: xr.DataArray, max_check: int = 1_000_000) -> bool:
    """
    Heuristic test for whether an array is a mask-like layer.

    Returns True for:
    - dtype=bool
    - values that appear restricted to {0, 1} or {0, 1, 255} (with optional NaNs)

    Notes
    -----
    We sample up to `max_check` values to avoid scanning very large arrays.
    """
    if a.dtype == bool:
        return True

    arr = np.asarray(a.data) if hasattr(a, "data") else np.asarray(a.values)
    flat = arr.ravel()
    if flat.size == 0:
        return False

    if flat.size > max_check:
        step = int(np.ceil(flat.size / max_check))
        flat = flat[::step]

    if np.issubdtype(flat.dtype, np.floating):
        flat = flat[~np.isnan(flat)]
        if flat.size == 0:
            return False

    u = np.unique(flat)
    return set(u.tolist()).issubset({0, 1, 255})


def _normalize_for_display(
    a: xr.DataArray,
    *,
    quantile_range: Tuple[float, float] = (0.02, 0.99),
) -> xr.DataArray:
    """
    Normalize image-like data into [0, 1] for display.

    Rules
    -----
    - Binary-like masks -> convert to float and display in [0, 1]
      (255 is treated as 1).
    - Otherwise -> quantile-normalize using positive pixels only.

    Parameters
    ----------
    a
        Array to normalize.
    quantile_range
        Quantiles (low, high) for normalization, computed from positive pixels only.

    Returns
    -------
    xr.DataArray
        Normalized array in [0, 1].
    """
    if _is_binary_like(a):
        out = a.astype(float)
        out = xr.where(out == 255, 1.0, out).clip(0, 1)
        return out

    pos = a.where(lambda x: x > 0)
    if int(pos.count()) == 0:
        q0, q1 = 0.0, 1.0
    else:
        q0 = pos.quantile(quantile_range[0])
        q1 = pos.quantile(quantile_range[1])

    out = ((a - q0) / (q1 - q0 + _EPS)).clip(0, 1)
    return out

def _facetgrid_cleanup(g, *, suppress_labels: bool, suptitle: Optional[str]) -> None:
    """
    Best-effort cleanup for xarray FacetGrid outputs.
    """
    if suppress_labels:
        try:
            g.set_titles("")
        except Exception:
            try:
                g.set_titles(template="")
            except Exception:
                pass
        try:
            g.set_xlabels("")
            g.set_ylabels("")
        except Exception:
            pass
        try:
            # remove ticks if present
            for ax in getattr(g, "axes", np.array([])).ravel():
                if ax is not None:
                    ax.set_xticks([])
                    ax.set_yticks([])
        except Exception:
            pass

    if suptitle and hasattr(g, "fig") and g.fig is not None:
        try:
            g.fig.suptitle(suptitle, y=1.02)
        except Exception:
            pass


# -----------------------
# Public API
# -----------------------
def plot_trackarray_crops(
    ds: xr.Dataset,
    *,
    layer: str = "best_z",
    fov: int = 0,
    track_ids: Union[int, Sequence[int], np.ndarray] = (1,),
    t: Tuple[int, int, int] = (0, 10, 3),
    rolling: int = 1,
    quantile_range: Tuple[float, float] = (0.02, 0.99),
    # Display
    show_grayscale: bool = True,
    show_merge_chs: Optional[Tuple[int, int, int]] = None,
    ch: Optional[int] = None,
    # Presentation
    suppress_labels: bool = True,
    show_suptitle: bool = True,
) -> Dict[int, xr.DataArray]:
    """
    Plot track-centered image crops across time using xarray.plot.imshow.

    Behavior
    --------
    - If the data has no `ch` dimension: plot grayscale.
    - If the data has `ch` and `ch is None`: plot each channel in grayscale (default).
      If `show_merge_chs` is provided, also plot an RGB composite using that mapping.
    - If `ch` is provided: plot only that channel in grayscale and skip merge.

    Parameters
    ----------
    show_merge_chs
        Optional mapping (r_src, g_src, b_src) using *positional* channel indices.
        Example for two channels: (0, 1, 0) -> ch0 in R/B, ch1 in G.

    Returns
    -------
    dict[int, xr.DataArray]
        Mapping track_id -> normalized DataArray used for plotting.
    """
    if layer not in ds:
        raise KeyError(f"Dataset must contain layer {layer!r}. Available: {list(ds.data_vars)}")

    da = ds[layer]
    for req in ("t", "y", "x"):
        if req not in da.dims:
            raise ValueError(f"Layer {layer!r} must include dim {req!r}. Found dims: {da.dims}")

    tids = _as_list_of_ints(track_ids)
    out: Dict[int, xr.DataArray] = {}

    for tid in tids:
        # Select track/fov if present
        bz = da.sel(track_id=tid) if "track_id" in da.dims else da
        if "fov" in bz.dims:
            bz = bz.sel(fov=fov)

        # Time slicing
        start, stop, step = t
        bz = bz.isel(t=slice(start, stop, step))

        # Rolling mean
        if rolling and rolling > 1:
            bz = bz.rolling(t=rolling, center=True, min_periods=1).mean()

        if "ch" in bz.dims:
            # canonical order; prepend any unknown dims (e.g. rna, znf)
            _known = {"t", "y", "x", "ch"}
            _extra = [d for d in bz.dims if d not in _known]
            bz = bz.transpose(*_extra, "t", "y", "x", "ch")

        # ---- No channel dimension ----
        if "ch" not in bz.dims:
            _known = {"t", "y", "x"}
            _extra = [d for d in bz.dims if d not in _known]
            bz = bz.transpose(*_extra, "t", "y", "x")  
            normed = _normalize_for_display(bz, quantile_range=quantile_range)

            g = normed.plot.imshow(
                col="t",
                cmap="gray",
                aspect=1,
                size=5,
                vmin=0,
                vmax=1,
                robust=False,
                add_labels=not suppress_labels,
                add_colorbar=False,
            )

            if show_suptitle:
                _facetgrid_cleanup(
                    g,
                    suppress_labels=suppress_labels,
                    suptitle=f"track_id={tid} | {layer}",
                )

            out[int(tid)] = normed
            continue

        # ---- Single-channel override ----
        if ch is not None:
            _extra1 = [d for d in bz.isel(ch=int(ch)).dims if d not in {"t", "y", "x"}]
            bz1 = bz.isel(ch=int(ch)).transpose(*_extra1, "t", "y", "x")
            normed = _normalize_for_display(bz1, quantile_range=quantile_range)

            g = normed.plot.imshow(
                col="t",
                cmap="gray",
                aspect=1,
                size=5,
                vmin=0,
                vmax=1,
                robust=False,
                add_labels=not suppress_labels,
                add_colorbar=False,
            )

            if show_suptitle:
                _facetgrid_cleanup(
                    g,
                    suppress_labels=suppress_labels,
                    suptitle=f"track_id={tid} | {layer} | ch={int(ch)}",
                )

            out[int(tid)] = normed
            continue

        # ---- Normalize each channel separately ----
        n_ch = int(bz.sizes["ch"])
        _known_tyx = {"t", "y", "x"}
        ch_normed = [
            _normalize_for_display(
                bz.isel(ch=i).transpose(
                    *[d for d in bz.isel(ch=i).dims if d not in _known_tyx], "t", "y", "x"
                ),
                quantile_range=quantile_range,
            )
            for i in range(n_ch)
        ]
        normed_all = xr.concat(ch_normed, dim="ch").assign_coords(ch=bz["ch"].values)
        out[int(tid)] = normed_all

        # ---- Grayscale panels (default) ----
        if show_grayscale:
            for i in range(n_ch):
                g = normed_all.isel(ch=i).plot.imshow(
                    col="t",
                    cmap="gray",
                    aspect=1,
                    size=5,
                    vmin=0,
                    vmax=1,
                    robust=False,
                    add_labels=not suppress_labels,
                    add_colorbar=False,
                )
                if show_suptitle:
                    _facetgrid_cleanup(
                        g,
                        suppress_labels=suppress_labels,
                        suptitle=f"track_id={tid} | {layer} | ch={i}",
                    )

        # ---- Optional RGB merge ----
        if show_merge_chs is not None:
            r_src, g_src, b_src = map(int, show_merge_chs)
            need_max = max(r_src, g_src, b_src)
            if n_ch <= need_max:
                raise ValueError(
                    f"show_merge_chs={show_merge_chs} requires at least {need_max+1} channels, "
                    f"but dataset has {n_ch}."
                )

            r_da = normed_all.isel(ch=r_src).expand_dims(rgb=["R"])
            g_da = normed_all.isel(ch=g_src).expand_dims(rgb=["G"])
            b_da = normed_all.isel(ch=b_src).expand_dims(rgb=["B"])

            rgb_da = xr.concat([r_da, g_da, b_da], dim="rgb")

            g = rgb_da.plot.imshow(
                col="t",
                rgb="rgb",
                aspect=1,
                size=5,
                vmin=0,
                vmax=1,
                add_labels=not suppress_labels,
                add_colorbar=False,
            )

            if show_suptitle:
                _facetgrid_cleanup(
                    g,
                    suppress_labels=suppress_labels,
                    suptitle=f"track_id={tid} | {layer} | MERGE {tuple(show_merge_chs)}",
                )

    return out

def plot_track_signal_traces(
    ta_dataset: xr.Dataset,
    track_ids: Sequence[int],
    *,
    var: str = "signal",
    rgb: Optional[Tuple[int, int, int]] = (1, 1, 1),
    colors: Tuple[str, ...] = ("#00f670", "#f67000", "#7000f6"),
    markers: Tuple[str, ...] = ("o", "s", "D"),
    marker_size: int = 6,
    scatter_size: int = 25,
    markevery: int = 5,
    figsize: Tuple[float, float] = (7, 2.8),
    ylim=None,
    xlim=None,
    col_wrap: int = 3,
    y2: Optional[Union[int, str]] = None,
    y2lim=None,
    y2_label: Optional[str] = None,
    y2_color: Optional[str] = None,
    legend_loc: str = "upper right",
    show_legend: bool = True,
) -> None:
    """
    Plot per-track traces for a chosen variable (default: 'signal').

    Works for both:
    - channelled variables (dims include 'ch')
    - channel-less variables (no usable 'ch')

    Parameters
    ----------
    ta_dataset
        TrackArray dataset containing `var` with dims including (track_id, t) and optionally (ch).
    track_ids
        Track IDs to plot.
    var
        Variable name to plot.
    rgb
        Channel inclusion mask for the left axis (unless a channel is assigned to `y2`).
        If None, plot all channels. Ignored for channel-less variables.
    colors, markers
        Per-channel colors/markers (cycled by channel index).
    marker_size, scatter_size, markevery
        Marker and scatter display controls.
    figsize
        Base size per subplot; actual figure size scales with `col_wrap` and number of tracks.
    ylim, xlim
        Axis limits for the left axis.
    col_wrap
        Number of columns in the subplot grid.
    y2
        If an int, place that channel index on a secondary (right) y-axis (legacy behavior).
        If a str, treat as a variable name to plot on the right y-axis (may be channel-less).
    y2lim, y2_label
        Right-axis limits/label.
    y2_color
        Color for the right y-axis line/ticks/label. Defaults to "gray" when y2 is a variable
        name, or the corresponding channel color when y2 is an int.
    legend_loc
        'upper right', 'best', etc. Use 'outside' to place legend outside axes.
    show_legend
        Toggle legend on/off.

    Returns
    -------
    None
        Displays the figure via matplotlib.
    """
    # Imported locally to keep this module lightweight for image plotting use cases.
    from ..dataframe import variables_to_df
    import math
    import seaborn as sns

    if var not in ta_dataset:
        raise KeyError(f"Dataset does not contain variable {var!r}")

    df = variables_to_df(ta_dataset, [var])

    # Determine whether this variable is channelled in the dataframe.
    has_ch = ("ch" in df.columns) and df["ch"].notna().any()

    # y2 as a variable name: load a separate df for the right axis.
    y2_is_var = isinstance(y2, str)
    df2 = None
    if y2_is_var:
        if y2 not in ta_dataset:
            raise KeyError(f"y2 variable {y2!r} not found. Available: {list(ta_dataset.data_vars)}")
        df2 = variables_to_df(ta_dataset, [y2])

    n = len(track_ids)
    ncols = int(col_wrap)
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize[0] * ncols, figsize[1] * nrows),
        squeeze=False,
    )

    # Ensure markers covers all channels
    if len(markers) < len(colors):
        markers = tuple(list(markers) + ["o"] * (len(colors) - len(markers)))

    def _rgb_on(ch: int) -> bool:
        if rgb is None:
            return True
        if ch < len(rgb):
            return bool(rgb[ch])
        return False

    for idx, track_id in enumerate(track_ids):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        ax2 = ax.twinx() if (y2_is_var or (has_ch and y2 is not None)) else None

        if not has_ch:
            subset = df[df["track_id"] == track_id]
            if not subset.empty:
                color0 = colors[0] if len(colors) else None
                marker0 = markers[0] if len(markers) else "o"

                sns.lineplot(
                    data=subset,
                    x="t",
                    y=var,
                    ax=ax,
                    color=color0,
                    lw=2,
                    dashes=False,
                    legend=False,
                    marker=marker0,
                    markersize=marker_size,
                    markevery=markevery,
                )

                mean_df = subset.groupby("t")[var].mean().reset_index()
                sns.scatterplot(
                    data=mean_df,
                    x="t",
                    y=var,
                    ax=ax,
                    color=color0,
                    s=scatter_size,
                    legend=False,
                )
        else:
            for ch_i in range(len(colors)):
                if not (_rgb_on(ch_i) or (y2 == ch_i)):
                    continue

                color = colors[ch_i]
                marker = markers[ch_i]

                subset = df[(df["track_id"] == track_id) & (df["ch"] == ch_i)]
                if subset.empty:
                    continue

                target_ax = ax2 if (ax2 is not None and ch_i == y2) else ax

                sns.lineplot(
                    data=subset,
                    x="t",
                    y=var,
                    ax=target_ax,
                    color=color,
                    label=f"ch {ch_i}",
                    lw=2,
                    dashes=False,
                    legend=False,
                    marker=marker,
                    markersize=marker_size,
                    markevery=markevery,
                )

                mean_df = subset.groupby("t")[var].mean().reset_index()
                sns.scatterplot(
                    data=mean_df,
                    x="t",
                    y=var,
                    ax=target_ax,
                    color=color,
                    s=scatter_size,
                    legend=False,
                )

        # Plot y2 variable (string case) on right axis.
        if y2_is_var and ax2 is not None and df2 is not None:
            subset2 = df2[df2["track_id"] == track_id]
            if "ch" in subset2.columns and subset2["ch"].notna().any():
                subset2 = subset2[subset2["ch"] == subset2["ch"].iloc[0]]
            if not subset2.empty:
                _y2_color = y2_color or "gray"
                sns.lineplot(
                    data=subset2, x="t", y=y2, ax=ax2,
                    color=_y2_color, lw=2, dashes=False, legend=False,
                    marker="o", markersize=marker_size, markevery=markevery,
                )
                mean_df2 = subset2.groupby("t")[y2].mean().reset_index()
                sns.scatterplot(
                    data=mean_df2, x="t", y=y2, ax=ax2,
                    color=_y2_color, s=scatter_size, legend=False,
                )

        ax.set_title(f"Track {int(track_id)}")
        ax.set_xlabel("time (sec)")
        ax.set_ylabel(f"{var} (a.u.)")
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)

        if ax2 is not None:
            if y2_is_var:
                right_color = y2_color or "gray"
                ax2.set_ylabel(y2_label or y2, color=right_color)
            else:
                right_color = colors[int(y2) % len(colors)]
                ax2.set_ylabel(y2_label or f"{var} (a.u.) [ch {y2}]", color=right_color)
            if y2lim is not None:
                ax2.set_ylim(y2lim)
            if xlim is not None:
                ax2.set_xlim(xlim)
            ax2.tick_params(axis="y", colors=right_color)
            ax2.spines["right"].set_color(right_color)

        if show_legend and has_ch:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = (ax2.get_legend_handles_labels() if ax2 else ([], []))
            handles, labels = h1 + h2, l1 + l2

            if legend_loc == "outside":
                ax.legend(
                    handles,
                    labels,
                    loc="upper left",
                    bbox_to_anchor=(1.15, 1.0),
                    borderaxespad=0.0,
                    frameon=True,
                )
            else:
                ax.legend(handles, labels, loc=legend_loc, frameon=True)
        else:
            if ax.get_legend():
                ax.get_legend().remove()

    # Hide unused subplots
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        fig.delaxes(axes[r][c])

    if legend_loc == "outside":
        fig.subplots_adjust(right=0.82)

    plt.tight_layout()
    plt.show()



def trajectories_xy(
    self,
    xvar: str = "xc",
    yvar: str = "yc",
    hue: str | None = None,
    col: str | None = None,
    row: str | None = None,
    space: str = "units",
    center_tracks: bool = False,
    alpha: float = 0.5,
    linewidth: float = 1.0,
    height: float = 4,
    aspect: float = 1,
    dropna: bool = True,
    legend: bool = True,
    color_time: bool = False,
    time_var: str = "t",
    time_palette: str = "viridis",
    add_colorbar: bool = True,
    label_tracks: bool = False,
    label_offset: float = 0.1,
    label_fontsize: float = 8,
    facet_kws: dict | None = None,
    **kwargs,
):
    """
    Plot XY trajectories with optional faceting and optional time-colored paths.

    Parameters
    ----------
    xvar, yvar : str
        Position variables, typically 'xc' and 'yc'.
    hue, col, row : str or None
        Grouping/faceting variables.
    space : {'units', 'pixels'}
        Convert coordinates using dx/dy if 'units'.
    center_tracks : bool
        If True, shift each trajectory so it starts at (0, 0).
    alpha : float
        Transparency for lines.
    linewidth : float
        Line width.
    height, aspect : float
        Facet sizing.
    dropna : bool
        Drop rows with NaN x/y before plotting.
    legend : bool
        Show legend when using categorical hue.
    color_time : bool
        If True, color each trajectory segment by time.
        When True, `hue` is ignored.
    time_var : str
        Column used for time coloring, usually 't'.
    time_palette : str
        Matplotlib colormap name for time coloring.
    add_colorbar : bool
        If True and color_time=True, add a shared colorbar.
    label_tracks : bool
        If True, label each trajectory at its first valid point using track_id.
    label_offset : float
        Offset added to x and y when placing trajectory labels.
        Interpreted in plot coordinates.
    label_fontsize : float
        Font size for trajectory labels.
    facet_kws : dict or None
        Extra kwargs passed to seaborn.FacetGrid.
    **kwargs
        Extra kwargs passed to seaborn.lineplot when color_time=False.

    Returns
    -------
    g : seaborn.FacetGrid
    """
    import numpy as np
    import seaborn as sns
    import xarray as xr
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    ds = self._obj if hasattr(self, "_obj") else self.ds if hasattr(self, "ds") else self

    if xvar not in ds or yvar not in ds:
        raise ValueError(f"Dataset must contain '{xvar}' and '{yvar}'")

    if space not in {"units", "pixels"}:
        raise ValueError("space must be 'units' or 'pixels'")

    if color_time and time_var not in ds.dims and time_var not in ds.coords and time_var not in ds.variables:
        raise ValueError(f"time_var='{time_var}' not found in dataset")

    # ---- scale coordinates ----
    x = ds[xvar]
    y = ds[yvar]

    if space == "units":
        dx = getattr(self, "dx", None)
        dy = getattr(self, "dy", dx)
        if dx is None:
            raise ValueError("space='units' requested but dx not found")
        x = x * dx
        y = y * dy
        units = getattr(self, "xyz_units", None)
    else:
        units = "px"

    # ---- gather variables needed in dataframe ----
    needed = {"_x": x, "_y": y}

    for v in [hue, col, row]:
        if v is not None and v not in needed:
            if v in ds:
                needed[v] = ds[v]
            elif v in ds.coords:
                needed[v] = ds.coords[v]

    if color_time and time_var in ds and time_var not in needed:
        needed[time_var] = ds[time_var]

    tmp = xr.Dataset(needed)
    df = tmp.to_dataframe().reset_index()

    # avoid duplicate columns after reset_index
    df = df.loc[:, ~df.columns.duplicated()].copy()

    if dropna:
        df = df.dropna(subset=["_x", "_y"])

    # ---- trajectory ID ----
    id_cols = [c for c in ["exp", "fov", "track_id"] if c in df.columns]
    if not id_cols:
        raise ValueError("Need exp/fov/track_id to define trajectories")

    df["_traj_id"] = df[id_cols].astype(str).agg(" | ".join, axis=1)

    # ---- sort by time ----
    if time_var in df.columns:
        df = df.sort_values(id_cols + [time_var])
    else:
        df = df.sort_values(id_cols)

    # ---- center tracks ----
    if center_tracks:
        df["_x"] = df["_x"] - df.groupby("_traj_id")["_x"].transform("first")
        df["_y"] = df["_y"] - df.groupby("_traj_id")["_y"].transform("first")

    facet_kws = {} if facet_kws is None else dict(facet_kws)

    # ---- build FacetGrid ----
    g = sns.FacetGrid(
        data=df,
        row=row,
        col=col,
        height=height,
        aspect=aspect,
        legend_out=True,
        **facet_kws,
    )

    def _track_label_from_traj_id(traj_id: str) -> str:
        parts = [p.strip() for p in str(traj_id).split("|")]
        return parts[-1] if len(parts) else str(traj_id)

    if not color_time:
        # standard single-color-per-track mode
        def _plot_lines(data, color=None, **kws):
            import matplotlib.pyplot as plt

            ax = plt.gca()

            sns.lineplot(
                data=data,
                x="_x",
                y="_y",
                units="_traj_id",
                estimator=None,
                hue=hue,
                alpha=alpha,
                linewidth=linewidth,
                legend=False,
                sort=False,
                ax=ax,
                **kwargs,
            )

            if label_tracks:
                for traj_id, sub in data.groupby("_traj_id", sort=False):
                    sort_cols = [time_var] if time_var in sub.columns else []
                    if sort_cols:
                        sub = sub.sort_values(sort_cols)

                    sub = sub.dropna(subset=["_x", "_y"])
                    if len(sub) == 0:
                        continue

                    x0 = sub["_x"].iloc[0]
                    y0 = sub["_y"].iloc[0]

                    ax.text(
                        x0 + label_offset,
                        y0 + label_offset,
                        _track_label_from_traj_id(traj_id),
                        fontsize=label_fontsize,
                        color=color if color is not None else "black",
                        alpha=min(1.0, alpha + 0.2),
                        ha="left",
                        va="bottom",
                    )

        g.map_dataframe(_plot_lines)

        if hue is not None and legend:
            g.add_legend()

    else:
        # time-colored segment mode
        if hue is not None:
            print("Note: color_time=True, so `hue` is ignored.")

        if time_var not in df.columns:
            raise ValueError(f"time_var='{time_var}' must appear in dataframe columns")

        tvals = df[time_var].to_numpy()
        tvals = tvals[np.isfinite(tvals)]
        if len(tvals) == 0:
            raise ValueError(f"No finite values found in time_var='{time_var}'")

        norm = Normalize(vmin=np.nanmin(tvals), vmax=np.nanmax(tvals))
        cmap = plt.get_cmap(time_palette)

        def _plot_time_colored(data, **kws):
            ax = plt.gca()

            for traj_id, sub in data.groupby("_traj_id", sort=False):
                sub = sub.sort_values(time_var)

                xvals = sub["_x"].to_numpy()
                yvals = sub["_y"].to_numpy()
                t = sub[time_var].to_numpy()

                good = np.isfinite(xvals) & np.isfinite(yvals) & np.isfinite(t)
                xvals = xvals[good]
                yvals = yvals[good]
                t = t[good]

                if len(xvals) < 2:
                    if label_tracks and len(xvals) == 1:
                        t0 = t[0]
                        label_color = cmap(norm(t0))
                        ax.text(
                            xvals[0] + label_offset,
                            yvals[0] + label_offset,
                            _track_label_from_traj_id(traj_id),
                            fontsize=label_fontsize,
                            color=label_color,
                            alpha=min(1.0, alpha + 0.2),
                            ha="left",
                            va="bottom",
                        )
                    continue

                pts = np.column_stack([xvals, yvals]).reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

                lc = LineCollection(
                    segs,
                    cmap=cmap,
                    norm=norm,
                    linewidths=linewidth,
                    alpha=alpha,
                )
                lc.set_array(t[:-1])
                ax.add_collection(lc)

                if label_tracks:
                    label_color = cmap(norm(t[0]))
                    ax.text(
                        xvals[0] + label_offset,
                        yvals[0] + label_offset,
                        _track_label_from_traj_id(traj_id),
                        fontsize=label_fontsize,
                        color=label_color,
                        alpha=min(1.0, alpha + 0.2),
                        ha="left",
                        va="bottom",
                    )

            ax.autoscale_view()

        g.map_dataframe(_plot_time_colored)

        if add_colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            g.figure.subplots_adjust(right=0.75)
            cax = g.figure.add_axes([0.95, 0.2, 0.02, 0.6])

            cbar = g.figure.colorbar(sm, cax=cax)
            cbar.set_label(time_var)

    # ---- axis labels ----
    xlabel = "x"
    ylabel = "y"
    if center_tracks:
        xlabel = "Δx"
        ylabel = "Δy"
    if units is not None:
        xlabel += f" ({units})"
        ylabel += f" ({units})"

    g.set_axis_labels(xlabel, ylabel)

    # ---- equal aspect ----
    for ax in g.axes.flat:
        if ax is not None:
            ax.set_aspect("equal")

    # ---- title ----
    title = "XY trajectories"
    if center_tracks:
        title = "Centered XY trajectories"
    if color_time:
        title += f" colored by {time_var}"
    if space == "pixels":
        title += " (pixels)"

    g.figure.suptitle(title, y=1.02)
    g.figure.tight_layout()

    return g