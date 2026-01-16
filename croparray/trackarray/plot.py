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

__all__ = ["plot_trackarray_crops", "plot_track_signal_traces"]


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

        # ---- No channel dimension ----
        if "ch" not in bz.dims:
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
            bz1 = bz.isel(ch=int(ch))
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
        ch_normed = [
            _normalize_for_display(bz.isel(ch=i), quantile_range=quantile_range)
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
    y2: Optional[int] = None,
    y2lim=None,
    y2_label: Optional[str] = None,
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
        If not None, place that channel index on a secondary (right) y-axis.
        Only applies to channelled variables.
    y2lim, y2_label
        Right-axis limits/label.
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

    sns.set_style("whitegrid")
    sns.set(font_scale=1.1)

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

        ax2 = ax.twinx() if (has_ch and y2 is not None) else None

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

        ax.set_title(f"Track {int(track_id)}")
        ax.set_xlabel("time (sec)")
        ax.set_ylabel(f"{var} (a.u.)")
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)

        if ax2 is not None:
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


# from __future__ import annotations

# import numpy as np
# import matplotlib.pyplot as plt
# import xarray as xr


# __all__ = ["plot_trackarray_crops","plot_track_signal_traces"]

# def plot_trackarray_crops(
#     ds,
#     *,
#     layer: str = "best_z",
#     fov=0,
#     track_id=1,
#     t=(0, 10, 3),
#     rolling=1,
#     quantile_range=(0.02, 0.99),
#     rgb_channels=(0, 1, 2),
#     ch=None,
#     suppress_labels=True,
#     show_suptitle=True,
# ):
#     """
#     Plot track-centered image crops across time and channels using xarray.plot.imshow.

#     Shows optional RGB composites and per-channel grayscale panels; returns normalized arrays.

#     Parameters
#     ----------
#     ds : xr.Dataset
#         Trackarray dataset.
#     layer : str, default "best_z"
#         Name of the DataArray in `ds` to plot (e.g., "best_z", "raw", "ch0_mask_manual").
#         Must be image-like with dims including (track_id, t, y, x) and optionally (fov, ch).
#     fov : int, default 0
#         Field of view to select, if applicable.
#     track_id : int or sequence[int], default 1
#         Track(s) to plot.
#     t : tuple[int, int, int], default (0, 10, 3)
#         (start, stop, step) along time axis.
#     rolling : int, default 1
#         Rolling mean window over time. If 1, no smoothing.
#     quantile_range : tuple[float, float], default (0.02, 0.99)
#         Quantiles used for per-channel normalization (positive pixels only).
#         Ignored for binary-like masks (0/1 or bool), which are displayed as black/white.
#     rgb_channels : tuple, default (0, 1, 2)
#         Channels to use for RGB composite when available.
#         Duplicates are allowed (e.g. (1, 1, 2)).
#     ch : int or None, default None
#         If provided, plot only this channel in grayscale.
#     suppress_labels : bool, default True
#         If True, removes per-crop titles, axis labels, and ticks.
#     show_suptitle : bool, default True
#         If True, adds a readable figure-level title per track.

#     Returns
#     -------
#     dict[int, xr.DataArray]
#         Mapping from track_id -> normalized DataArray used for plotting with dims (t, y, x, ch)
#         when ch exists. If the selected layer has no 'ch' dim, the returned arrays have
#         dims (t, y, x).
#     """
#     import numpy as np
#     import xarray as xr

#     if layer not in ds:
#         raise KeyError(f"Dataset must contain layer {layer!r}. Available: {list(ds.data_vars)}")

#     da = ds[layer]

#     # Minimal sanity check for image-like content
#     for req in ("t", "y", "x"):
#         if req not in da.dims:
#             raise ValueError(f"Layer {layer!r} must have dim {req!r}. Found dims: {da.dims}")

#     def _decorate_facetgrid(g, suptitle=None):
#         if suppress_labels:
#             try:
#                 g.set_titles("")
#             except Exception:
#                 try:
#                     g.set_titles(template="")
#                 except Exception:
#                     pass
#             try:
#                 g.set_xlabels("")
#                 g.set_ylabels("")
#             except Exception:
#                 pass

#         if show_suptitle and suptitle and hasattr(g, "fig") and g.fig is not None:
#             try:
#                 g.fig.suptitle(suptitle, y=1.02)
#             except Exception:
#                 pass
#         return g

#     eps = 1e-6

#     def _is_binary_like(a: xr.DataArray, max_check: int = 1_000_000) -> bool:
#         """Heuristic: True if values look like a binary mask (bool or subset of {0,1,255})."""
#         if a.dtype == bool:
#             return True

#         arr = np.asarray(a.data) if hasattr(a, "data") else np.asarray(a.values)
#         flat = arr.ravel()
#         if flat.size == 0:
#             return False

#         if flat.size > max_check:
#             step = int(np.ceil(flat.size / max_check))
#             flat = flat[::step]

#         if np.issubdtype(flat.dtype, np.floating):
#             flat = flat[~np.isnan(flat)]
#             if flat.size == 0:
#                 return False

#         u = np.unique(flat)
#         return set(u.tolist()).issubset({0, 1, 255})

#     def _normalize_for_display(a: xr.DataArray):
#         """
#         Returns (normed, vmin, vmax). For binary-like arrays, return black/white mask.
#         Otherwise, quantile-normalize using positive pixels only.
#         """
#         if _is_binary_like(a):
#             out = a.astype(float)
#             out = xr.where(out == 255, 1.0, out).clip(0, 1)
#             return out, 0.0, 1.0

#         da_pos = a.where(lambda x: x > 0)
#         if da_pos.count() == 0:
#             q0, q1 = 0.0, 1.0
#         else:
#             q0 = da_pos.quantile(quantile_range[0])
#             q1 = da_pos.quantile(quantile_range[1])
#         out = ((a - q0) / (q1 - q0 + eps)).clip(0, 1)
#         return out, 0.0, 1.0

#     track_ids = list(track_id) if isinstance(track_id, (list, tuple, np.ndarray)) else [track_id]
#     results = {}

#     for tid in track_ids:
#         bz = da.sel(track_id=tid) if "track_id" in da.dims else da

#         # --- FOV selection ---
#         if "fov" in bz.dims:
#             bz = bz.sel(fov=fov)
#         elif "fov" in ds.coords:
#             try:
#                 bz = bz.where(ds["fov"] == fov, drop=True)
#             except Exception:
#                 pass

#         # --- Time slicing ---
#         start, stop, step = t
#         bz = bz.isel(t=slice(start, stop, step))

#         # --- Rolling average ---
#         if rolling and rolling > 1:
#             bz = bz.rolling(t=rolling, center=True, min_periods=1).mean()

#         # If the layer has no channel dimension, just plot it as grayscale
#         if "ch" not in bz.dims:
#             normed, vmin, vmax = _normalize_for_display(bz)

#             g = normed.plot.imshow(
#                 col="t",
#                 cmap="gray",
#                 xticks=[] if suppress_labels else None,
#                 yticks=[] if suppress_labels else None,
#                 aspect=1,
#                 size=5,
#                 vmin=vmin,
#                 vmax=vmax,
#                 robust=False,
#                 add_labels=not suppress_labels,
#                 add_colorbar=False,
#             )
#             _decorate_facetgrid(g, suptitle=f"track_id={tid} | {layer}")
#             results[int(tid)] = normed
#             continue

#         # --- Single-channel mode ---
#         if ch is not None:
#             try:
#                 bz1 = bz.sel(ch=ch)
#             except Exception:
#                 bz1 = bz.isel(ch=int(ch))

#             bz1 = bz1.expand_dims("ch").assign_coords(ch=[ch])

#             normed0, vmin, vmax = _normalize_for_display(bz1.isel(ch=0))
#             normed = normed0.expand_dims(ch=[ch])

#             g = normed.isel(ch=0).plot.imshow(
#                 col="t",
#                 cmap="gray",
#                 xticks=[] if suppress_labels else None,
#                 yticks=[] if suppress_labels else None,
#                 aspect=1,
#                 size=5,
#                 vmin=vmin,
#                 vmax=vmax,
#                 robust=False,
#                 add_labels=not suppress_labels,
#                 add_colorbar=False,
#             )
#             _decorate_facetgrid(g, suptitle=f"track_id={tid} | {layer} | ch={ch}")
#             results[int(tid)] = normed
#             continue

#         # --- Per-channel normalization ---
#         ch_normed = []
#         for ch_val in bz["ch"].values:
#             n, _, _ = _normalize_for_display(bz.sel(ch=ch_val))
#             ch_normed.append(n)

#         normed = xr.concat(ch_normed, dim="ch").assign_coords(ch=bz["ch"].values)

#         # --- RGB composite ---
#         if normed.sizes.get("ch", 0) >= 3:
#             try:
#                 rgb_da = normed.sel(ch=list(rgb_channels))
#             except Exception:
#                 rgb_da = normed.isel(ch=slice(0, 3))

#             if rgb_da.sizes.get("ch", 0) == 3:
#                 g = rgb_da.plot.imshow(
#                     col="t",
#                     rgb="ch",
#                     xticks=[] if suppress_labels else None,
#                     yticks=[] if suppress_labels else None,
#                     aspect=1,
#                     size=5,
#                     vmin=0,
#                     vmax=1,
#                     add_labels=not suppress_labels,
#                     add_colorbar=False,
#                 )
#                 _decorate_facetgrid(g, suptitle=f"track_id={tid} | {layer} (RGB)")

#         # --- Grayscale rows ---
#         for ch_val in normed["ch"].values:
#             g = normed.sel(ch=ch_val).plot.imshow(
#                 col="t",
#                 cmap="gray",
#                 xticks=[] if suppress_labels else None,
#                 yticks=[] if suppress_labels else None,
#                 aspect=1,
#                 size=5,
#                 vmin=0,
#                 vmax=1,
#                 robust=False,
#                 add_labels=not suppress_labels,
#                 add_colorbar=False,
#             )
#             _decorate_facetgrid(g, suptitle=f"track_id={tid} | {layer} | ch={ch_val}")

#         results[int(tid)] = normed

#     return results



# def plot_track_signal_traces(
#     ta_dataset,
#     track_ids,
#     var: str = "signal",
#     rgb=(1, 1, 1),
#     colors=("#00f670", "#f67000", "#7000f6"),
#     markers=("o", "s", "D"),
#     marker_size=6,
#     scatter_size=25,
#     markevery=5,
#     figsize=(7, 2.8),
#     ylim=None,
#     xlim=None,
#     col_wrap=3,
#     y2=None,
#     y2lim=None,
#     y2_label=None,
#     legend_loc="upper right",
#     show_legend=True,
# ):
#     """
#     Plot per-track traces for a chosen variable (default: 'signal') in a subplot grid.
#     Optionally place one channel on a secondary (right) y-axis.

#     Works for both channelled variables (dims include 'ch') and channel-less variables
#     (dims do not include 'ch', or dataframe has no usable 'ch' column).

#     Parameters
#     ----------
#     ta_dataset : xarray.Dataset
#         TrackArray dataset containing `var` with dims including (track_id, t) and optionally (ch).
#     track_ids : list[int]
#         Track IDs to plot.
#     var : str, default "signal"
#         Variable name to plot (e.g. "signal", "signal_raw", "ID", "MEAN_INTENSITY").
#     rgb : tuple[int,int,int] or None, default (1,1,1)
#         Channel inclusion mask for left axis (unless a channel is assigned to y2).
#         If None, plot all channels. Ignored for channel-less variables.
#     colors, markers, marker_size, scatter_size, markevery, figsize, ylim, xlim, col_wrap, y2, y2lim, y2_label,
#     legend_loc, show_legend : as before.

#     Notes
#     -----
#     - For channel-less variables, the trace is plotted once per track (no channel loop),
#       using colors[0]/markers[0], and y2 is ignored.
#     """
#     from ..dataframe import variables_to_df
#     import math
#     import seaborn as sns
#     import matplotlib.pyplot as plt

#     if var not in ta_dataset:
#         raise KeyError(f"Dataset does not contain variable '{var}'")

#     df = variables_to_df(ta_dataset, [var])

#     # Determine whether this variable is actually channelled in the dataframe.
#     has_ch = ("ch" in df.columns) and df["ch"].notna().any()

#     sns.set_style("whitegrid")
#     sns.set(font_scale=1.1)

#     n = len(track_ids)
#     ncols = col_wrap
#     nrows = math.ceil(n / col_wrap)

#     fig, axes = plt.subplots(
#         nrows=nrows,
#         ncols=ncols,
#         figsize=(figsize[0] * ncols, figsize[1] * nrows),
#         squeeze=False,
#     )

#     # Ensure markers covers all channels
#     if len(markers) < len(colors):
#         markers = list(markers) + ["o"] * (len(colors) - len(markers))

#     def _rgb_on(ch: int) -> bool:
#         if rgb is None:
#             return True
#         if ch < len(rgb):
#             return bool(rgb[ch])
#         return False

#     for idx, track_id in enumerate(track_ids):
#         row, col = divmod(idx, col_wrap)
#         ax = axes[row][col]

#         # Only meaningful for channelled vars
#         ax2 = ax.twinx() if (has_ch and y2 is not None) else None

#         if not has_ch:
#             # ---- Channel-less variable: plot a single trace ----
#             subset = df[df["track_id"] == track_id]
#             if not subset.empty:
#                 color0 = colors[0] if len(colors) else None
#                 marker0 = markers[0] if len(markers) else "o"

#                 sns.lineplot(
#                     data=subset,
#                     x="t",
#                     y=var,
#                     ax=ax,
#                     color=color0,
#                     lw=2,
#                     dashes=False,
#                     legend=False,
#                     marker=marker0,
#                     markersize=marker_size,
#                     markevery=markevery,
#                 )

#                 # Scatter of the (possibly already-unique) timepoints
#                 mean_df = subset.groupby("t")[var].mean().reset_index()
#                 sns.scatterplot(
#                     data=mean_df,
#                     x="t",
#                     y=var,
#                     ax=ax,
#                     color=color0,
#                     s=scatter_size,
#                     legend=False,
#                 )

#         else:
#             # ---- Channelled variable: loop channels as before ----
#             for ch in range(len(colors)):
#                 if not (_rgb_on(ch) or (y2 == ch)):
#                     continue

#                 color = colors[ch]
#                 marker = markers[ch]

#                 subset = df[(df["track_id"] == track_id) & (df["ch"] == ch)]
#                 if subset.empty:
#                     continue

#                 target_ax = ax2 if (ax2 is not None and ch == y2) else ax

#                 sns.lineplot(
#                     data=subset,
#                     x="t",
#                     y=var,
#                     ax=target_ax,
#                     color=color,
#                     label=f"ch {ch}",
#                     lw=2,
#                     dashes=False,
#                     legend=False,
#                     marker=marker,
#                     markersize=marker_size,
#                     markevery=markevery,
#                 )

#                 mean_df = subset.groupby("t")[var].mean().reset_index()
#                 sns.scatterplot(
#                     data=mean_df,
#                     x="t",
#                     y=var,
#                     ax=target_ax,
#                     color=color,
#                     s=scatter_size,
#                     legend=False,
#                 )

#         ax.set_title(f"Track {int(track_id)}")
#         ax.set_xlabel("time (sec)")
#         ax.set_ylabel(f"{var} (a.u.)")
#         if ylim is not None:
#             ax.set_ylim(ylim)
#         if xlim is not None:
#             ax.set_xlim(xlim)

#         if ax2 is not None:
#             right_color = colors[y2 % len(colors)]
#             ax2.set_ylabel(y2_label or f"{var} (a.u.) [ch {y2}]", color=right_color)
#             if y2lim is not None:
#                 ax2.set_ylim(y2lim)
#             if xlim is not None:
#                 ax2.set_xlim(xlim)
#             ax2.tick_params(axis="y", colors=right_color)
#             ax2.spines["right"].set_color(right_color)

#         if show_legend and has_ch:
#             h1, l1 = ax.get_legend_handles_labels()
#             h2, l2 = (ax2.get_legend_handles_labels() if ax2 else ([], []))
#             handles, labels = h1 + h2, l1 + l2

#             if legend_loc == "outside":
#                 ax.legend(
#                     handles,
#                     labels,
#                     loc="upper left",
#                     bbox_to_anchor=(1.15, 1.0),
#                     borderaxespad=0.0,
#                     frameon=True,
#                 )
#             else:
#                 ax.legend(handles, labels, loc=legend_loc, frameon=True)
#         else:
#             if ax.get_legend():
#                 ax.get_legend().remove()

#     # Hide unused subplots
#     for j in range(n, nrows * ncols):
#         r, c = divmod(j, col_wrap)
#         fig.delaxes(axes[r][c])

#     if legend_loc == "outside":
#         fig.subplots_adjust(right=0.82)

#     plt.tight_layout()
#     plt.show()

