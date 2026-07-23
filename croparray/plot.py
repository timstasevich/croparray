
from __future__ import annotations
"""
Plotting utilities for croparray.

Conventions:
- All functions here are pure (no CropArray mutation).
- Dataset-aware functions accept `ds` explicitly.
- Generic helpers accept arrays/images and are auto-exposed via CropArrayPlot.
"""
import numpy as np
import xarray as xr
from typing import Any, Sequence
import pandas as pd

__all__ = ["montage", "plot_croparray_crops", "relplot", "displot", "catplot", "swarmplot"]

_EPS = 1e-9

def _as_range_list(v):
    """
    Convert scalar / sequence / (start, stop, step) tuple into a list of ints.
    """
    if isinstance(v, tuple) and len(v) == 3:
        start, stop, step = v
        return list(range(start, stop, step))
    elif isinstance(v, (list, np.ndarray)):
        return list(map(int, v))
    else:
        return [int(v)]

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

def montage(ds: xr.Dataset, *, col: str = "t", row: str = "n", **kwargs) -> xr.Dataset:
    """
    Returns a montage of a crop array for easier visualization *and* robust manual
    tile-annotation in napari.

    Key guarantees (for downstream click-to-label tools)
    ----------------------------------------------------
    This function always produces:
      - integer tile axes:   montage_row, montage_col
      - pixel axes:          y, x
      - stacked pixel dims:  r=(montage_row, y), c=(montage_col, x)

    It also preserves the identity of each tile:
      - If row != col:
            tile_row_id[montage_row] = original coordinate values for `row`
            tile_col_id[montage_col] = original coordinate values for `col`
      - If row == col (square packing):
            tile_id[montage_row, montage_col] = original coordinate value for that tile
            (padded tiles get a fill value; see `pad_value`)

    Notes
    -----
    - This is designed so you can click a tile in napari and map it back to the
      underlying `row` / `col` identity, even for square montages where identity
      used to be dropped.
    - Backward-compat: callers may pass col/row via kwargs.

    Parameters
    ----------
    ds : xr.Dataset
        A crop array dataset.
    col : str, optional
        Coordinate/dimension to arrange in columns.
    row : str, optional
        Coordinate/dimension to arrange in rows.
    pad_value : scalar, optional (kwarg only)
        Value used to fill padded tiles (default: 0).

    Returns
    -------
    xr.Dataset
        A reshaped dataset arranged in a montage.
    """
    # Backward-compat: allow callers to pass col/row via kwargs
    col = kwargs.pop("col", col)
    row = kwargs.pop("row", row)
    pad_value = kwargs.pop("pad_value", 0)

    if row not in ds.dims:
        raise ValueError(f"`row` must be a dimension in ds. Got row={row!r}, ds.dims={ds.dims}")
    if col not in ds.dims:
        raise ValueError(f"`col` must be a dimension in ds. Got col={col!r}, ds.dims={ds.dims}")
    if "y" not in ds.dims or "x" not in ds.dims:
        raise ValueError(f"Expected ds to have dims 'y' and 'x'. Got ds.dims={ds.dims}")

    # -------------------------------------------------------------------------
    # Case 1: rectangular montage (row != col)
    # -------------------------------------------------------------------------
    if row != col:
        # Rename the tiling dims to canonical names
        out = ds.rename({row: "montage_row", col: "montage_col"})

        # Preserve tile identity as simple 1D coords
        out = out.assign_coords(
            tile_row_id=("montage_row", ds.coords[row].values if row in ds.coords else np.arange(ds.sizes[row])),
            tile_col_id=("montage_col", ds.coords[col].values if col in ds.coords else np.arange(ds.sizes[col])),
        )

        # Stack into pixel-plane montage dims
        out = out.stack(r=("montage_row", "y"), c=("montage_col", "x"))

        # Keep preferred order; prepend any unknown dims (e.g. 'rna', 'znf') first
        _known = {"cell", "rep", "exp", "tracks", "fov", "n", "t", "z", "r", "c", "ch"}
        _extra = [d for d in out.dims if d not in _known]
        out = out.transpose(
            *_extra, "cell", "rep", "exp", "tracks", "fov", "n", "t", "z", "r", "c", "ch",
            missing_dims="ignore",
        )
        return out

    # -------------------------------------------------------------------------
    # Case 2: square-packed montage (row == col)
    # -------------------------------------------------------------------------
    dim = row  # same as col
    n_tiles = ds.sizes[dim]

    # Determine square size
    my_size = int(np.ceil(np.sqrt(n_tiles)))
    pad_amount = my_size * my_size - n_tiles

    # Pad tiles along `dim` so we can reshape into a perfect square
    padded = ds.pad(
        pad_width={dim: (0, pad_amount)},
        mode="constant",
        constant_values=pad_value,
    )

    # Capture identity of each tile (including padded entries)
    if dim in padded.coords:
        tile_ids_1d = padded.coords[dim].values
        # If coordinate values are not unique or are objects, that’s fine; we store them verbatim.
    else:
        tile_ids_1d = np.arange(padded.sizes[dim])

    tile_ids_2d = tile_ids_1d.reshape(my_size, my_size)

    # Build canonical tile axes and preserve identity as a 2D coord
    out = (
        padded
        .assign_coords(montage_row=np.arange(my_size), montage_col=np.arange(my_size))
        .assign_coords(tile_id=(("montage_row", "montage_col"), tile_ids_2d))
        # Create a linear montage index then unstack into montage_row/col
        .stack(montage=("montage_row", "montage_col"))
        # Drop the original `dim` index safely (identity is kept in tile_id)
        .reset_index(dim, drop=True)
        .rename({dim: "montage"})        # rename the padded dim axis to montage (if present)
        .unstack("montage")              # returns montage_row, montage_col as dims
        # Now stack into pixel-plane montage dims
        .stack(r=("montage_row", "y"), c=("montage_col", "x"))
    )
    _known = {"cell", "rep", "exp", "tracks", "fov", "n", "t", "z", "r", "c", "ch"}
    _extra = [d for d in out.dims if d not in _known]
    out = out.transpose(
        *_extra, "cell", "rep", "exp", "tracks", "fov", "n", "t", "z", "r", "c", "ch",
        missing_dims="ignore",
    )

    return out

def rescale_rgb_0_255(arr):
    """
    Rescale an image array to uint8 [0, 255] using global min/max.
    Works for (Y,X,3) or any array with last dim = channels.
    """
    import numpy as np
    arr = np.asarray(arr, dtype=float)

    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)

    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.uint8)

    out = (arr - vmin) / (vmax - vmin)
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return out

def _finish_figure(fig, *, save_path=None, transparent=False, **savefig_kwargs):
    """
    Either save `fig` to disk or display it, but never both.

    Notebook inline backends close a figure the moment `plt.show()` runs, so
    grabbing `plt.gcf()` afterward (e.g. to save it) yields a blank figure.
    Passing `save_path` here sidesteps that entirely by saving instead of showing.
    """
    import matplotlib.pyplot as plt

    if save_path is not None:
        if transparent:
            fig.patch.set_alpha(0)
            for ax in fig.axes:
                ax.set_facecolor("none")
        fig.savefig(save_path, transparent=transparent, bbox_inches="tight", **savefig_kwargs)
        plt.close(fig)
    else:
        plt.show()

def show_rgb_large(img8, *, scale=1.0, title=None, save_path=None, transparent=False):
    """
    Display an RGB image at an appropriate physical size in matplotlib.

    Parameters
    ----------
    img8 : ndarray
        (Y, X, 3) uint8 image
    scale : float
        Multiplicative scale factor for display size (1.0 ≈ 1 pixel = 1/100 inch)
    save_path : str or None
        If given, save the figure here instead of displaying it.
    transparent : bool
        If True (and save_path is given), save with a transparent background.
    """
    import matplotlib.pyplot as plt
    h, w = img8.shape[:2]
    dpi = 100

    fig = plt.figure(figsize=(w / dpi * scale, h / dpi * scale), dpi=dpi)
    plt.imshow(img8)
    plt.axis("off")
    if title:
        plt.title(title)
    _finish_figure(fig, save_path=save_path, transparent=transparent)

# --- Seaborn figure-level wrappers (relplot/displot/catplot) -----------------

def _extract_needed_cols(kwargs: dict[str, Any]) -> list[str]:
    """
    Pull out seaborn column references. Conservative (ok to include extras).
    """
    keys = ("x", "y", "hue", "row", "col", "style", "size", "units", "weights")
    cols: list[str] = []
    for k in keys:
        v = kwargs.get(k, None)
        if isinstance(v, str) and v:
            cols.append(v)
    # stable unique
    return list(dict.fromkeys(cols))


def _build_plot_df(ds, *, vars: Sequence[str] | None, query: str | None, dropna: bool, kwargs: dict[str, Any]) -> pd.DataFrame:
    # local import keeps plot.py lightweight
    from .dataframe import variables_to_df

    if vars is None:
        inferred = _extract_needed_cols(kwargs)
        if not inferred:
            raise ValueError("Could not infer vars. Provide vars=[...] or at least x=.../y=...")
        vars = tuple(inferred)

    df = variables_to_df(ds, list(vars))

    if query:
        df = df.query(query)

    if dropna:
        df = df.dropna(subset=list(vars), how="all")

    return df


def relplot(ds, /, *, vars: Sequence[str] | None = None, query: str | None = None, dropna: bool = True, **kwargs: Any):
    """
    Seaborn relplot with data auto-built from ds via variables_to_df.

    Supports seaborn-native facetting: row=, col=, col_wrap=, etc.
    """
    import seaborn as sns
    df = _build_plot_df(ds, vars=vars, query=query, dropna=dropna, kwargs=kwargs)
    return sns.relplot(data=df, **kwargs)

def displot(ds, /, *, vars: Sequence[str] | None = None, query: str | None = None, dropna: bool = True, **kwargs: Any):
    """
    Seaborn displot with data auto-built from ds via variables_to_df.
    """
    import seaborn as sns
    df = _build_plot_df(ds, vars=vars, query=query, dropna=dropna, kwargs=kwargs)
    return sns.displot(data=df, **kwargs)

def catplot(ds, /, *, vars: Sequence[str] | None = None, query: str | None = None, dropna: bool = True, **kwargs: Any):
    """
    Seaborn catplot with data auto-built from ds via variables_to_df.
    """
    import seaborn as sns
    df = _build_plot_df(ds, vars=vars, query=query, dropna=dropna, kwargs=kwargs)
    return sns.catplot(data=df, **kwargs)

def swarmplot(ds, /, *, vars: Sequence[str] | None = None, query: str | None = None, dropna: bool = True, **kwargs: Any):
    """
    Seaborn swarmplot with data auto-built from ds via variables_to_df.
    """
    import seaborn as sns
    df = _build_plot_df(ds, vars=vars, query=query, dropna=dropna, kwargs=kwargs)
    return sns.swarmplot(data=df, **kwargs)

def plot_croparray_crops(
    ds: xr.Dataset,
    *,
    layer: str = "best_z",
    n: Union[int, Sequence[int], np.ndarray, Tuple[int, int, int]] = (0, 1, 1),
    t: Union[int, Sequence[int], np.ndarray, Tuple[int, int, int]] = (0, 1, 1),
    col: str = "t",
    rolling: int = 1,
    quantile_range: Tuple[float, float] = (0.02, 0.99),
    # Display
    show_grayscale: bool = True,
    show_merge_chs: Optional[Tuple[int, int, int]] = None,
    ch: Optional[int] = None,
    # Presentation
    suppress_labels: bool = True,
    show_suptitle: bool = True,
) -> xr.DataArray:
    """
    Plot CropArray crops for selected `n` and `t`.

    Parameters
    ----------
    ds
        CropArray-like dataset containing image crops.
    layer
        Image layer to display, e.g. "best_z" or "int".
    n
        Crop indices to display. Can be:
          - scalar: 5
          - list: [0, 2, 4]
          - range tuple: (start, stop, step)
    t
        Time indices to display. Can be:
          - scalar: 5
          - list: [0, 2, 4]
          - range tuple: (start, stop, step)
    col
        Facet dimension for plotting. Must be "t" or "n".
    rolling
        Optional rolling mean along time before selecting `t`.
    quantile_range
        Quantiles used for display normalization.
    show_grayscale
        If True, show grayscale panels for each channel.
    show_merge_chs
        Optional mapping (r_src, g_src, b_src) using positional channel indices.
    ch
        If provided, show only this channel in grayscale and skip merge.
    suppress_labels
        If True, remove titles, axis labels, and ticks from facet plots.
    show_suptitle
        If True, add a suptitle above each plotted selection.

    Returns
    -------
    xr.DataArray
        The normalized DataArray used for plotting.
    """
    if layer not in ds:
        raise KeyError(f"Dataset must contain layer {layer!r}. Available: {list(ds.data_vars)}")

    if col not in ("t", "n"):
        raise ValueError(f"col must be 't' or 'n', got {col!r}")

    da = ds[layer]
    for req in ("t", "y", "x"):
        if req not in da.dims:
            raise ValueError(f"Layer {layer!r} must include dim {req!r}. Found dims: {da.dims}")

    n_list = _as_range_list(n)
    t_list = _as_range_list(t)

    if "n" not in da.dims:
        raise ValueError(f"Layer {layer!r} must include dim 'n'. Found dims: {da.dims}")

    bz = da.sel(n=n_list)

    if rolling and rolling > 1:
        bz = bz.rolling(t=rolling, center=True, min_periods=1).mean()

    bz = bz.isel(t=t_list)

    other_facet = "n" if col == "t" else "t"

    # keep dimensions in a predictable order
    desired_order = [d for d in ("n", "t", "y", "x", "ch") if d in bz.dims]
    bz = bz.transpose(*desired_order)

    # ---- No channel dimension ----
    if "ch" not in bz.dims:
        normed = _normalize_for_display(
            bz.transpose("n", "t", "y", "x"),
            quantile_range=quantile_range,
        )

        g = normed.plot.imshow(
            col=col,
            row=other_facet,
            cmap="gray",
            aspect=1,
            size=3,
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
                suptitle=f"{layer}",
            )

        return normed

    # ---- Single-channel override ----
    if ch is not None:
        normed = _normalize_for_display(
            bz.isel(ch=int(ch)).transpose("n", "t", "y", "x"),
            quantile_range=quantile_range,
        )

        g = normed.plot.imshow(
            col=col,
            row=other_facet,
            cmap="gray",
            aspect=1,
            size=3,
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
                suptitle=f"{layer} | ch={int(ch)}",
            )

        return normed

    # ---- Normalize each channel separately ----
    n_ch = int(bz.sizes["ch"])
    ch_normed = [
        _normalize_for_display(
            bz.isel(ch=i).transpose("n", "t", "y", "x"),
            quantile_range=quantile_range,
        )
        for i in range(n_ch)
    ]
    normed_all = xr.concat(ch_normed, dim="ch").assign_coords(ch=bz["ch"].values)

    # ---- Grayscale panels ----
    if show_grayscale:
        for i in range(n_ch):
            g = normed_all.isel(ch=i).plot.imshow(
                col=col,
                row=other_facet,
                cmap="gray",
                aspect=1,
                size=3,
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
                    suptitle=f"{layer} | ch={i}",
                )

    # ---- Optional RGB merge ----
    if show_merge_chs is not None:
        r_src, g_src, b_src = map(int, show_merge_chs)
        need_max = max(r_src, g_src, b_src)
        if n_ch <= need_max:
            raise ValueError(
                f"show_merge_chs={show_merge_chs} requires at least {need_max + 1} channels, "
                f"but dataset has {n_ch}."
            )

        r_da = normed_all.isel(ch=r_src).expand_dims(rgb=["R"])
        g_da = normed_all.isel(ch=g_src).expand_dims(rgb=["G"])
        b_da = normed_all.isel(ch=b_src).expand_dims(rgb=["B"])
        rgb_da = xr.concat([r_da, g_da, b_da], dim="rgb")

        g = rgb_da.plot.imshow(
            col=col,
            row=other_facet,
            rgb="rgb",
            aspect=1,
            size=3,
            vmin=0,
            vmax=1,
            add_labels=not suppress_labels,
            add_colorbar=False,
        )

        if show_suptitle:
            _facetgrid_cleanup(
                g,
                suppress_labels=suppress_labels,
                suptitle=f"{layer} | MERGE {tuple(show_merge_chs)}",
            )

    return normed_all