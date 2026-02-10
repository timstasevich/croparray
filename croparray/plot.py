
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

__all__ = ["montage"]


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

        # Keep your preferred transpose order; allow missing dims
        out = out.transpose(
            "cell", "rep", "exp", "tracks", "fov", "n", "t", "z", "r", "c", "ch",
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
        .transpose(
            "cell", "rep", "exp", "tracks", "fov", "n", "t", "z", "r", "c", "ch",
            missing_dims="ignore",
        )
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


def show_rgb_large(img8, *, scale=1.0, title=None):
    """
    Display an RGB image at an appropriate physical size in matplotlib.

    Parameters
    ----------
    img8 : ndarray
        (Y, X, 3) uint8 image
    scale : float
        Multiplicative scale factor for display size (1.0 ≈ 1 pixel = 1/100 inch)
    """
    import matplotlib.pyplot as plt
    h, w = img8.shape[:2]
    dpi = 100

    fig = plt.figure(figsize=(w / dpi * scale, h / dpi * scale), dpi=dpi)
    plt.imshow(img8)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


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
