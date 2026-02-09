"""
Plotting utilities for croparray.

Conventions:
- All functions here are pure (no CropArray mutation).
- Dataset-aware functions accept `ds` explicitly.
- Generic helpers accept arrays/images and are auto-exposed via CropArrayPlot.
"""
import numpy as np
import xarray as xr

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



# # Make various montages of crop array for easy viewing in napari
# def montage(ds: xr.Dataset, *, col: str = "t", row: str = "n", **kwargs) -> xr.Dataset:
#     """
#     Returns a montage of a crop array for easier visualization.

#     Parameters
#     ----------
#     ds : xr.Dataset
#         A crop array dataset.
#     col : str, optional
#         Coordinate to arrange in columns, typically 'fov', 'n', or 't' (default 't').
#     row : str, optional
#         Coordinate to arrange in rows, typically 'fov', 'n' (default 'n'), or 't'.

#     Returns
#     -------
#     xr.Dataset
#         A reshaped crop array in which individual crops are arranged in a 2D array
#         of dimensions row x col. If row and col are the same, a square montage is returned.
#     """
#     # Backward-compat: allow callers to pass col/row via kwargs
#     col = kwargs.pop("col", col)
#     row = kwargs.pop("row", row)

#     if row != col:  # arrange crops in rows and columns in the xy plane
#         output = ds.stack(r=(row, "y"), c=(col, "x")).transpose(
#             "cell", "rep", "exp", "tracks", "fov", "n", "t", "z", "r", "c", "ch",
#             missing_dims="ignore",
#         )

#     else:  # row == col: arrange crops in square montage in the xy-plane
#         col_length = len(ds.coords.get(col))
#         my_sqrt = np.sqrt(col_length)
#         remainder = my_sqrt % 1

#         if remainder == 0:
#             my_size = int(my_sqrt)  # montage square will be (my_size x my_size)
#         else:
#             my_size = int(np.floor(my_sqrt) + 1)  # montage square will be (my_size x my_size)

#         # pad w/ 0 so there are enough crops to fill a perfect my_size x my_size square
#         pad_amount = my_size * my_size - col_length

#         # Reshape dataset so 'col' coordinate is rearranged into a perfect square.
#         # See https://stackoverflow.com/questions/59504320/how-do-i-subdivide-refine-a-dimension-in-an-xarray-dataset/59685729#59685729
#         output = (
#             ds.pad(
#                 pad_width={col: (0, pad_amount)},
#                 mode="constant",
#                 constant_values=0,  # Careful: ds.pad behavior could change across xarray versions
#             )
#             .assign_coords(montage_row=np.arange(my_size), montage_col=np.arange(my_size))
#             .stack(montage=("montage_row", "montage_col"))  # stack into montage index
#             .reset_index(col, drop=True)  # remove 'col' coordinate, keep data
#             .rename({col: "montage"})  # rename 'col' dimension to 'montage'
#             .unstack("montage")  # unstack montage_row and montage_col
#             .stack(r=("montage_row", "y"), c=("montage_col", "x"))  # arrange in xy plane
#             .transpose(
#                 "cell", "rep", "exp", "tracks", "fov", "n", "t", "z", "r", "c", "ch",
#                 missing_dims="ignore",
#             )
#         )

#     return output





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
