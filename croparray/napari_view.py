from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import xarray as xr
import napari

__all__ = ["montage_viewer","manual_filter_montage"]


# @dataclass(frozen=True)
# class ContrastSpec:
#     lo_pct: float = 2.0
#     hi_pct: float = 98.0

def _normalize_image_contrast(image_contrast):
    """
    Normalize image_contrast to (lo, hi_percentile).
    - None        -> (0, 99.8)
    - number      -> (0, number)
    - (lo, hi)    -> (lo, hi)
    """
    if image_contrast is None:
        return 0.0, 99.8

    if isinstance(image_contrast, (int, float)):
        return 0.0, float(image_contrast)

    if isinstance(image_contrast, (tuple, list)) and len(image_contrast) == 2:
        return float(image_contrast[0]), float(image_contrast[1])

    raise ValueError(
        "image_contrast must be None, a number (hi percentile), "
        "or a (lo, hi) tuple"
    )

def _robust_limits_nonneg(a: np.ndarray, hi_pct: float, *, lo_fixed: float = 0.0) -> tuple[float, float]:
    """Upper percentile limit ignoring NaNs; lower fixed (default 0)."""
    a = np.asarray(a)
    if a.size == 0:
        return (lo_fixed, lo_fixed + 1.0)

    a = a[np.isfinite(a)]
    if a.size == 0:
        return (lo_fixed, lo_fixed + 1.0)

    # Ignore negatives when setting the upper bound (optional but recommended here)
    a_pos = a[a > lo_fixed]
    if a_pos.size == 0:
        return (lo_fixed, lo_fixed + 1.0)

    hi_v = float(np.percentile(a_pos, hi_pct))
    if hi_v <= lo_fixed:
        hi_v = lo_fixed + 1e-6
    return (lo_fixed, hi_v)

def _is_mask_like(da: xr.DataArray, *, name_hint: str | None = None) -> bool:
    """Heuristic: treat as labels/mask if boolean or low-cardinality integer."""
    name = (name_hint or da.name or "").lower()

    if da.dtype == bool:
        return True

    if np.issubdtype(da.dtype, np.integer):
        # Low-cardinality check (sample)
        data = da.data
        try:
            # sample a bit to avoid loading huge arrays
            flat = np.asarray(data).ravel()
            if flat.size > 200_000:
                flat = flat[:: max(1, flat.size // 200_000)]
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                return False
            uniq = np.unique(flat)
            if uniq.size <= 16:
                return True
        except Exception:
            pass

    if "mask" in name or "label" in name or name.endswith("_mask"):
        return True

    return False

def _select_first_if_present(da: xr.DataArray, dim: str, idx: int = 0) -> xr.DataArray:
    return da.isel({dim: idx}) if dim in da.dims else da

def _squeeze_safe(da: xr.DataArray) -> xr.DataArray:
    return da.squeeze(drop=True)

def _infer_tile_hw_from_montage_image(m_or_ref: xr.Dataset | xr.DataArray) -> tuple[int, int]:
    """
    Robustly infer per-tile (height, width) from montage object with dims r,c.
    Works even when r/c indexes are not MultiIndex.
    """
    obj = m_or_ref
    if "r" not in obj.sizes or "c" not in obj.sizes:
        raise ValueError("Expected montage object to have 'r' and 'c' dims.")

    # If montage_row/col are explicit dims (square montage unstack kept them), use them
    n_mr = int(getattr(obj, "sizes", {}).get("montage_row", 0) or 0)
    n_mc = int(getattr(obj, "sizes", {}).get("montage_col", 0) or 0)

    # If not dims, they should still exist as coords on r/c (because r=(montage_row,y))
    if n_mr == 0:
        if hasattr(obj, "coords") and "montage_row" in obj.coords:
            n_mr = len(np.unique(np.asarray(obj.coords["montage_row"].values)))
        else:
            raise ValueError("Cannot infer montage_row count; missing montage_row.")
    if n_mc == 0:
        if hasattr(obj, "coords") and "montage_col" in obj.coords:
            n_mc = len(np.unique(np.asarray(obj.coords["montage_col"].values)))
        else:
            raise ValueError("Cannot infer montage_col count; missing montage_col.")

    tile_h = int(obj.sizes["r"] // max(n_mr, 1))
    tile_w = int(obj.sizes["c"] // max(n_mc, 1))

    return tile_h, tile_w

def _tile_values_to_pixel_image(tile_vals: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
    """
    Expand tile-wise values to pixel-wise montage.
    tile_vals: shape (T, MR, MC) or (MR, MC)
    returns:   shape (T, R, C) or (R, C)
    """
    tile_vals = np.asarray(tile_vals)
    if tile_vals.ndim == 2:
        # (MR, MC) -> (R, C)
        return np.repeat(np.repeat(tile_vals, tile_h, axis=0), tile_w, axis=1)
    if tile_vals.ndim == 3:
        # (T, MR, MC) -> (T, R, C)
        out = np.repeat(np.repeat(tile_vals, tile_h, axis=1), tile_w, axis=2)
        return out
    raise ValueError(f"tile_vals must be 2D or 3D, got shape {tile_vals.shape}")

def _get_tile_ids_2d(m: xr.Dataset, *, row: str, col: str) -> xr.DataArray:
    """
    Returns a DataArray tile_id_2d with dims (montage_row, montage_col) that gives:
      - for square montages (row==col): uses m.coords['tile_id']
      - for rectangular montages: builds outer product from tile_row_id and tile_col_id
        (mainly useful for debugging; labeling will use both 1D ids).
    """
    if row == col:
        if "tile_id" not in m.coords:
            raise ValueError("Square montage expected coord 'tile_id' (montage_row,montage_col).")
        return m.coords["tile_id"]

    # Rectangular montage: expect 1D ids
    if "tile_row_id" not in m.coords or "tile_col_id" not in m.coords:
        raise ValueError("Rectangular montage expected coords 'tile_row_id' and 'tile_col_id'.")
    # tile_row_id is 1D over montage_row, tile_col_id is 1D over montage_col
    tr = m.coords["tile_row_id"]
    tc = m.coords["tile_col_id"]
    # Make a 2D array of tuples (row_id, col_id) just for completeness/debug.
    # Downstream overlay logic uses them separately.
    rr, cc = xr.broadcast(tr, tc)
    return xr.concat([rr, cc], dim="tile_pair").transpose("montage_row", "montage_col", "tile_pair")

def _select_channels(da: xr.DataArray, ch):
    if "ch" not in da.dims:
        return da, None  # no channel dim

    if isinstance(ch, (list, tuple)):
        da = da.sel(ch=list(ch))
        if len(ch) == 3:
            return da, "rgb"
        else:
            return da, "multi"
    else:
        return da.sel(ch=ch), None

def montage_viewer(
    ca_or_ta: xr.Dataset,
    *,
    row: str,
    col: str,
    show: Iterable[str] = ("best_z", "ch0_mask"),
    ch: int | list[int] | tuple[int, ...] = 0,
    z_index: int = 0,
    viewer: napari.Viewer | None = None,
    image_contrast = None,
    tile_overlay_contrast = None,
    tile_overlay_opacity: float = 0.35,
    default_blending: str = "additive",
    colormaps: dict[str, str] | None = None,
) -> tuple[napari.Viewer, dict[str, Any]]:
    """
    Create a napari viewer for a montage with smart defaults:
      - mask-like layers => add_labels
      - image-like layers (has r,c) => add_image with robust contrast
      - non-xy layers (e.g., (track_id,t) or (n,t)) => drawn as per-tile overlays

    Parameters
    ----------
    ca_or_ta : xr.Dataset
        The CropArray/TrackArray dataset.
    row, col : str
        The montage tiling dims you want (e.g. row="track_id", col="track_id"; row="n", col="t").
    show : iterable[str]
        Variable names from the montage dataset to display.
    ch : int
        Channel to show for image-like layers that have a 'ch' dimension.
    z_index : int
        z-slice to show if z is present.
    viewer : napari.Viewer, optional
        If provided, add layers into this viewer; otherwise create a new one.
    image_contrast : 
        Percentiles for image layers (default 0–99.5). 
    tile_overlay_contrast : 
        Percentiles for tile overlays (default 5–95).
    tile_overlay_opacity : float
        Opacity used for tile overlays and labels.
    default_blending : str
        Napari blending mode for image layers.
    colormaps : dict[str,str], optional
        Colormap per layer name; falls back to 'gray' for images and 'magenta' for overlays.

    Returns
    -------
    viewer, layers_dict
        layers_dict maps requested layer name -> napari layer object.
    """
    from .plot import montage
    if image_contrast is None:
        image_contrast = (0.0, 99.5)
    if tile_overlay_contrast is None:
        tile_overlay_contrast = (5.0, 95.0)
    colormaps = colormaps or {}

    # Build montage once
    m = montage(ca_or_ta, row=row, col=col)  # uses your improved montage()

    if viewer is None:
        viewer = napari.Viewer()

    layers: dict[str, Any] = {}

    # Choose a representative "reference image" to infer tile size and to host overlays.
    # Prefer best_z if present, else any var with r/c.
    ref_name = None
    for nm in ("best_z", "int"):
        if nm in m.data_vars:
            da0 = _squeeze_safe(m[nm])
            if "r" in da0.dims and "c" in da0.dims:
                ref_name = nm
                break
    if ref_name is None:
        # find any var with r/c
        for nm in list(m.data_vars):
            da0 = _squeeze_safe(m[nm])
            if "r" in da0.dims and "c" in da0.dims:
                ref_name = nm
                break
    if ref_name is None:
        raise ValueError("Could not find any montage variable with dims including ('r','c') to anchor the view.")

    ref = _squeeze_safe(m[ref_name])
    ref = _select_first_if_present(ref, "z", z_index)
    ref = _select_first_if_present(ref, "ch", ch)
    # Ensure (t,r,c) ordering if t exists
    # Ensure (..., t, r, c) ordering if t exists; preserve extra dims (exp/cell/rep/...) as leading sliders
    if "t" in ref.dims:
        ref = ref.transpose(..., "t", "r", "c", missing_dims="ignore")
    else:
        ref = ref.transpose(..., "r", "c", missing_dims="ignore")


    tile_h, tile_w = _infer_tile_hw_from_montage_image(m)

    # Helper: get a 3D (t, montage_row, montage_col) grid of tile ids for square montage,
    # or 1D ids for rectangular montage.
    square = (row == col)

    # For non-xy overlays, we need to generate an image-shaped overlay aligned to ref (t,r,c).
    # Square montage: use tile_id[montage_row,montage_col] to map track_id/n -> tiles.
    # Rect montage: use tile_row_id[montage_row] and tile_col_id[montage_col] separately.
    tile_id_2d = None
    tile_row_id_1d = None
    tile_col_id_1d = None
    if square:
        tile_id_2d = m.coords["tile_id"]
    else:
        tile_row_id_1d = m.coords.get("tile_row_id", None)
        tile_col_id_1d = m.coords.get("tile_col_id", None)

    def _add_image_layer(name: str, da: xr.DataArray) -> None:
        da = _squeeze_safe(da)
        da = _select_first_if_present(da, "z", z_index)

        da, ch_mode = _select_channels(da, ch)

        # Preserve extra dims as leading sliders
        if "t" in da.dims:
            da = da.transpose(..., "t", "r", "c", "ch", missing_dims="ignore")
        else:
            da = da.transpose(..., "r", "c", "ch", missing_dims="ignore")


        data = np.asarray(da.data)

        if ch_mode == "rgb":
            # napari expects (..., r, c, 3)
            # so we move ch to last axis
            if "t" in da.dims:
                data = np.moveaxis(data, -1, -1)  # already last
            else:
                data = np.moveaxis(data, -1, -1)

            clim = None  # RGB ignores contrast_limits
            lyr = viewer.add_image(
                data,
                name=name,
                rgb=True,
                blending=default_blending,
            )

        else:
            print("DEBUG image_contrast:", type(image_contrast), image_contrast)
            lo, hi = _normalize_image_contrast(image_contrast)
            clim = _robust_limits_nonneg(
                np.asarray(da.data),
                hi,
                lo_fixed=lo,
            )
            lyr = viewer.add_image(
                data,
                name=name,
                colormap=colormaps.get(name, "gray"),
                blending=default_blending,
                contrast_limits=list(clim),
            )

        layers[name] = lyr

    def _add_labels_layer(name: str, da: xr.DataArray) -> None:
        from napari.utils.colormaps import DirectLabelColormap
        from napari.utils.colormaps.standardize_color import transform_color

        da = _squeeze_safe(da)
        da = _select_first_if_present(da, "z", z_index)
        da = _select_first_if_present(da, "ch", ch)

        # Preserve extra dims as leading sliders
        if "t" in da.dims:
            da = da.transpose(..., "t", "r", "c", missing_dims="ignore")
        else:
            da = da.transpose(..., "r", "c", missing_dims="ignore")

        lbl = (da > 0).astype(np.uint8).data
        lyr = viewer.add_labels(
            lbl,
            name=name,
            opacity=tile_overlay_opacity,
        )

        # ---- Force label colors (avoid napari's default brown) ----
        on_color = colormaps.get(name, "magenta")  # e.g. colormaps["ch0_mask"] = "magenta"
        try:
            rgba = tuple(float(x) for x in transform_color([on_color])[0])  # (4,)
        except Exception:
            rgba = on_color  # some napari versions accept strings directly

        lyr.colormap = DirectLabelColormap(
            color_dict={
                0: (0.0, 0.0, 0.0, 0.0),  # transparent background
                1: rgba,                  # label=1 color
            }
        )

        layers[name] = lyr

    # def _add_labels_layer(name: str, da: xr.DataArray) -> None:
    #     da = _squeeze_safe(da)
    #     da = _select_first_if_present(da, "z", z_index)
    #     da = _select_first_if_present(da, "ch", ch)
    #     # Preserve extra dims as leading sliders
    #     if "t" in da.dims:
    #         da = da.transpose(..., "t", "r", "c", missing_dims="ignore")
    #     else:
    #         da = da.transpose(..., "r", "c", missing_dims="ignore")


    #     lbl = (da > 0).astype(np.uint8).data
    #     lyr = viewer.add_labels(
    #         lbl,
    #         name=name,
    #         opacity=tile_overlay_opacity,
    #     )
    #     layers[name] = lyr

    def _add_tile_overlay(name: str, da: xr.DataArray) -> None:
        """
        da is expected to be tile-like (no r/c) and indexable by (row, t) or (n,t), etc.
        We expand it to an image overlay of shape (t,r,c) and add as an image layer.
        """
        da = _squeeze_safe(da)

        # Normalize to include t if present; allow scalars per tile.
        has_t = ("t" in da.dims)

        if square:
            # Need da indexed by the same identity as tile_id_2d (e.g., track_id or n)
            # and optionally by t. We infer which dim in da corresponds to the tile identity:
            # - prefer 'track_id' if present, else 'n' if present, else 'tile'
            tile_dim = None
            for candidate in ("track_id", "n", row):
                if candidate in da.dims:
                    tile_dim = candidate
                    break
            if tile_dim is None:
                raise ValueError(
                    f"Tile overlay for square montage needs a tile identity dim (e.g. 'track_id' or 'n'). "
                    f"Got da.dims={da.dims}."
                )

            # Build per-tile grid by selecting along tile_dim using tile_id_2d
            # tile_id_2d dims: (montage_row, montage_col)
            if has_t:
                # da: (tile_dim, t, ...) -> select => (montage_row, montage_col, t)
                tv = da.transpose(tile_dim, "t", missing_dims="ignore").sel({tile_dim: tile_id_2d})
                # xarray returns dims (montage_row, montage_col, t); move to (t, montage_row, montage_col)
                tv = tv.transpose("t", "montage_row", "montage_col")
                tile_vals = tv.data  # (t, MR, MC)
                pix = _tile_values_to_pixel_image(tile_vals, tile_h, tile_w)  # (t, R, C)
            else:
                tv = da.sel({tile_dim: tile_id_2d})
                tv = tv.transpose("montage_row", "montage_col")
                pix = _tile_values_to_pixel_image(tv.data, tile_h, tile_w)  # (R, C)

        else:
            # Rectangular montage: row identity and col identity are separate 1D coords.
            if tile_row_id_1d is None or tile_col_id_1d is None:
                raise ValueError("Rectangular montage expected tile_row_id and tile_col_id coords.")

            # We support overlays keyed by (row, t) or (row, col) or (row, t, ...)
            # Most common for you: (track_id, t) or (n, t).
            row_dim = None
            for candidate in ("track_id", "n", row):
                if candidate in da.dims:
                    row_dim = candidate
                    break
            if row_dim is None:
                raise ValueError(
                    f"Tile overlay needs a row identity dim (e.g. 'track_id' or 'n'). Got da.dims={da.dims}."
                )

            # If the montage col is t, we can map directly. Otherwise, we still allow overlays that
            # are per-row only (broadcast across montage_col).
            col_is_t = (col == "t")

            if has_t:
                # da: (row_dim, t) -> select row -> (montage_row, t)
                row_sel = da.transpose(row_dim, "t").sel({row_dim: tile_row_id_1d})
                if col_is_t:
                    # Need tile grid (t, montage_row, montage_col), where montage_col indexes t
                    # tile_col_id_1d are the t values in montage columns.
                    t_vals = tile_col_id_1d
                    grid = row_sel.sel(t=t_vals)  # (montage_row, montage_col)
                    grid = grid.transpose("montage_col", "montage_row")  # (montage_col, montage_row)
                    # But we want (t, MR, MC). Here montage_col corresponds to t; rename in place:
                    # easiest: treat montage_col as t-axis order already.
                    tile_vals = np.asarray(grid.data).T  # -> (MR, MC) if single t; not right for time
                    # Better: if col==t, the view will typically be (t, r, c) with time as first axis.
                    # So we create a full (t, MR, MC) by indexing row_sel at each t and placing into MC.
                    # Implement directly:
                    t_list = np.asarray(t_vals.data)
                    mr = row_sel.sizes["montage_row"]
                    mc = len(t_list)
                    out_tv = np.full((len(t_list), mr, mc), np.nan, dtype=float)
                    for i, tt in enumerate(t_list):
                        out_tv[i, :, i] = np.asarray(row_sel.sel(t=tt).data)
                    pix = _tile_values_to_pixel_image(out_tv, tile_h, tile_w)
                else:
                    # Overlay per (row,t) but montage columns are not time: show current t only
                    # by rendering a (t, R, C) where each timepoint uses same per-row value across cols.
                    row_sel = row_sel  # (montage_row, t)
                    # Build tile values (t, MR, MC) by broadcasting across MC
                    t_len = row_sel.sizes["t"]
                    mr = row_sel.sizes["montage_row"]
                    mc = int(len(tile_col_id_1d))
                    tv = np.repeat(np.asarray(row_sel.transpose("t", "montage_row").data)[:, :, None], mc, axis=2)
                    pix = _tile_values_to_pixel_image(tv, tile_h, tile_w)
            else:
                # da: (row_dim,) -> select => (montage_row,)
                row_sel = da.sel({row_dim: tile_row_id_1d})
                mc = int(len(tile_col_id_1d))
                grid = np.repeat(np.asarray(row_sel.data)[:, None], mc, axis=1)  # (MR, MC)
                pix = _tile_values_to_pixel_image(grid, tile_h, tile_w)

        # Add overlay
        clim = _robust_limits(pix, tile_overlay_contrast.lo_pct, tile_overlay_contrast.hi_pct)
        lyr = viewer.add_image(
            pix,
            name=name,
            colormap=colormaps.get(name, "magenta"),
            blending="additive",
            opacity=tile_overlay_opacity,
            contrast_limits=list(clim),
        )
        layers[name] = lyr

    # Add requested layers
    for nm in show:
        if nm not in m.data_vars and nm not in m.coords:
            raise KeyError(f"Requested layer {nm!r} not found in montage dataset.")

        da = m[nm] if nm in m.data_vars else m.coords[nm]
        da0 = _squeeze_safe(da)

        if ("r" in da0.dims) and ("c" in da0.dims):
            # image-like or mask-like in montage pixel space
            if _is_mask_like(da0, name_hint=nm):
                _add_labels_layer(nm, da0)
            else:
                _add_image_layer(nm, da0)
        else:
            # non-xy layer => per-tile overlay
            _add_tile_overlay(nm, da0)

    return viewer, layers


#-------------------------------------------------------------------------
# Manual interactive filter labeling on montage tiles
# #-------------------------------------------------------------------------

def _default_save_path(ds: xr.Dataset, *, fallback: str | None = None) -> str | None:
    """
    Infer a default save path. SAFER default: never overwrite the source .nc;
    instead write a sidecar file with suffix '_manual.nc'.
    """
    src = None
    try:
        src = ds.encoding.get("source", None)
    except Exception:
        src = None

    if isinstance(src, str) and len(src) > 0:
        if src.endswith(".nc"):
            return src[:-3] + "_manual.nc"
        return src + "_manual"

    return fallback

def _add_manual_filter_save_widget(
    viewer: napari.Viewer,
    *,
    ds: xr.Dataset,
    filter_table: xr.DataArray,
    filter_name: str,
) -> None:
    """
    Save button that ONLY commits the current filter_table into the in-memory dataset:
        ds[filter_name] = filter_table
    No disk I/O.
    """
    try:
        from magicgui import magicgui
        from napari.utils.notifications import show_info
    except Exception:
        return

    @magicgui(call_button="Save (to my_ta only)", layout="vertical")
    def _save_widget():
        ds[filter_name] = filter_table
        show_info(f"Committed '{filter_name}' into the in-memory dataset (my_ta.ds).")

    viewer.window.add_dock_widget(_save_widget, name=f"{filter_name} commit", area="right")

def _infer_tile_dim(ds: xr.Dataset) -> str:
    if "track_id" in ds.dims:
        return "track_id"
    if "n" in ds.dims:
        return "n"
    raise ValueError("Expected dataset to have either 'track_id' or 'n' dimension.")

def _ensure_filter_table(
    ds: xr.Dataset, *, filter_name: str, tile_dim: str
) -> xr.DataArray:
    """
    Return a uint8 filter table with dims (tile_dim, t).
    If ds[filter_name] exists, it is validated and returned.
    Otherwise a new zero table is created (not inserted into ds here).
    """
    if "t" not in ds.dims:
        raise ValueError("manual_filter_montage expects a 't' dimension in the dataset.")

    if filter_name in ds:
        ft = ds[filter_name]
        if tile_dim not in ft.dims or "t" not in ft.dims:
            raise ValueError(
                f"Existing filter {filter_name!r} must include dims ({tile_dim!r}, 't'). "
                f"Got {ft.dims}."
            )
        return ft.astype(np.uint8)

    ft = xr.DataArray(
        np.zeros((ds.sizes[tile_dim], ds.sizes["t"]), dtype=np.uint8),
        dims=(tile_dim, "t"),
        coords={tile_dim: ds.coords[tile_dim].values, "t": ds.coords["t"].values},
        name=filter_name,
    )
    return ft

def _indexer_for_coord(da: xr.DataArray, dim: str, value) -> int:
    """Return integer index into da[dim] for coordinate value."""
    # Prefer pandas index if available
    try:
        idx = da.get_index(dim)
        return int(idx.get_loc(value))
    except Exception:
        # fall back to numpy search
        vals = np.asarray(da.coords[dim].values)
        hit = np.where(vals == value)[0]
        if hit.size == 0:
            raise KeyError(f"Value {value!r} not found in coordinate {dim!r}.")
        return int(hit[0])

def _tile_axis_1d_ids(
    m: xr.Dataset,
    *,
    coord_name: str,
    tile_dim: str,
    pixel_dim: str,
    step: int,
) -> np.ndarray:
    """
    Return 1D tile IDs for a montage axis.

    Prefer tile_dim (e.g., 'montage_row'/'montage_col'). If the coord is instead
    broadcast onto pixel_dim (e.g., 'r'/'c'), downsample by taking every `step`
    element (one per tile).
    """
    da = m.coords.get(coord_name, None)
    if da is None:
        raise ValueError(f"Rectangular montage expected coord {coord_name!r}.")

    # If it's already on montage tile dims, perfect.
    if tile_dim in da.dims:
        return np.asarray(da.values)

    # If it is broadcast on pixel dim only, downsample.
    if pixel_dim in da.dims:
        # If it's 2D (e.g. dims ('r','c')), reduce to 1D along the other axis.
        if da.ndim == 2:
            other = [d for d in da.dims if d != pixel_dim]
            if len(other) != 1:
                raise ValueError(f"Unexpected dims for {coord_name!r}: {da.dims}")
            da_1d = da.isel({other[0]: 0})
        else:
            da_1d = da

        vals = np.asarray(da_1d.values)
        return vals[::step]

    raise ValueError(f"Coord {coord_name!r} has unexpected dims {da.dims}.")

def _overlay_pixels_from_filter_table(
    *,
    m: xr.Dataset,
    filter_table: xr.DataArray,
    tile_dim: str,
    row: str,
    col: str,
    tile_h: int,
    tile_w: int,
) -> xr.DataArray:
    """
    Build a pixel-space overlay DataArray aligned to montage (r,c) and optionally t.
    Returned dims match the montage reference convention:
      - square montages: dims ('t','r','c')
      - row=tile_dim col='t' rectangular: dims ('r','c') (no 't' dim)
    """
    square = (row == col)

    # ----------------------------
    # Square case (UNCHANGED)
    # ----------------------------
    if square:
        tile_id_2d = m.coords["tile_id"]  # (montage_row, montage_col)
        ids = np.asarray(tile_id_2d.values)
        Mrow, Mcol = ids.shape

        # Compute per-tile pixel size from total montage size
        tile_h = int(m.sizes["r"] // max(Mrow, 1))
        tile_w = int(m.sizes["c"] // max(Mcol, 1))

        ft = filter_table.transpose(tile_dim, "t")  # (tile_dim, t)

        # Build a lookup from tile_dim value -> row index in ft
        tile_vals = np.asarray(ft.coords[tile_dim].values)
        tile_to_i = {v: i for i, v in enumerate(tile_vals)}

        ids = np.asarray(tile_id_2d.values)  # (Mrow, Mcol)
        Mrow, Mcol = ids.shape
        T = ft.sizes["t"]

        # Gather (t, Mrow, Mcol) values, filling missing/padded tiles with 0
        tv = np.zeros((T, Mrow, Mcol), dtype=np.uint8)
        ft_data = np.asarray(ft.data)  # (tile_dim, t)

        for mr in range(Mrow):
            for mc in range(Mcol):
                tid = ids[mr, mc]
                i = tile_to_i.get(tid, None)
                if i is None:
                    continue  # leave zeros for padded/unknown tiles
                tv[:, mr, mc] = ft_data[i, :]

        pix = _tile_values_to_pixel_image(tv, tile_h, tile_w)  # (t, R, C)
        assert pix.shape[1] == m.sizes["r"], (pix.shape, m.sizes["r"], tile_h, Mrow)
        assert pix.shape[2] == m.sizes["c"], (pix.shape, m.sizes["c"], tile_w, Mcol)

        out = xr.DataArray(
            pix.astype(np.uint8),
            dims=("t", "r", "c"),
            coords={"t": m.coords["t"].values, "r": m.coords["r"].values, "c": m.coords["c"].values},
            name=filter_table.name,
        )
        return out

    # ----------------------------
    # Rectangular case (FIXED)
    # ----------------------------
    # Always extract tile-level IDs even if coords are broadcast onto pixels.
    tile_row_id = _tile_axis_1d_ids(
        m,
        coord_name="tile_row_id",
        tile_dim="montage_row",
        pixel_dim="r",
        step=tile_h,
    )
    tile_col_id = _tile_axis_1d_ids(
        m,
        coord_name="tile_col_id",
        tile_dim="montage_col",
        pixel_dim="c",
        step=tile_w,
    )

    if col == "t":
        # Map each (montage_row, montage_col) -> (tile_dim, t) from row_id and col_id
        # This must be TILE-LEVEL (montage_row/montage_col), not pixel-level.
        row_sel = filter_table.sel({tile_dim: tile_row_id})   # -> (montage_row, t)
        grid = row_sel.sel(t=tile_col_id)                     # -> (montage_row, montage_col)

        pix = _tile_values_to_pixel_image(np.asarray(grid.data), tile_h, tile_w)  # -> (R, C)
        assert pix.shape[0] == m.sizes["r"], (pix.shape, m.sizes["r"], tile_h, len(tile_row_id))
        assert pix.shape[1] == m.sizes["c"], (pix.shape, m.sizes["c"], tile_w, len(tile_col_id))

        out = xr.DataArray(
            pix.astype(np.uint8),
            dims=("r", "c"),
            coords={"r": m.coords["r"].values, "c": m.coords["c"].values},
            name=filter_table.name,
        )
        return out

    raise NotImplementedError(
        "manual_filter_montage currently supports rectangular montages only when col == 't'."
    )


def _broadcast_overlay_like_ref(overlay: xr.DataArray, ref: xr.DataArray) -> np.ndarray:
    """
    Make overlay match ref's dimensionality by adding missing dims via expand_dims
    and ordering dims like ref. Avoid xarray coordinate alignment because montage
    r/c can be Index vs MultiIndex even when values match.
    Returns a numpy array suitable for napari (uint8).
    """
    out = overlay

    # If overlay has t but ref doesn't, take current t=0 by default
    if "t" in out.dims and "t" not in ref.dims:
        out = out.isel(t=0)

    # Add any missing dims from ref (size 1)
    for d in ref.dims:
        if d not in out.dims:
            out = out.expand_dims({d: ref.sizes[d]})
            # expand_dims with size sets length, but values are broadcast later by numpy

    # Reorder to match ref dims
    out = out.transpose(*ref.dims, missing_dims="ignore")

    # Finally broadcast by numpy, not xarray
    data = np.asarray(out.data)
    target_shape = tuple(ref.sizes[d] for d in ref.dims)

    # If needed, broadcast to target shape
    if data.shape != target_shape:
        data = np.broadcast_to(data, target_shape)

    return data.astype(np.uint8)

def _tile_axis_1d_ids(m: xr.Dataset, *, coord_name: str, tile_dim: str, pixel_dim: str, step: int) -> np.ndarray:
    da = m.coords.get(coord_name, None)
    if da is None:
        raise ValueError(f"Expected coord {coord_name!r} on montage.")

    if tile_dim in da.dims:
        return np.asarray(da.values)

    if pixel_dim in da.dims:
        # if 2D, pick a representative line to get 1D along pixel_dim
        if da.ndim == 2:
            other = [d for d in da.dims if d != pixel_dim]
            da = da.isel({other[0]: 0})
        vals = np.asarray(da.values)
        return vals[::step]

    raise ValueError(f"Coord {coord_name!r} has unexpected dims {da.dims}.")


def manual_filter_montage(
    ds: xr.Dataset,
    *,
    row: str,
    col: str,
    filter_name: str = "manual_filter",
    show: Iterable[str] = ("best_z", "ch0_mask"),
    ch: int | list[int] | tuple[int, ...] = 0,
    z_index: int = 0,
    viewer: napari.Viewer | None = None,
    write_back: bool = True,
    overlay_opacity: float = 0.35,
    single_click_delay_ms: int = 100,
    colormaps: dict[str, Any] | None = None,          # NEW
    label_colors: dict[int, Any] | None = None,       # NEW (optional override)
) -> tuple[napari.Viewer, dict[str, Any], xr.DataArray]:
    """
    Interactive manual labeling of montage tiles into a binary filter table of shape (tile_dim, t).

    Minimal interactions (no paint mode):
      - click: toggle tile (0 <-> 1)   [single-click is delayed to avoid double-click conflicts]
      - Shift+click: toggle entire montage_row

    """
    from .plot import montage  # lazy import to avoid circulars

    # --- Qt timer (for "delayed single click" so double-click doesn't trigger toggles) ---
    try:
        from qtpy.QtCore import QTimer
    except Exception:
        QTimer = None

    tile_dim = _infer_tile_dim(ds)

    # Any dims we want to *preserve as context* (exp/cell/rep/fov/...)
    # i.e. dims that are neither pixels nor the filter axes.
    _non_context = {
        tile_dim, "t",
        "y", "x", "r", "c", "z", "ch",
        "montage_row", "montage_col", "montage",
    }
    context_dims = [d for d in ds.dims if d not in _non_context]

    def _ensure_filter_table_contextual(ds: xr.Dataset, *, filter_name: str, tile_dim: str, context_dims: list[str]) -> xr.DataArray:
        """
        Ensure ds has a filter table with dims: (*context_dims, tile_dim, t).
        If ds[filter_name] exists, return it (after basic sanity checks).
        Otherwise create zeros with coords from ds.
        """
        if filter_name in ds:
            ft = ds[filter_name]
            # Must include tile_dim and t; context dims are optional but if present should match
            if tile_dim not in ft.dims or "t" not in ft.dims:
                raise ValueError(f"{filter_name!r} exists but missing dims {tile_dim!r} and/or 't'. Got {ft.dims}")
            return ft.astype(np.uint8)

        # Build coords
        coords = {}
        for d in context_dims:
            coords[d] = ds.coords[d].values if d in ds.coords else np.arange(ds.sizes[d])
        coords[tile_dim] = ds.coords[tile_dim].values if tile_dim in ds.coords else np.arange(ds.sizes[tile_dim])
        coords["t"] = ds.coords["t"].values if "t" in ds.coords else np.arange(ds.sizes["t"])

        shape = tuple(len(coords[d]) for d in context_dims) + (len(coords[tile_dim]), len(coords["t"]))
        data = np.zeros(shape, dtype=np.uint8)

        return xr.DataArray(
            data,
            dims=tuple(context_dims) + (tile_dim, "t"),
            coords=coords,
            name=filter_name,
        )

    # Base (committed) table from ds if present, else zeros
    filter_table_committed = _ensure_filter_table_contextual(
        ds, filter_name=filter_name, tile_dim=tile_dim, context_dims=context_dims
    )

    # Staged table: all edits happen here until user clicks Save/Update
    filter_table = filter_table_committed.copy(deep=True).astype(np.uint8)


    # Build montage view with your existing viewer helper
    viewer, layers = montage_viewer(
        ds,
        row=row,
        col=col,
        show=show,
        ch=ch,
        z_index=z_index,
        viewer=viewer,
        colormaps=colormaps,     # NEW
    )

    # Rebuild montage locally (montage_viewer doesn't return it)
    m = montage(ds, row=row, col=col)

    # Choose a reference image layer to infer tile sizes AND to match dims for overlays
    ref_name = None
    for nm in ("best_z", "int"):
        if nm in m.data_vars:
            da0 = _squeeze_safe(m[nm])
            if "r" in da0.dims and "c" in da0.dims:
                ref_name = nm
                break
    if ref_name is None:
        for nm in list(m.data_vars):
            da0 = _squeeze_safe(m[nm])
            if "r" in da0.dims and "c" in da0.dims:
                ref_name = nm
                break
    if ref_name is None:
        raise ValueError("Could not find any montage variable with dims including ('r','c') to anchor the view.")

    ref = _squeeze_safe(m[ref_name])
    ref = _select_first_if_present(ref, "z", z_index)
    if "ch" in ref.dims:
        ref = _select_first_if_present(ref, "ch", (ch[0] if isinstance(ch, (list, tuple)) else ch))

    # Preserve extra dims (exp/cell/rep/fov/...) as leading sliders
    if "t" in ref.dims:
        ref = ref.transpose(..., "t", "r", "c", missing_dims="ignore")
    else:
        ref = ref.transpose(..., "r", "c", missing_dims="ignore")


    tile_h, tile_w = _infer_tile_hw_from_montage_image(m)

    # tile_id grid for square montage
    tile_id_grid = np.asarray(m.coords["tile_id"].values) if "tile_id" in m.coords else None
    Mrow = int(m.sizes.get("montage_row", 0) or 0)
    Mcol = int(m.sizes.get("montage_col", 0) or 0)
    if Mrow == 0:
        if "montage_row" in m.coords:
            Mrow = int(len(np.unique(np.asarray(m.coords["montage_row"].values))))
    if Mcol == 0:
        if "montage_col" in m.coords:
            Mcol = int(len(np.unique(np.asarray(m.coords["montage_col"].values))))

    # ---- context selection from napari sliders ----
    ref_dims = list(ref.dims)
    def _current_context_isel() -> dict[str, int]:
        """
        Return {dim: index} for context dims using napari slider positions
        (if dim is present in the displayed ref).
        """
        out = {}
        for d in context_dims:
            if d in ref_dims:
                ax = ref_dims.index(d)
                out[d] = int(viewer.dims.current_step[ax])
            else:
                out[d] = 0
        return out

    def _view_filter_table() -> xr.DataArray:
        """2D view (tile_dim, t) at the current context."""
        sel = _current_context_isel()
        if sel:
            return filter_table.isel(sel)
        return filter_table


    # # Build initial overlay from filter_table and add as labels
    # overlay = _overlay_pixels_from_filter_table(
    #     m=m,
    #     filter_table=_view_filter_table(),   # <- IMPORTANT (2D slice)
    #     tile_dim=tile_dim,
    #     row=row,
    #     col=col,
    #     tile_h=tile_h,
    #     tile_w=tile_w,
    # )

    # overlay_np = _broadcast_overlay_like_ref(overlay, ref)
    def _overlay_np_for_all_contexts() -> np.ndarray:
        """
        Build a labels array aligned to `ref` (including exp/cell/rep/...) by
        filling each context slice from the corresponding slice of `filter_table`.
        """
        # Start from zeros in the full ref shape
        out = np.zeros(tuple(ref.sizes[d] for d in ref.dims), dtype=np.uint8)

        # If there are no context dims, keep old fast path
        if not context_dims:
            ov = _overlay_pixels_from_filter_table(
                m=m,
                filter_table=_view_filter_table(),
                tile_dim=tile_dim,
                row=row,
                col=col,
                tile_h=tile_h,
                tile_w=tile_w,
            )
            return _broadcast_overlay_like_ref(ov, ref).astype(np.uint8)

        # Iterate over all context index combinations (usually small: exp=2, etc.)
        from itertools import product
        ranges = [range(int(ref.sizes[d])) if d in ref.dims else range(1) for d in context_dims]

        for combo in product(*ranges):
            # Build isel dict for filter_table and for the output array
            ctx_isel = {d: combo[i] for i, d in enumerate(context_dims) if d in filter_table.dims}
            ft2 = filter_table.isel(ctx_isel)  # -> (tile_dim, t)

            ov = _overlay_pixels_from_filter_table(
                m=m,
                filter_table=ft2,
                tile_dim=tile_dim,
                row=row,
                col=col,
                tile_h=tile_h,
                tile_w=tile_w,
            )
            ov_np = _broadcast_overlay_like_ref(ov, ref).astype(np.uint8)

            # Now write ONLY into this context slice of `out`
            sl = [slice(None)] * out.ndim
            for d, idx in ctx_isel.items():
                if d in ref.dims:
                    sl[ref.dims.index(d)] = int(idx)

            out[tuple(sl)] = ov_np[tuple(sl)]

        return out

    # Build initial overlay from the staged table across ALL contexts
    overlay_np = _overlay_np_for_all_contexts()

    from napari.utils.colormaps import DirectLabelColormap
    lbl_layer = viewer.add_labels(
        overlay_np.astype(np.uint8),
        name=filter_name,
        opacity=overlay_opacity,
    )

    # ---- Label colors (avoid napari's default brown) ----
    def _as_rgba(c):
        # Accept either a string or an RGB/RGBA tuple
        if isinstance(c, str):
            return c
        arr = np.asarray(c, dtype=float)
        if arr.ndim == 1 and arr.size == 3:
            arr = np.concatenate([arr, [1.0]])
        return tuple(arr.tolist())

    # Determine "on" color
    on_color = None

    if label_colors is not None and 1 in label_colors:
        on_color = label_colors[1]
    elif colormaps is not None:
        if filter_name in colormaps:
            on_color = colormaps[filter_name]
        else:
            for nm in show:
                if nm in colormaps:
                    on_color = colormaps[nm]
                    break

    if on_color is None:
        on_color = (1.0, 1.0, 0.0, 1.0)  # default yellow

    lbl_layer.colormap = DirectLabelColormap(
        color_dict={
            None: (0.0, 0.0, 0.0, 0.0),
            0: (0.0, 0.0, 0.0, 0.0),   # background transparent
            1: _as_rgba(on_color),     # active label
        }
    )

    # ------------------------------------------------------------
    # Debug overlay: shows mapping info when clicking a tile
    # ------------------------------------------------------------
    click_info = viewer.add_points(
        np.zeros((0, overlay_np.ndim), dtype=float),
        name="click_info",
        size=0.0,  # hide point marker, show text only
    )
    # Hide point glyphs; text only
    if hasattr(click_info, "face_color"):
        click_info.face_color = "transparent"
    if hasattr(click_info, "edge_color"):
        click_info.edge_color = "transparent"


    # Configure text rendering
    click_info.text = {
        "string": "{label}",
        "size": 12,
        "anchor": "center",
    }

    layers["click_info"] = click_info

    # Make label=1 visible
    try:
        lbl_layer.color = {1: "yellow"}
    except Exception:
        lbl_layer.color = {1: np.array([1.0, 1.0, 0.0, 1.0])}

    # IMPORTANT: do NOT enter paint mode; avoid editing gestures
    lbl_layer.editable = False
    try:
        lbl_layer.mode = "pan_zoom"
    except Exception:
        pass

    layers[filter_name] = lbl_layer

    # Helpers
    square = (row == col)
    tile_row_id = m.coords.get("tile_row_id", None)
    tile_col_id = m.coords.get("tile_col_id", None)
    # Tile-level identity vectors (robust even if coords are pixel-broadcast)
    tile_row_id_1d = _tile_axis_1d_ids(m, coord_name="tile_row_id", tile_dim="montage_row", pixel_dim="r", step=tile_h) if tile_row_id is not None else None
    tile_col_id_1d = _tile_axis_1d_ids(m, coord_name="tile_col_id", tile_dim="montage_col", pixel_dim="c", step=tile_w) if tile_col_id is not None else None


    ref_dims = list(ref.dims)
    r_axis = ref_dims.index("r")
    c_axis = ref_dims.index("c")
    t_axis = ref_dims.index("t") if "t" in ref_dims else None

    tile_index = {v: i for i, v in enumerate(filter_table.coords[tile_dim].values)}
    t_index = {v: i for i, v in enumerate(filter_table.coords["t"].values)}

    def _set_filter_value(tile_val, t_val, value: int):
        if tile_val in tile_index and t_val in t_index:
            filter_table.values[tile_index[tile_val], t_index[t_val]] = np.uint8(value)

    def _paint_tile_pixel(mr: int, mc: int, value: int, *, t_idx_for_pixels: int | None):
        r0, r1 = mr * tile_h, (mr + 1) * tile_h
        c0, c1 = mc * tile_w, (mc + 1) * tile_w

        data = lbl_layer.data
        sl = [slice(None)] * data.ndim

        # --- NEW: restrict painting to the current context slice (exp/cell/rep/...) ---
        ctx = _current_context_isel()  # uses viewer sliders + ref_dims
        for d, i in ctx.items():
            if d in ref_dims:
                sl[ref_dims.index(d)] = int(i)

        # Pixel window
        sl[r_axis] = slice(r0, r1)
        sl[c_axis] = slice(c0, c1)

        # Time plane
        if t_axis is not None:
            if t_idx_for_pixels is None:
                raise ValueError("t_idx_for_pixels is required when label data has a 't' axis.")
            sl[t_axis] = int(t_idx_for_pixels)

        data[tuple(sl)] = value
        lbl_layer.data = data
        lbl_layer.refresh()


    def _update_click_info(
        *,
        mr: int,
        mc: int,
        tile_val,
        t_val,
        t_idx_for_pixels: int | None,
        display_tile_label: str | None = None,
        track_id_orig=None,
    ):
        """
        Place a text label at the center of the clicked montage tile showing
        montage indices and underlying identity.
        """
        # Center of the clicked tile in pixel space
        y = mr * tile_h + tile_h / 2
        x = mc * tile_w + tile_w / 2

        pt = np.zeros((1, overlay_np.ndim), dtype=float)

        # If montage includes time as an axis, place the point in the current frame plane
        if t_axis is not None:
            pt[0, t_axis] = float(t_idx_for_pixels) if t_idx_for_pixels is not None else 0.0

        pt[0, r_axis] = y
        pt[0, c_axis] = x

        click_info.data = pt

        # Prefer displaying the grid-pair label if provided
        disp = display_tile_label if display_tile_label is not None else str(tile_val)

        label_lines = [
            f"grid_rc={disp}   f={t_val}",
            f"track_id_orig={track_id_orig}" if track_id_orig is not None else "track_id_orig=None",
            f"montage (row,col)=({mr},{mc})",
        ]

        if track_id_orig is not None:
            label_lines.insert(1, f"track_id_orig={track_id_orig}")

        click_info.features = {"label": np.array(["\n".join(label_lines)], dtype=object)}
        click_info.refresh()


    def _paint_row_pixels(mr: int, value: int, *, t_idx_for_pixels: int | None):
        for mc in range(Mcol):
            _paint_tile_pixel(mr, mc, value, t_idx_for_pixels=t_idx_for_pixels)

    # --- delayed single-click state ---
    click_state = {
        "timer": None,
        "pending": False,
        "payload": None,  # (mr, mc, shift_down, t_idx_for_pixels, tile_val, t_val)
    }

    def _apply_toggle_or_row(payload):
        """
        payload = (mr, mc, shift_down, t_idx_for_pixels, tile_val, t_idx_for_table)

        Updates staged `filter_table` at the CURRENT context (exp/cell/rep/fov sliders),
        and updates the displayed overlay pixels in napari.
        """
        nonlocal dirty

        mr, mc, shift_down, t_idx_for_pixels, tile_val, t_idx_for_table = payload

        # Clamp time
        T = int(filter_table.sizes["t"])
        t_idx_for_table = int(np.clip(int(t_idx_for_table), 0, max(0, T - 1)))

        # Only clamp pixels-time if the displayed labels layer actually has a t axis
        if t_axis is not None:
            t_idx_for_pixels = int(np.clip(int(t_idx_for_pixels), 0, max(0, T - 1)))
        else:
            t_idx_for_pixels = None


        ctx = _current_context_isel()  # {exp: i, cell: j, ...}

        # Build a numpy index tuple into filter_table.data
        # dims: (*context_dims, tile_dim, t)
        def _idx(tile_i: int, t_i: int):
            parts = []
            for d in context_dims:
                parts.append(ctx.get(d, 0))
            parts.append(tile_i)
            parts.append(t_i)
            return tuple(parts)

        # Map tile_val -> positional index along tile_dim
        # (tile_val is the true track_id coordinate value, not an index)
        try:
            tile_i = int(np.where(filter_table.coords[tile_dim].values == tile_val)[0][0])
        except Exception:
            # If coordinate values aren’t unique or something odd, fall back to dict lookup
            tile_index = {v: i for i, v in enumerate(filter_table.coords[tile_dim].values)}
            if tile_val not in tile_index:
                return
            tile_i = tile_index[tile_val]

        # --- SHIFT: toggle the whole track (tile) across ALL t, but only in this context ---
        if shift_down:
            cur = int(filter_table.data[_idx(tile_i, t_idx_for_table)])
            new = 0 if cur == 1 else 1

            # set all t for this tile in this context
            for ti in range(T):
                filter_table.data[_idx(tile_i, ti)] = np.uint8(new)

            dirty = True

            # Paint: for square montage you already decided to paint same tile across all frames
            if square:
                for ti in range(T):
                    _paint_tile_pixel(mr, mc, new, t_idx_for_pixels=ti)
                return

            # # Non-square: rebuild overlay for this context only
            # overlay = _overlay_pixels_from_filter_table(
            #     m=m,
            #     filter_table=_view_filter_table(),
            #     tile_dim=tile_dim,
            #     row=row,
            #     col=col,
            #     tile_h=tile_h,
            #     tile_w=tile_w,
            # )
            # overlay_np = _broadcast_overlay_like_ref(overlay, ref)
            # lbl_layer.data = overlay_np.astype(np.uint8)
            # lbl_layer.refresh()
            # return
            overlay = _overlay_pixels_from_filter_table(
                m=m,
                filter_table=_view_filter_table(),
                tile_dim=tile_dim,
                row=row,
                col=col,
                tile_h=tile_h,
                tile_w=tile_w,
            )
            overlay_np = _broadcast_overlay_like_ref(overlay, ref).astype(np.uint8)

            # write overlay_np into just the current context slice
            data = lbl_layer.data
            sl = [slice(None)] * data.ndim
            ctx = _current_context_isel()
            for d, i in ctx.items():
                if d in ref_dims:
                    sl[ref_dims.index(d)] = int(i)
            data[tuple(sl)] = overlay_np[tuple(sl)]
            lbl_layer.data = data
            lbl_layer.refresh()
            return


        # --- Normal click: toggle single (tile, t) in this context ---
        cur = int(filter_table.data[_idx(tile_i, t_idx_for_table)])
        new = 0 if cur == 1 else 1
        filter_table.data[_idx(tile_i, t_idx_for_table)] = np.uint8(new)
        dirty = True

        _paint_tile_pixel(mr, mc, new, t_idx_for_pixels=t_idx_for_pixels)




    dirty = False  # staged edits not yet committed to ds


    def _write_filter_table_back():
        nonlocal dirty
        # Write the full contextual filter table back
        ds[filter_name] = filter_table.astype(np.uint8)
        dirty = False


    def _clear_filter_table():
        """Clear staged edits only (does not touch ds)."""
        nonlocal dirty

        # Clear ONLY the current context slice (exp/cell/rep/...)
        ctx = _current_context_isel()
        if ctx:
            # Only keep dims that actually exist on filter_table
            ft_ctx = {d: int(i) for d, i in ctx.items() if d in filter_table.dims}
            if ft_ctx:
                filter_table.loc[ft_ctx] = 0
            else:
                # no matching context dims on filter_table -> clear all
                filter_table.loc[:] = 0
        else:
            filter_table.loc[:] = 0

        dirty = True

        # Rebuild overlay pixels from the current context slice
        overlay = _overlay_pixels_from_filter_table(
            m=m,
            filter_table=_view_filter_table(),  # must return (tile_dim, t) slice
            tile_dim=tile_dim,
            row=row,
            col=col,
            tile_h=tile_h,
            tile_w=tile_w,
        )
        overlay_np = _broadcast_overlay_like_ref(overlay, ref).astype(np.uint8)

        # Write overlay into just the current context slice of the labels layer
        data = lbl_layer.data
        sl = [slice(None)] * data.ndim
        for d, i in ctx.items():
            if d in ref_dims:
                sl[ref_dims.index(d)] = int(i)

        data[tuple(sl)] = overlay_np[tuple(sl)]
        lbl_layer.data = data
        lbl_layer.refresh()





    def _tile_id_orig_from_m(*, mr: int, mc: int):
        """
        For square montages (row == col), return the true underlying identity from
        the 2D tile_id grid (robust even after stacking).
        """
        if tile_id_grid is None:
            return None
        try:
            return tile_id_grid[mr, mc].item() if hasattr(tile_id_grid[mr, mc], "item") else tile_id_grid[mr, mc]
        except Exception:
            return None

    def _track_id_orig_from_grid(*, mr: int, mc: int):
        """
        Undo square-packing for row==col=='track_id' montages.

        Returns the *true* ds track_id coord value for the tile at (mr, mc),
        or None if that tile is padding/out of range.
        """
        if row != col:
            return None
        if row != "track_id":
            return None
        if "track_id" not in ds.dims:
            return None

        n_tiles = int(ds.sizes["track_id"])
        S = int(np.ceil(np.sqrt(n_tiles)))
        k = int(mr) * S + int(mc)

        if k < 0 or k >= n_tiles:
            return None

        v = ds.coords["track_id"].values[k]
        try:
            return v.item()
        except Exception:
            return v


    @lbl_layer.mouse_drag_callbacks.append
    def _on_click(layer, event):
        # Only act on mouse press
        if event.type != "mouse_press":
            yield
            return

        # Cancel pending single-click if this is the 2nd click (double-click => do nothing)
        if QTimer is not None and click_state.get("pending", False):
            try:
                click_state["timer"].stop()
            except Exception:
                pass
            click_state["pending"] = False
            click_state["payload"] = None
            yield
            return

        data_pos = layer.world_to_data(event.position)

        # Pixel coords in montage canvas (r/c axes in data coords)
        ry = int(np.clip(int(round(data_pos[r_axis])), 0, m.sizes["r"] - 1))
        cx = int(np.clip(int(round(data_pos[c_axis])), 0, m.sizes["c"] - 1))

        # Tile indices in montage grid
        mr = int(np.clip(ry // tile_h, 0, max(0, Mrow - 1)))
        mc = int(np.clip(cx // tile_w, 0, max(0, Mcol - 1)))

        mods = set(getattr(event, "modifiers", ()) or ())
        shift_down = ("Shift" in mods)

        # ---- Identity from montage coordinate mappings (tile-level, not pixel-broadcast) ----
        row_val = tile_row_id_1d[mr] if tile_row_id_1d is not None else mr
        col_val = tile_col_id_1d[mc] if tile_col_id_1d is not None else mc

        T = int(filter_table.sizes["t"])

        # ---- Determine which time index we are editing ----
        if square:
            # Square: use napari t slider if present
            if t_axis is not None:
                t_idx_for_table = int(viewer.dims.current_step[t_axis])
            else:
                t_idx_for_table = 0
            t_idx_for_table = int(np.clip(t_idx_for_table, 0, max(0, T - 1)))

            # Square labels layer has a t axis, so we also paint only that plane
            t_idx_for_pixels = t_idx_for_table
            t_val_for_debug = t_idx_for_table

        else:
            # Rectangular row=track_id, col=t: time is encoded in montage columns.
            # col_val is the *t coordinate value* for that montage column.
            t_val_for_debug = col_val
            t_idx_for_table = int(t_index.get(t_val_for_debug, 0))
            t_idx_for_table = int(np.clip(t_idx_for_table, 0, max(0, T - 1)))

            # Rectangular labels layer is 2D (r,c), no t axis to index
            t_idx_for_pixels = None

        # ---- Determine tile identity and debug label ----
        if square:
            # grid_rc purely for display/debug
            debug_tile_label = f"{mr},{mc}"

            # True track id from the original ds (NOT the grid coords)
            track_id_orig = _track_id_orig_from_grid(mr=mr, mc=mc)

            # If this is a padded tile, ignore clicks
            if track_id_orig is None:
                yield
                return

            # This is what we use for filter_table indexing / saving back to ds
            tile_val = track_id_orig

        else:
            # Rectangular: row_val is the track_id value for that montage row
            tile_val = int(row_val)
            debug_tile_label = str(tile_val)
            track_id_orig = None

        # Optional: update debug overlay (if you have it wired up)
        _update_click_info(
            mr=mr,
            mc=mc,
            tile_val=tile_val,
            t_val=t_val_for_debug,              # shows true t coord in rectangular mode
            t_idx_for_pixels=t_idx_for_pixels,  # None in rectangular mode
            display_tile_label=debug_tile_label,
            track_id_orig=track_id_orig,
        )

        # payload = (mr, mc, shift_down, t_idx_for_pixels, tile_val, t_idx_for_table)
        payload = (mr, mc, shift_down, t_idx_for_pixels, tile_val, t_idx_for_table)

        # Fire immediately if no Qt timer available; otherwise delay to avoid double-click conflicts
        if QTimer is None:
            _apply_toggle_or_row(payload)
            yield
            return

        click_state["pending"] = True
        click_state["payload"] = payload

        timer = QTimer()
        click_state["timer"] = timer

        def _fire():
            if not click_state.get("pending", False):
                return
            click_state["pending"] = False
            pl = click_state.get("payload", None)
            click_state["payload"] = None
            if pl is not None:
                _apply_toggle_or_row(pl)

        timer.setSingleShot(True)
        timer.timeout.connect(_fire)
        timer.start(int(single_click_delay_ms))

        yield


    


    from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel

    w = QWidget()
    layout = QVBoxLayout(w)

    help_lbl = QLabel(
        "Manual filter:\n"
        "- Click: toggle crop\n"
        "- Shift+Click: take all frames (f)\n"
        "- Press 'Save/Update' to write into ds\n"
        "- Press 'Clear' to reset staged edits\n"
    )
    help_lbl.setWordWrap(True)

    btn_save = QPushButton("Save/Update (write to ds)")
    btn_clear = QPushButton("Clear (staged only)")

    def _on_save():
        _write_filter_table_back()

    def _on_clear():
        _clear_filter_table()

    btn_save.clicked.connect(_on_save)
    btn_clear.clicked.connect(_on_clear)

    layout.addWidget(help_lbl)
    layout.addWidget(btn_save)
    layout.addWidget(btn_clear)

    viewer.window.add_dock_widget(w, name=f"{filter_name} controls", area="right")


    return viewer, layers, filter_table
