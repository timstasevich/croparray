from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import xarray as xr
import napari

__all__ = ["montage_viewer"]



@dataclass(frozen=True)
class ContrastSpec:
    lo_pct: float = 2.0
    hi_pct: float = 98.0

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

def _infer_tile_hw_from_montage_image(img: xr.DataArray) -> tuple[int, int]:
    """
    Montage image dims should include r,c where:
      r is a MultiIndex of (montage_row, y)
      c is a MultiIndex of (montage_col, x)
    So y/x appear as coords even if not dims.
    """
    if "y" not in img.coords or "x" not in img.coords:
        raise ValueError("Expected montage image to have coords 'y' and 'x' (from stacked r/c).")
    y_vals = np.unique(img.coords["y"].values)
    x_vals = np.unique(img.coords["x"].values)
    return int(len(y_vals)), int(len(x_vals))

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
    image_contrast : ContrastSpec
        Percentiles for image layers (default 2–98).
    tile_overlay_contrast : ContrastSpec
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
        image_contrast = ContrastSpec(0.5, 99.5)
    if tile_overlay_contrast is None:
        tile_overlay_contrast = ContrastSpec(5, 95)
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
    if "t" in ref.dims:
        ref = ref.transpose("t", "r", "c", missing_dims="ignore")
    else:
        ref = ref.transpose("r", "c", missing_dims="ignore")

    tile_h, tile_w = _infer_tile_hw_from_montage_image(ref)

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

        if "t" in da.dims:
            da = da.transpose("t", "r", "c", "ch", missing_dims="ignore")
        else:
            da = da.transpose("r", "c", "ch", missing_dims="ignore")

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
        da = _squeeze_safe(da)
        da = _select_first_if_present(da, "z", z_index)
        da = _select_first_if_present(da, "ch", ch)
        if "t" in da.dims:
            da = da.transpose("t", "r", "c", missing_dims="ignore")
        else:
            da = da.transpose("r", "c", missing_dims="ignore")

        lbl = (da > 0).astype(np.uint8).data
        lyr = viewer.add_labels(
            lbl,
            name=name,
            opacity=tile_overlay_opacity,
        )
        layers[name] = lyr

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



