from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import xarray as xr
import napari
import os
from pathlib import Path

__all__ = ["montage_viewer","manual_filter_montage"]


# -----------------------------------------------------------------------------
# Axis-label helpers (xarray/CropArray dims -> napari axis labels)
# -----------------------------------------------------------------------------
_DIM_TO_NAPARI_AXIS = {
    # montage pixel dims
    "r": "y",
    "c": "x",
    # common alternates
    "row": "y",
    "col": "x",
}


def _napari_axis_labels_from_xr_dims(dims: Iterable[str]) -> list[str]:
    """Map xarray dimension names to napari axis labels.

    Napari uses semantic axis labels for sliders; we prefer to propagate the
    underlying xarray dims so time is labeled 't' and montage pixel dims become
    y/x instead of r/c.
    """
    out: list[str] = []
    for d in dims:
        out.append(_DIM_TO_NAPARI_AXIS.get(str(d), str(d)))
    return out


def _add_layer_with_axis_labels(
    viewer: "napari.Viewer",
    add_fn,
    data: np.ndarray,
    *,
    xr_dims: Iterable[str] | None,
    name: str,
    **kwargs,
):
    """Add a napari layer and attach axis labels derived from xr_dims.

    - Newer napari supports `axis_labels=[...]` in add_image/add_labels.
    - Older napari ignores it; in that case we store labels in layer.metadata
      and (optionally) set viewer.dims.axis_labels when the dimensionality
      matches.

    This is intentionally low-risk: the metadata is tiny and does not hold onto
    the data array (so it won't create the "memory sap" you saw earlier).
    """
    axis_labels = _napari_axis_labels_from_xr_dims(xr_dims) if xr_dims is not None else None

    if axis_labels is not None:
        try:
            layer = add_fn(data, name=name, axis_labels=axis_labels, **kwargs)
        except TypeError:
            # Older napari: no axis_labels kwarg
            layer = add_fn(data, name=name, **kwargs)
    else:
        layer = add_fn(data, name=name, **kwargs)

    # Always stash for debugging/introspection
    try:
        if axis_labels is not None:
            layer.metadata = dict(getattr(layer, "metadata", {}) or {})
            layer.metadata["axis_labels"] = list(axis_labels)
    except Exception:
        pass

    # Best-effort: if napari hasn't labeled axes, set them when sizes match.
    # (This affects the global slider labels, not per-layer.)
    try:
        if axis_labels is not None and hasattr(viewer, "dims"):
            if len(getattr(viewer.dims, "axis_labels", [])) == data.ndim:
                viewer.dims.axis_labels = list(axis_labels)
    except Exception:
        pass

    return layer

# @dataclass(frozen=True)
# class ContrastSpec:
#     lo_pct: float = 2.0
#     hi_pct: float = 98.0

def save_manual_filter_sidecar(
    ds: xr.Dataset,
    *,
    filter_name: str,
    output_dir: str | None = None,
) -> str:
    """
    Save only ds[filter_name] + its coords into a per-filter sidecar NetCDF.
    Returns the path written.
    """
    if filter_name not in ds:
        raise ValueError(f"'{filter_name}' not found in dataset.")

    path = _manual_filter_sidecar_path(ds, filter_name=filter_name, output_dir=output_dir)

    da = ds[filter_name]

    # Save as a tiny dataset with coords needed for alignment
    out = xr.Dataset({filter_name: da})

    # (Optional) store some metadata about provenance
    src = ds.encoding.get("source", None)
    if isinstance(src, str):
        out.attrs["croparray_source"] = src
    out.attrs["manual_filter_name"] = filter_name

    # Atomic write: write to temp then replace
    tmp = path + ".tmp"
    out.to_netcdf(tmp, mode="w")
    os.replace(tmp, path)

    return path

def _sidecar_var_compatible(ds: xr.Dataset, v: str, da: xr.DataArray) -> bool:
    """
    Return True if `da` can be assigned into ds[v] without reindexing/broadcasting surprises.
    Requires:
      - same dims (order-independent)
      - for any dim present, identical coordinate values (if both have coords)
    """
    # dims must match as a set
    if set(da.dims) != set(ds.dims) and v not in ds:
        # If ds doesn't already have v, we still require da dims to be subset of ds dims.
        # (Your filters are usually full dims; change this if you intentionally store reduced dims.)
        if not set(da.dims).issubset(set(ds.dims)):
            return False

    # coordinate values must match for each dim in da
    for d in da.dims:
        if d not in ds.dims:
            return False

        # Compare coordinate arrays if both exist
        if d in da.coords and d in ds.coords:
            a = da.coords[d].values
            b = ds.coords[d].values
            # exact match (including order)
            if a.shape != b.shape:
                return False
            try:
                if not np.array_equal(a, b):
                    return False
            except Exception:
                return False

    return True

def _manual_filter_sidecar_path(
    ds: xr.Dataset,
    *,
    filter_name: str,
    output_dir: str | None = None,
) -> str:
    """
    Example:
      filename attr: mydata.nc
      filter: stretchy
      -> <out_dir>/mydata__manual__stretchy.nc
    """
    # Prefer ds.attrs["filename"] for standardized affiliation
    fn_attr = None
    try:
        fn_attr = ds.attrs.get("filename", None)
    except Exception:
        fn_attr = None

    if isinstance(fn_attr, str) and fn_attr.strip():
        base = os.path.basename(fn_attr.strip())
        stem = base[:-3] if base.endswith(".nc") else base
        src = ds.encoding.get("source", None)
        default_dir = os.path.dirname(src) if isinstance(src, str) and src.strip() else os.getcwd()
    else:
        src = ds.encoding.get("source", None)
        if isinstance(src, str) and src.strip():
            base = os.path.basename(src)
            stem = base[:-3] if base.endswith(".nc") else base
            default_dir = os.path.dirname(src)
        else:
            stem = "dataset"
            default_dir = os.getcwd()

    out_dir = output_dir if output_dir is not None else default_dir
    filename = f"{stem}__manual__{filter_name}.nc"
    return os.path.join(out_dir, filename)

def _manual_filter_sidecar_glob(
    ds: xr.Dataset,
    *,
    output_dir: str | None = None,
) -> tuple[str, str]:
    """Return (out_dir, glob_pattern) for sidecars affiliated with ds."""
    # Derive the same stem used by _manual_filter_sidecar_path
    path_for_dummy = _manual_filter_sidecar_path(ds, filter_name="__DUMMY__", output_dir=output_dir)
    out_dir = os.path.dirname(path_for_dummy)
    dummy_base = os.path.basename(path_for_dummy)
    stem = dummy_base.split("__manual__", 1)[0]
    pattern = os.path.join(out_dir, f"{stem}__manual__*.nc")
    return out_dir, pattern

def load_manual_filter_sidecars(
    ds: xr.Dataset,
    *,
    output_dir: str | None = None,
    overwrite: bool = True,
) -> tuple[xr.Dataset, list[str]]:
    """
    Load any existing manual-filter sidecars affiliated with ds and merge them in.

    If overwrite=True, sidecar values replace ds[var] when present.
    """
    import glob

    _, pattern = _manual_filter_sidecar_glob(ds, output_dir=output_dir)
    paths = sorted(glob.glob(pattern))
    if not paths:
        return ds, []

    loaded: list[str] = []
    for p in paths:
        try:
            s = xr.open_dataset(p).load()  # tiny file; load eagerly to avoid open handles

            # Prefer explicit attr; else first data_var
            v = s.attrs.get("manual_filter_name", None)
            if not (isinstance(v, str) and v in s.data_vars):
                dvs = list(s.data_vars)
                if not dvs:
                    continue
                v = dvs[0]

            da = s[v].astype(np.uint8)

            # ---- NEW: compatibility gate ----
            if not _sidecar_var_compatible(ds, v, da):
                # skip incompatible sidecars (prevents "misaligned layers")
                continue

            if overwrite or (v not in ds):
                ds[v] = da

            loaded.append(p)
        except Exception:
            continue

    if loaded:
        show_info("Loaded manual filters:\n" + "\n".join(loaded))

    return ds, loaded

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
    show_tile_text: bool = True,
    tile_text_name: str = "tile_text",
    tile_text_size: int = 14,
    tile_text_color: str | tuple = "white",
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

    # ---------------------------------------------------------------------
    # Tile text overlay (track_id,t) or (track_id) centered per tile
    # ---------------------------------------------------------------------
    def _add_tile_text_layer():
        # Determine montage grid size (MR x MC)
        Mrow = int(m.sizes.get("montage_row", 0) or 0)
        Mcol = int(m.sizes.get("montage_col", 0) or 0)
        if Mrow == 0 and "montage_row" in m.coords:
            Mrow = int(len(np.unique(np.asarray(m.coords["montage_row"].values))))
        if Mcol == 0 and "montage_col" in m.coords:
            Mcol = int(len(np.unique(np.asarray(m.coords["montage_col"].values))))

        # Robustly get tile-level row/col IDs even if coords are pixel-broadcast
        row_ids = None
        col_ids = None
        if not square:
            row_ids = _tile_axis_1d_ids(
                m, coord_name="tile_row_id",
                tile_dim="montage_row", pixel_dim="r", step=tile_h
            )
            col_ids = _tile_axis_1d_ids(
                m, coord_name="tile_col_id",
                tile_dim="montage_col", pixel_dim="c", step=tile_w
            )

        # Build label grid (MR x MC) as strings
        labels_2d = np.empty((Mrow, Mcol), dtype=object)

        if square and row == col == "track_id" and ("track_id" in ca_or_ta.dims):
            # Square-packed track_id x track_id
            n_tiles = int(ca_or_ta.sizes["track_id"])
            S = int(np.ceil(np.sqrt(n_tiles)))
            track_vals = np.asarray(ca_or_ta.coords["track_id"].values)

            for mr in range(Mrow):
                for mc in range(Mcol):
                    k = mr * S + mc
                    if 0 <= k < n_tiles:
                        v = track_vals[k]
                        try:
                            v = v.item()
                        except Exception:
                            pass
                        labels_2d[mr, mc] = f"{v}"
                    else:
                        labels_2d[mr, mc] = ""

        elif square:
            # Generic square
            ids = np.asarray(tile_id_2d.values)
            for mr in range(Mrow):
                for mc in range(Mcol):
                    v = ids[mr, mc]
                    try:
                        v = v.item()
                    except Exception:
                        pass
                    labels_2d[mr, mc] = f"{v}"

        else:
            # Rectangular: row=track_id, col=t
            for mr in range(Mrow):
                rid = row_ids[mr]
                try:
                    rid = int(rid)
                except Exception:
                    pass
                for mc in range(Mcol):
                    cid = col_ids[mc]
                    if col == "t":
                        try:
                            tv = int(cid)
                        except Exception:
                            tv = cid
                        labels_2d[mr, mc] = f"{rid},{tv}"
                    else:
                        labels_2d[mr, mc] = f"{rid}"

        # Create a Points layer with text
        ref_dims = list(ref.dims)
        r_axis = ref_dims.index("r")
        c_axis = ref_dims.index("c")
        t_axis = ref_dims.index("t") if "t" in ref_dims else None
        extra_dims = [d for d in ref_dims if d not in ("t", "r", "c")]

        from itertools import product
        extra_ranges = [range(int(ref.sizes[d])) for d in extra_dims]
        t_range = range(int(ref.sizes["t"])) if ("t" in ref.dims) else range(1)

        pts = []
        txt = []

        for extra_idx in product(*extra_ranges) if extra_dims else [()]:
            for tt in t_range:
                for mr in range(Mrow):
                    y = mr * tile_h + tile_h / 2
                    for mc in range(Mcol):
                        lab = labels_2d[mr, mc]
                        if not lab:
                            continue
                        x = mc * tile_w + tile_w / 2

                        coord = np.zeros(len(ref_dims), dtype=float)
                        for j, d in enumerate(extra_dims):
                            coord[ref_dims.index(d)] = float(extra_idx[j])
                        if t_axis is not None:
                            coord[t_axis] = float(tt)
                        coord[r_axis] = float(y)
                        coord[c_axis] = float(x)

                        pts.append(coord)
                        txt.append(lab)

        pts = np.asarray(pts, dtype=float) if pts else np.zeros((0, len(ref_dims)))

        text_layer = viewer.add_points(
            pts,
            name=tile_text_name,
            size=0.0,
        )
        try:
            text_layer.face_color = "transparent"
            text_layer.edge_color = "transparent"
        except Exception:
            pass

        text_layer.text = {
            "string": "{label}",
            "size": tile_text_size,
            "anchor": "center",
            "color": tile_text_color,
        }
        text_layer.features = {"label": np.asarray(txt, dtype=object)}
        layers[tile_text_name] = text_layer

    if show_tile_text:
        _add_tile_text_layer()



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
            lyr = _add_layer_with_axis_labels(
                viewer,
                viewer.add_image,
                data,
                xr_dims=list(da.dims),
                name=name,
                rgb=True,
                blending=default_blending,
            )

        else:
            #print("DEBUG image_contrast:", type(image_contrast), image_contrast)
            lo, hi = _normalize_image_contrast(image_contrast)
            clim = _robust_limits_nonneg(
                np.asarray(da.data),
                hi,
                lo_fixed=lo,
            )
            lyr = _add_layer_with_axis_labels(
                viewer,
                viewer.add_image,
                data,
                xr_dims=list(da.dims),
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
        lyr = _add_layer_with_axis_labels(
            viewer,
            viewer.add_labels,
            np.asarray(lbl),
            xr_dims=list(da.dims),
            name=name,
            opacity=tile_overlay_opacity,
        )

        # ---- Force label colors (avoid napari's default brown) ----
        on_color = colormaps.get(name, "magenta")  # e.g. colormaps["ch0_mask"] = "magenta"
        try:
            rgba = tuple(float(x) for x in transform_color([on_color])[0])  # (4,)
        except Exception:
            rgba = on_color  # some napari versions accept strings directly

        transparent = (0.0, 0.0, 0.0, 0.0)
        lyr.colormap = DirectLabelColormap(
            color_dict={
                None: transparent,
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
        lo, hi = _normalize_image_contrast(tile_overlay_contrast)
        clim = _robust_limits_nonneg(np.asarray(pix), hi, lo_fixed=lo)

        # pix is pixel-space overlay aligned to the montage. Give it explicit axis labels.
        pix_axis = ["t", "y", "x"] if np.asarray(pix).ndim == 3 else ["y", "x"]
        lyr = _add_layer_with_axis_labels(
            viewer,
            viewer.add_image,
            np.asarray(pix),
            xr_dims=pix_axis,
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
def _resolve_output_path(
    ds: xr.Dataset,
    *,
    output_dir: str | None = None,
    default_name: str = "manual_filter.nc",
) -> str:
    # Try to infer from dataset source
    src = ds.encoding.get("source", None)

    if isinstance(src, str):
        base = os.path.basename(src)   # KEEP ORIGINAL NAME
    else:
        base = default_name

    # Determine directory
    if output_dir is not None:
        out_dir = output_dir
    elif isinstance(src, str):
        out_dir = os.path.dirname(src)
    else:
        out_dir = os.getcwd()

    return os.path.join(out_dir, base)

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
    output_dir: str | Path,
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
    colormaps: dict[str, Any] | None = None,
    label_colors: dict[int, Any] | None = None,
    show_click_info: bool = False,
    show_tile_text: bool = False,
) -> tuple[napari.Viewer, dict[str, Any], xr.DataArray]:
    ...
    """
    Launch an interactive napari montage viewer for manual binary filtering
    of tiles (e.g., track_id × t) and create/update a filter table.

    This function allows interactive selection of tiles in a montage layout.
    Edits are staged in memory and can be committed to the dataset or written
    to disk via UI controls.

    Interaction model
    -----------------
    - Alt+Click:
        Toggle a single (tile, t) entry.
    - Shift+Click:
        Toggle the selected tile across all timepoints (row operation).
    - Click-drag:
        Navigation only (pan/zoom). No edits occur without modifiers.

    Parameters
    ----------
    ds : xr.Dataset
        CropArray or TrackArray dataset containing tile identity dimension
        (e.g., ``track_id`` or ``n``) and time dimension ``t``.

    row, col : str
        Dimensions used to construct the montage grid (e.g.,
        ``row="track_id", col="t"`` or square layouts).

    filter_name : str, default "manual_filter"
        Name of the binary filter variable to create or update in ``ds``.

    show : Iterable[str]
        Dataset variables to display in the montage (image or mask layers).

    ch : int or sequence of int
        Channel(s) to display for image layers.

    z_index : int
        Z-plane index if the dataset contains a ``z`` dimension.

    viewer : napari.Viewer, optional
        Existing viewer to reuse. If None, a new viewer is created.

    write_back : bool, default True
        If True, staged edits can be written into ``ds[filter_name]`` via
        the "Save/Update" UI control.

    overlay_opacity : float, default 0.35
        Opacity of the manual filter overlay.

    single_click_delay_ms : int, default 100
        Delay used to distinguish single-click from double-click actions.

    colormaps : dict, optional
        Mapping from layer name to napari colormap.

    label_colors : dict, optional
        Mapping from label integer value (e.g., 1) to display color.

    output_dir : str, optional
        Directory to write dataset when "Save to file" is pressed.
        If None, the original dataset location is used if available.

    show_click_info : bool, default False
        If True, display a debugging overlay showing clicked tile identity.

    show_tile_text : bool, default False
        If True, display per-tile text labels in the montage view.

    Returns
    -------
    viewer : napari.Viewer
        The napari viewer instance.

    layers : dict[str, Any]
        Mapping from layer names to napari layer objects.

    filter_table : xr.DataArray
        The staged binary filter table with dimensions
        (*context_dims, tile_dim, t). Edits are applied here until committed.

    Notes
    -----
    - The filter table is context-aware: additional dataset dimensions
      (e.g., exp, cell, fov) are preserved and editable via napari sliders.
    - Edits are staged in memory until "Save/Update" is pressed.
    - "Save to file" optionally overwrites the source dataset after confirmation.
    """
    from .plot import montage  # lazy import to avoid circulars

    # # Load any existing manual-filter sidecars affiliated with this dataset
    # _loaded_sidecars: list[str] = []
    # try:
    #     ds, _loaded_sidecars = load_manual_filter_sidecars(ds, output_dir=str(output_dir))
    # except Exception:
    #     _loaded_sidecars = []

    # # Fallback: if output_dir differs from source dir, also scan source dir
    # if not _loaded_sidecars:
    #     try:
    #         src = ds.encoding.get("source", None)
    #     except Exception:
    #         src = None
    #     if isinstance(src, str) and src.strip():
    #         src_dir = os.path.dirname(src)
    #         if src_dir and (os.path.abspath(src_dir) != os.path.abspath(str(output_dir))):
    #             try:
    #                 ds, _loaded_sidecars = load_manual_filter_sidecars(ds, output_dir=src_dir)
    #             except Exception:
    #                 _loaded_sidecars = []


    # Load any existing manual-filter sidecars affiliated with this dataset.
    # BUT: if open_* already merged the filter into ds, do NOT hit disk again.
    _loaded_sidecars: list[str] = []

    if filter_name not in ds:
        try:
            ds, _loaded_sidecars = load_manual_filter_sidecars(ds, output_dir=str(output_dir))
        except Exception:
            _loaded_sidecars = []

        # Fallback: if output_dir differs from source dir, also scan source dir
        if not _loaded_sidecars:
            try:
                src = ds.encoding.get("source", None)
            except Exception:
                src = None
            if isinstance(src, str) and src.strip():
                src_dir = os.path.dirname(src)
                if src_dir and (os.path.abspath(src_dir) != os.path.abspath(str(output_dir))):
                    try:
                        ds, _loaded_sidecars = load_manual_filter_sidecars(ds, output_dir=src_dir)
                    except Exception:
                        _loaded_sidecars = []



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
        colormaps=colormaps,     
        show_tile_text = show_tile_text,
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

    def _overlay_np_for_current_context() -> np.ndarray:
        """
        Build a labels array ONLY for the currently viewed context.
        Returns a small array aligned to the displayed montage (no exp/cell/rep dims).
        """
        ov = _overlay_pixels_from_filter_table(
            m=m,
            filter_table=_view_filter_table(),  # already context-sliced
            tile_dim=tile_dim,
            row=row,
            col=col,
            tile_h=tile_h,
            tile_w=tile_w,
        )

        # ov is typically (t, r, c) for square montages, or (r, c) for row=tile_dim,col='t'
        return np.asarray(ov.data, dtype=np.uint8)


    # Build initial overlay ONLY for current context
    overlay_np = _overlay_np_for_current_context()

    from napari.utils.colormaps import DirectLabelColormap

    # Reuse an existing Labels layer if present (avoid creating duplicates)
    if filter_name in viewer.layers:
        lbl_layer = viewer.layers[filter_name]
        lbl_layer.data = overlay_np.astype(np.uint8)
        lbl_layer.opacity = overlay_opacity
    else:
        ov_axis = ["t", "y", "x"] if overlay_np.ndim == 3 else ["y", "x"]
        lbl_layer = _add_layer_with_axis_labels(
            viewer,
            viewer.add_labels,
            overlay_np.astype(np.uint8),
            xr_dims=ov_axis,
            name=filter_name,
            opacity=overlay_opacity,
        )

    def _refresh_overlay_from_sliders(event=None):
        try:
            lbl_layer.data = _overlay_np_for_current_context()
            lbl_layer.refresh()
        except Exception:
            pass

    # When the user changes dims sliders (exp/cell/rep/etc.), rebuild overlay for that context
    viewer.dims.events.current_step.connect(_refresh_overlay_from_sliders)



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
    click_info_layer = None
    if show_click_info:
        click_info_layer = viewer.add_points(
            np.zeros((0, overlay_np.ndim), dtype=float),
            name="click_info",
            size=1.0,  # don't use 0; some napari versions are flaky with text-only points
        )
        # Hide point glyphs; text only
        try:
            click_info_layer.face_color = "transparent"
            click_info_layer.edge_color = "transparent"
        except Exception:
            pass

        click_info_layer.text = {
            "string": "{label}",
            "size": 12,
            "anchor": "center",
        }

        # Use features to back the "{label}" template
        click_info_layer.features = {"label": np.asarray([], dtype=object)}

        # Make sure clicks pass through to labels layer
        click_info_layer.interactive = False
        try:
            click_info_layer.editable = False
        except Exception:
            pass

        layers["click_info"] = click_info_layer


    # # Make label=1 visible
    # try:
    #     lbl_layer.color = {1: "yellow"}
    # except Exception:
    #     lbl_layer.color = {1: np.array([1.0, 1.0, 0.0, 1.0])}

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


    # Axes in the displayed label overlay (lbl_layer.data).
    # The overlay is either (t, r, c) or (r, c); spatial axes are always the last two.
    _lbl_ndim = int(getattr(getattr(lbl_layer, "data", None), "ndim", 0) or 0)
    t_axis = 0 if _lbl_ndim == 3 else None
    r_axis = max(0, _lbl_ndim - 2) if _lbl_ndim else 0
    c_axis = max(1, _lbl_ndim - 1) if _lbl_ndim else 1

    tile_index = {v: i for i, v in enumerate(filter_table.coords[tile_dim].values)}
    t_index = {v: i for i, v in enumerate(filter_table.coords["t"].values)}

    def _set_filter_value(tile_val, t_val, value: int):
        if tile_val in tile_index and t_val in t_index:
            filter_table.values[tile_index[tile_val], t_index[t_val]] = np.uint8(value)

    def _paint_tile_pixel(mr: int, mc: int, value: int, *, t_idx_for_pixels: int | None):
        r0, r1 = mr * tile_h, (mr + 1) * tile_h
        c0, c1 = mc * tile_w, (mc + 1) * tile_w

        data = lbl_layer.data  # now only (t,r,c) or (r,c)

        if data.ndim == 3:
            data[int(t_idx_for_pixels), r0:r1, c0:c1] = value
        else:
            data[r0:r1, c0:c1] = value

        lbl_layer.data = data
        lbl_layer.refresh()


    def _update_click_info(*, mr: int, mc: int, track_id: int | None, t_idx: int | None, y: float, x: float):
        if click_info_layer is None:
            return

        def _fmt_int(v):
            if v is None:
                return "None"
            try:
                return str(int(v))
            except Exception:
                try:
                    return str(v.item())
                except Exception:
                    return str(v)

        tid_str = _fmt_int(track_id)
        t_str = _fmt_int(t_idx)

        label = f"grid_rc=({mr},{mc})  track_id={tid_str}  t={t_str}"

        click_info_layer.features = {"label": np.asarray([label], dtype=object)}

        nd = click_info_layer.data.shape[1]
        pt = np.zeros((1, nd), dtype=float)
        if nd == 3:
            pt[0, 0] = 0 if t_idx is None else float(t_idx)
            pt[0, 1] = float(y)
            pt[0, 2] = float(x)
        else:
            pt[0, 0] = float(y)
            pt[0, 1] = float(x)

        click_info_layer.data = pt
        try:
            click_info_layer.refresh()
        except Exception:
            pass


    def _paint_row_pixels_from(mr: int, mc0: int, value: int, *, t_idx_for_pixels: int | None):
        mc0 = int(np.clip(mc0, 0, max(0, Mcol - 1)))
        for mc in range(mc0, Mcol):
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
                    
        if shift_down:
            # Determine "new" by looking at the REGION you will toggle.
            # Rule: if the whole region is currently 1 -> set 0, else set 1.
            if square:
                # Square montage (row==col==track_id): toggle ONE montage row from clicked mc onward.
                # Use the SAME mapping as Alt+click: track_id = ds.track_id[k] where k = mr*S + mc2.
                tile_coords = np.asarray(filter_table.coords[tile_dim].values)
                tile_to_i = {v: i for i, v in enumerate(tile_coords)}

                # Collect the tiles in this montage row slice (skip padding/out-of-range)
                tile_i_list: list[int] = []
                mc_list: list[int] = []
                for mc2 in range(int(mc), int(Mcol)):
                    tid = _track_id_orig_from_grid(mr=mr, mc=mc2)  # <-- key fix
                    if tid is None:
                        continue
                    if tid not in tile_to_i:
                        continue
                    tile_i_list.append(tile_to_i[tid])
                    mc_list.append(mc2)

                if not tile_i_list:
                    return  # nothing to do

                # Decide toggle target based on ALL tiles in the region at THIS t index
                cur_region = np.array([filter_table.data[_idx(ti, t_idx_for_table)] for ti in tile_i_list], dtype=np.uint8)
                new = np.uint8(0 if np.all(cur_region == 1) else 1)

                # Write: set ALL times for each selected tile (square mode has no time in grid)
                for ti in tile_i_list:
                    filter_table.data[_idx(ti, t_idx_for_table)] = new

                dirty = True

                # Paint efficiently: update lbl_layer.data in bulk (avoid per-tile refresh loops)
                data = lbl_layer.data
                r0, r1 = mr * tile_h, (mr + 1) * tile_h

                if data.ndim == 3:
                    tt = int(np.clip(int(t_idx_for_pixels if t_idx_for_pixels is not None else t_idx_for_table), 0, T - 1))
                    for mc2 in mc_list:
                        c0, c1 = mc2 * tile_w, (mc2 + 1) * tile_w
                        data[tt, r0:r1, c0:c1] = int(new)
                else:
                    # data is (r, c)
                    for mc2 in mc_list:
                        c0, c1 = mc2 * tile_w, (mc2 + 1) * tile_w
                        data[r0:r1, c0:c1] = int(new)

                lbl_layer.data = data
                lbl_layer.refresh()
                return

            else:
                # Rectangular montage (row=tile_dim, col=t): toggle this row from clicked column onward.
                # Here columns already represent t, so we only change t >= clicked t.
                # tile_val is the row's tile_dim coordinate value (e.g. track_id)
                # tile_i is already computed above for tile_val
                cur_region = np.asarray([filter_table.data[_idx(tile_i, tt)] for tt in range(t_idx_for_table, T)], dtype=np.uint8)
                new = np.uint8(0 if (cur_region.size > 0 and np.all(cur_region == 1)) else 1)

                for tt in range(t_idx_for_table, T):
                    filter_table.data[_idx(tile_i, tt)] = new

                dirty = True

                # Rect overlay has no t-axis; each montage column corresponds to a specific t already.
                _paint_row_pixels_from(mr, mc, int(new), t_idx_for_pixels=None)
                return

        # --- Normal click: toggle single (tile, t) in this context ---
        cur = int(filter_table.data[_idx(tile_i, t_idx_for_table)])
        new = 0 if cur == 1 else 1
        filter_table.data[_idx(tile_i, t_idx_for_table)] = np.uint8(new)
        dirty = True

        _paint_tile_pixel(mr, mc, new, t_idx_for_pixels=t_idx_for_pixels)

    dirty = False  # staged edits not yet committed to ds

    def _write_filter_table_back():
        nonlocal dirty, filter_table_committed
        # Commit staged table to ds
        filter_table_u8 = filter_table.astype(np.uint8)
        ds[filter_name] = filter_table_u8
        # Refresh committed snapshot (used by "Clear staged" = revert)
        filter_table_committed = filter_table_u8.copy(deep=True)
        dirty = False

    def _clear_filter_table():
        """Revert staged edits back to last-committed state (does not modify ds)."""
        nonlocal dirty, filter_table_committed

        ctx = _current_context_isel()  # indices for context dims (e.g. fov/exp/cell)
        if ctx:
            ft_ctx = {d: int(i) for d, i in ctx.items() if d in filter_table.dims}
        else:
            ft_ctx = {}

        if ft_ctx:
            sl = [slice(None)] * filter_table.ndim
            for d, i in ft_ctx.items():
                sl[filter_table.dims.index(d)] = int(i)
            sl = tuple(sl)

            # Restore staged slice from committed snapshot
            filter_table.data[sl] = filter_table_committed.data[sl]
        else:
            # No context dims -> restore entire staging table
            filter_table.data[...] = filter_table_committed.data[...]

        dirty = False  # optional: you just reverted staged to committed, so no longer dirty

        lbl_layer.data = _overlay_np_for_current_context().astype(np.uint8)
        lbl_layer.refresh()
        return

    def _tile_id_orig_from_m(*, mr: int, mc: int):
        """Return original tile id for montage position (mr,mc) using m.coords['tile_id'] if present."""
        try:
            grid = m.coords.get("tile_id", None)
            if grid is None:
                return None
            val = grid.values[mr, mc] if hasattr(grid, "values") else grid[mr, mc]
            return val
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

        # If a second click arrives before the delayed single-click fires,
        # execute the pending action immediately (do NOT cancel to "do nothing").
        if QTimer is not None and click_state.get("pending", False):
            try:
                click_state["timer"].stop()
            except Exception:
                pass

            pl = click_state.get("payload", None)
            click_state["pending"] = False
            click_state["payload"] = None

            if pl is not None:
                _apply_toggle_or_row(pl)

            # consume this event so we don't double-toggle
            yield
            return

        # Convert click position to *this layer's* data coordinates.
        # This avoids assumptions about viewer dims / context dims.
        data_pos = layer.world_to_data(event.position)

        # Last two axes are spatial.
        r_axis = len(data_pos) - 2
        c_axis = len(data_pos) - 1

        # Clip to the layer's data shape
        shape = layer.data.shape
        ry = int(np.clip(int(round(float(data_pos[r_axis]))), 0, shape[-2] - 1))
        cx = int(np.clip(int(round(float(data_pos[c_axis]))), 0, shape[-1] - 1))

        # For click-info/debug overlays
        y = float(ry)
        x = float(cx)

        # Tile indices in montage grid
        mr = int(np.clip(ry // tile_h, 0, max(0, Mrow - 1)))
        mc = int(np.clip(cx // tile_w, 0, max(0, Mcol - 1)))

        mods = set(getattr(event, "modifiers", ()) or ())
        alt_down = ("Alt" in mods) or ("Option" in mods)   # Option sometimes on mac
        shift_down = ("Shift" in mods)

        # ---- Identity from montage coordinate mappings (tile-level) ----
        row_val = tile_row_id_1d[mr] if tile_row_id_1d is not None else mr
        col_val = tile_col_id_1d[mc] if tile_col_id_1d is not None else mc

        T = int(filter_table.sizes["t"])

        # ---- Determine which time index we are editing ----
        if square:
            # Square montage: time is not encoded in the grid; use the layer's t axis if present
            if getattr(layer.data, "ndim", 0) == 3 and len(data_pos) >= 3:
                t_idx_for_table = int(round(float(data_pos[0])))
            else:
                t_idx_for_table = 0
            t_idx_for_table = int(np.clip(t_idx_for_table, 0, max(0, T - 1)))

            # Only index a t-plane in pixels if the overlay layer actually has a t axis
            t_idx_for_pixels = t_idx_for_table if getattr(lbl_layer.data, "ndim", 0) == 3 else None
        else:
            # Rectangular row=track_id, col=t: time is encoded in montage columns.
            t_idx_for_table = int(t_index.get(col_val, 0))
            t_idx_for_table = int(np.clip(t_idx_for_table, 0, max(0, T - 1)))
            t_idx_for_pixels = None  # rectangular overlay is (r,c)

        # ---- Determine tile identity (tile_val) and a canonical track_id for debug ----
        track_id_orig = None

        if square and (row == col == "track_id"):
            # YOUR intended mapping: track_id = ds.track_id[k] where k = mr*S + mc
            track_id_orig = _track_id_orig_from_grid(mr=mr, mc=mc)
            if track_id_orig is None:
                yield
                return
            tile_val = track_id_orig
        else:
            # For rectangular track_id×t montages, row_val is the track_id coord value
            if (not square) and (row == "track_id"):
                track_id_orig = int(row_val)
                tile_val = track_id_orig
            else:
                # Generic fallback: use tile_id_grid if present
                if tile_id_grid is None:
                    yield
                    return
                try:
                    tile_val = tile_id_grid[mr, mc]
                except Exception:
                    yield
                    return

        # reject padded tiles
        try:
            if tile_val is None or (isinstance(tile_val, float) and np.isnan(tile_val)):
                yield
                return
        except Exception:
            pass

        # ---- Require a modifier for editing; plain click is navigation ----
        if not (shift_down or alt_down):
            # still show click info even for navigation clicks (optional)
            if show_click_info:
                _update_click_info(
                    mr=mr,
                    mc=mc,
                    track_id=track_id_orig,
                    t_idx=t_idx_for_table,
                    y=y,
                    x=x,
                )
            yield
            return

        # Optional: update click-info overlay
        if show_click_info:
            _update_click_info(
                mr=mr,
                mc=mc,
                track_id=track_id_orig,
                t_idx=t_idx_for_table,
                y=y,
                x=x,
            )

        payload = (mr, mc, shift_down, t_idx_for_pixels, tile_val, t_idx_for_table)

        # Shift+click should feel instantaneous (no single-click delay)
        if shift_down:
            _apply_toggle_or_row(payload)
            yield
            return

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
    from napari.utils.notifications import show_info, show_error
    from qtpy.QtCore import Qt

    # --- Upper-right: Help + Save/Update ---
    w_top = QWidget()
    layout_top = QVBoxLayout(w_top)

    help_lbl = QLabel(
        "In the manual filter layer:\n"
        "- Alt+Click: Select crop (toggles)\n"
        "- Shift+Click: Select row from column forward (toggles)\n"
        "- Press 'Save/Update' to write into ds\n"
        "- Press 'Clear' to reset staged edits\n"
    )
    help_lbl.setWordWrap(True)

    btn_save = QPushButton("Save/Update (write to ds)")

    def _on_save():
        _write_filter_table_back()

    btn_save.clicked.connect(_on_save)

    layout_top.addWidget(help_lbl)
    layout_top.addWidget(btn_save)

    dock_top = viewer.window.add_dock_widget(
        w_top, name=f"{filter_name} help", area="right"
    )

    # --- Lower-right: Clear + Save to file ---
    w_bottom = QWidget()
    layout_bottom = QVBoxLayout(w_bottom)

    btn_clear = QPushButton("Revert staged to saved")

    def _on_clear():
        _clear_filter_table()

    from napari.utils.notifications import show_info, show_error
    from qtpy.QtWidgets import QMessageBox, QLineEdit, QLabel


    def _resolve_out_dir() -> str:
        if output_dir is not None:
            return str(output_dir)
        try:
            src = ds.encoding.get("source", None)
        except Exception:
            src = None
        if isinstance(src, str) and src.strip():
            return os.path.dirname(src)
        return os.getcwd()


    out_dir = _resolve_out_dir()
    _default_fn = os.path.basename(
        _manual_filter_sidecar_path(ds, filter_name=filter_name, output_dir=out_dir)
    )
    layout_bottom.addWidget(QLabel("Manual filter file (output_dir/filename):"))
    # le_filename = QLineEdit(_default_fn)
    # layout_bottom.addWidget(le_filename)
    le_filename = QLineEdit(_default_fn)
    le_filename.setReadOnly(True)     # user can’t type
    # le_filename.setEnabled(False)     # (optional) visually indicates locked
    layout_bottom.addWidget(le_filename)


    btn_save_file = QPushButton("Save to file (overwrite)")

    def _on_save_file():
        try:
            # Commit staged edits into ds[filter_name]
            _write_filter_table_back()

            out_dir = _resolve_out_dir()
            os.makedirs(out_dir, exist_ok=True)

            # --- compute standardized sidecar path WITHOUT writing yet ---
            # IMPORTANT: don't call save_manual_filter_sidecar() until after overwrite confirmation
            try:
                # If you added the helper I suggested earlier:
                path = _manual_filter_sidecar_path(ds, filter_name=filter_name, output_dir=out_dir)
            except NameError:
                # Fallback if you haven't added _manual_filter_sidecar_path:
                # replicate the current save_manual_filter_sidecar naming convention as needed
                # (but ideally add _manual_filter_sidecar_path so this stays single-source-of-truth)
                path = os.path.join(out_dir, f"manual_filter__{filter_name}.nc")

            # Confirm overwrite if exists (same UX as current warning)
            if os.path.exists(path):
                reply = QMessageBox.question(
                    viewer.window._qt_window,
                    "Overwrite file?",
                    f"This will overwrite:\n\n{path}\n\nAre you sure?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

            # --- now actually write the sidecar ---
            # Ensure save_manual_filter_sidecar writes to exactly `path`
            # by passing output_dir and relying on standardized naming.
            # If save_manual_filter_sidecar returns a path, trust it and show it.
            saved_path = save_manual_filter_sidecar(ds, filter_name=filter_name, output_dir=out_dir)

            # Optional sanity check: warn if the helper-path and returned-path diverge
            # (useful while you standardize naming)
            try:
                if os.path.abspath(saved_path) != os.path.abspath(path):
                    show_info(f"Saved manual filter sidecar:\n{saved_path}\n\n(note: expected path was:\n{path})")
                else:
                    show_info(f"Saved manual filter sidecar:\n{saved_path}")
            except Exception:
                show_info(f"Saved manual filter sidecar:\n{saved_path}")

        except Exception as e:
            show_error(f"Failed to save manual filter sidecar:\n{e}")


    btn_clear.clicked.connect(_on_clear)
    btn_save_file.clicked.connect(_on_save_file)

    layout_bottom.addWidget(btn_clear)
    layout_bottom.addWidget(btn_save_file)

    dock_bottom = viewer.window.add_dock_widget(
        w_bottom, name=f"{filter_name} actions", area="right"
    )

    # --- Force vertical split: top above bottom ---
    try:
        viewer.window._qt_window.splitDockWidget(dock_top, dock_bottom, Qt.Vertical)
    except Exception:
        # If splitDockWidget isn't available in this napari build,
        # it will still usually stack in insertion order.
        pass


    return viewer, layers, filter_table
