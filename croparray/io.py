from __future__ import annotations

from typing import Any, Literal, overload, Union
import xarray as xr
from .crop_array_object import CropArray
import os
from pathlib import Path
from .trackarray.object import TrackArray
import re
import json

from .crop_array_object import CropArray
from .trackarray.object import TrackArray


__all__ = ["save_croparray", "open_trackarray", "open_as_trackarray", "open_croparray", "open_croparray_zarr"]

def _manual_filter_sidecar_paths(path_nc: str) -> list[str]:
    """
    Return all per-filter manual sidecars for a given dataset path.

    For /path/mydata.nc, matches:
      /path/mydata__manual__*.nc
    """
    p = Path(path_nc)
    if not p.exists():
        return []
    stem = p.stem  # "mydata" from "mydata.nc"
    pattern = f"{stem}__manual__*.nc"
    return sorted(str(x) for x in p.parent.glob(pattern))

def _merge_manual_filter_sidecars_crop(ds: xr.Dataset, path_nc: str) -> xr.Dataset:
    """Merge manual filter sidecars into a CropArray dataset."""
    sidecars = _manual_filter_sidecar_paths(path_nc)

    if not sidecars:
        print("[CropArray] No sidecars found.")
        return ds

    print(f"[CropArray] Found {len(sidecars)} sidecar(s).")

    to_merge: list[xr.Dataset] = [ds]

    for sc in sidecars:
        print(f"[CropArray] Checking sidecar: {sc}")

        try:
            with xr.open_dataset(sc) as _tmp:
                ds_sc = _tmp.load()
        except Exception as e:
            print(f"[CropArray] Failed to load {sc}: {e}")
            continue

        if len(ds_sc.data_vars) == 0:
            print(f"[CropArray] Skipping {sc} (no data variables).")
            continue

        print(f"[CropArray] Merging sidecar {sc} with vars: {list(ds_sc.data_vars)}")
        to_merge.append(ds_sc)

    if len(to_merge) == 1:
        print("[CropArray] No valid sidecars merged.")
        return ds

    print("[CropArray] Merging sidecars into CropArray dataset.")
    return xr.merge(to_merge, compat="override", join="outer")

def _merge_manual_filter_sidecars_track(ds: xr.Dataset, path_nc: str) -> xr.Dataset:
    """Merge only sidecars that look like TrackArray-space filters (have 'track_id' dim)."""
    sidecars = _manual_filter_sidecar_paths(path_nc)
    if not sidecars:
        return ds

    to_merge: list[xr.Dataset] = [ds]
    for sc in sidecars:
        try:
            with xr.open_dataset(sc) as _tmp:
                ds_sc = _tmp.load()
        except Exception:
            continue

        # Only take track-space sidecars here
        if "track_id" not in ds_sc.dims:
            continue

        if len(ds_sc.data_vars) == 0:
            continue

        to_merge.append(ds_sc)

    if len(to_merge) == 1:
        return ds

    return xr.merge(to_merge, compat="override", join="outer")

@overload
def open_croparray(path: str, *, as_object: Literal[True] = True, **kwargs: Any) -> CropArray: ...
@overload
def open_croparray(path: str, *, as_object: Literal[False], **kwargs: Any) -> xr.Dataset: ...

def open_croparray(path: str, *, as_object: bool = True, load_manual_filters: bool = True, **kwargs: Any) -> CropArray | xr.Dataset:
    """
    Open a saved CropArray dataset and optionally wrap it as a CropArray object.

    This function uses ``xarray.open_dataset`` and is suitable for CropArrays
    stored in NetCDF or other formats supported by xarray. For large or
    chunked datasets, consider using :func:`open_croparray_zarr`.

    Parameters
    ----------
    path : str
        Path to a dataset readable by ``xarray.open_dataset`` (e.g. NetCDF).
    as_object : bool, default True
        If True, return a ``CropArray`` wrapper providing the method-style API
        (e.g. ``ca.best_z_proj()``, ``ca.measure_signal()``).
        If False, return the raw ``xarray.Dataset``.
    **kwargs
        Additional keyword arguments passed directly to
        ``xarray.open_dataset`` (e.g. ``engine``, ``chunks``).

    Returns
    -------
    CropArray or xarray.Dataset
        If ``as_object=True``, returns a ``CropArray`` wrapping the opened
        dataset. Otherwise, returns the underlying ``xarray.Dataset``.

    Notes
    -----
    Datasets opened with ``open_croparray`` may be loaded eagerly or lazily
    depending on the file format and the arguments passed to
    ``xarray.open_dataset``.

    Examples
    --------
    Open a CropArray from a NetCDF file and compute best-z projections::

        from croparray import open_croparray

        ca = open_croparray("my_croparray.nc")
        ca.best_z_proj()
        ca.measure_signal()

    Open the raw Dataset without wrapping::

        ds = open_croparray("my_croparray.nc", as_object=False)
    """
    ds = xr.open_dataset(path, **kwargs)
    if load_manual_filters:
        ds = _merge_manual_filter_sidecars_crop(ds, path)


    if as_object:
        # Local import avoids circular dependency
        return CropArray(ds)

    return ds

@overload
def open_croparray_zarr(store: str, *, as_object: Literal[True] = True, **kwargs: Any) -> CropArray: ...
@overload
def open_croparray_zarr(store: str, *, as_object: Literal[False], **kwargs: Any) -> xr.Dataset: ...

def open_croparray_zarr(store: str, *, as_object: bool = True, **kwargs: Any) -> CropArray | xr.Dataset:
    """
    Open a saved CropArray stored in Zarr format and optionally wrap it as a
    CropArray object.

    This function mirrors :func:`open_croparray`, but uses
    ``xarray.open_zarr`` instead of ``xarray.open_dataset``. It is intended
    for large crop arrays that benefit from chunked, lazy loading.

    Parameters
    ----------
    store : str
        Path to the Zarr store (directory or consolidated Zarr archive).
    as_object : bool, default True
        If True, return a ``CropArray`` wrapper providing the method-style API
        (e.g. ``ca.best_z_proj()``, ``ca.measure_signal()``).
        If False, return the raw ``xarray.Dataset``.
    **kwargs
        Additional keyword arguments passed directly to
        ``xarray.open_zarr`` (e.g. ``consolidated=True``).

    Returns
    -------
    CropArray or xarray.Dataset
        If ``as_object=True``, returns a ``CropArray`` wrapping the opened
        dataset. Otherwise, returns the underlying ``xarray.Dataset``.

    Notes
    -----
    Zarr-backed CropArrays are loaded lazily; data are read from disk only
    when required for computation. This makes Zarr the preferred storage
    format for large or multi-FOV crop arrays.

    Examples
    --------
    Open a Zarr-backed CropArray and compute best-z projections::

        from croparray import open_croparray_zarr

        ca = open_croparray_zarr("my_croparray.zarr")
        ca.best_z_proj()
        ca.measure_signal()

    Open the raw Dataset without wrapping::

        ds = open_croparray_zarr("my_croparray.zarr", as_object=False)
    """
    ds = xr.open_zarr(store, **kwargs)

    if as_object:
        from .crop_array_object import CropArray
        return CropArray(ds)

    return ds

def open_as_trackarray(
    path: str,
    *,
    drop_vars=("int",),
    drop_errors: str = "ignore",
    as_object: bool = True,
    load_manual_filters: bool = True,
    **kwargs,
):
    """
    Open a CropArray dataset and immediately convert it to a TrackArray.

    Parameters
    ----------
    path : str
        Path to a dataset readable by ``open_croparray`` (e.g. NetCDF).
    drop_vars : sequence of str or None, default ("int",)
        Variables to drop from the underlying Dataset before track conversion.
        Set to None or empty to disable dropping.
    drop_errors : {"ignore","raise"}, default "ignore"
        Passed to ``Dataset.drop_vars``.
    as_object : bool, default True
        If True, return a TrackArray wrapper. If False, return the raw Dataset.
    **kwargs
        Additional keyword arguments forwarded to ``open_croparray``.

    Returns
    -------
    TrackArray or xarray.Dataset
    """
    # Always open as CropArray internally
    ca = open_croparray(
        path,
        as_object=True,
        load_manual_filters=load_manual_filters,
        **kwargs,
    )

    # --- backward-compat / schema guard ---
    if "track_id" not in ca.ds:
        raise KeyError(
            "Cannot convert to TrackArray: dataset has no 'track_id'.\n"
            "This file was likely created with an older croparray version where "
            "'id' encoded track labels.\n\n"
            "Fix options:\n"
            "  • Add ca['track_id'] manually (-1 = untracked)\n"
            "  • Or re-run tracking with to_track_array()\n"
        )

    if drop_vars:
        ca.ds = ca.ds.drop_vars(list(drop_vars), errors=drop_errors)

    # Local import avoids circular dependency
    from . import crop_array_tools

    ta = crop_array_tools.track_array(ca, as_object=as_object)
    if load_manual_filters:
    # ta may be TrackArray wrapper or raw dataset depending on as_object
        if hasattr(ta, "ds"):
            ta.ds = _merge_manual_filter_sidecars_track(ta.ds, path)
        else:
            ta = _merge_manual_filter_sidecars_track(ta, path)


    # Help GC in large pipelines
    del ca
    return ta

@overload
def open_trackarray(
    path: str,
    *,
    as_object: Literal[True] = True,
    load_manual_filters: bool = True,
    **kwargs: Any,
) -> TrackArray: ...
@overload
def open_trackarray(
    path: str,
    *,
    as_object: Literal[False],
    load_manual_filters: bool = True,
    **kwargs: Any,
) -> xr.Dataset: ...
def open_trackarray(
    path: str,
    *,
    as_object: bool = True,
    load_manual_filters: bool = True,
    **kwargs: Any,
) -> TrackArray | xr.Dataset:
    ds = xr.open_dataset(path, **kwargs)

    if "track_id" not in ds.dims:
        raise ValueError(
            f"{path} does not look like a TrackArray dataset (missing 'track_id' dim). "
            "Use `open_croparray()` or `open_as_trackarray()` instead."
        )

    if load_manual_filters:
        ds = _merge_manual_filter_sidecars_track(ds, path)

    if as_object:
        return TrackArray(ds)

    return ds

def _slugify_filename(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "croparray"
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    s = s.strip("._-")
    return s or "croparray"

def _concat_ultraminimal_stem(ds: xr.Dataset) -> str | None:
    meta_json = ds.attrs.get("concat_meta_json", None)
    if not isinstance(meta_json, str) or not meta_json.strip():
        return None
    try:
        meta = json.loads(meta_json)
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None

    base = str(meta.get("base_name", "")).strip()
    if not base:
        return None

    try:
        n_files = int(meta.get("n_files", 0))
    except Exception:
        return None

    dims = meta.get("dims", [])
    if not isinstance(dims, (list, tuple)):
        dims = []

    base_s = _slugify_filename(base)
    dims_s = "_".join(_slugify_filename(str(d)) for d in dims if str(d).strip())
    if dims_s:
        return f"{base_s}__{n_files}{dims_s}"
    return f"{base_s}__{n_files}"

def save_croparray(
    obj: xr.Dataset,
    *,
    output_dir: str | Path,
    filename: str | None = None,
    ext: str = ".nc",
    overwrite: bool = False,
    mkdir: bool = True,
    to_netcdf_kwargs: dict[str, Any] | None = None,
) -> Path: ...
def save_croparray(
    obj: CropArray | TrackArray | xr.Dataset,
    *,
    output_dir: str | Path,
    filename: str | None = None,
    ext: str = ".nc",
    overwrite: bool = False,
    mkdir: bool = True,
    to_netcdf_kwargs: dict[str, Any] | None = None,
) -> Path:
    """
    Save a CropArray/TrackArray (or raw xarray.Dataset) to NetCDF.

    If `filename` is not provided, uses ds.attrs["name"] (slugified) as the stem.

    Parameters
    ----------
    obj
        CropArray, TrackArray, or xarray.Dataset. If object has `.ds`, that dataset is saved.
    output_dir
        Directory to write into.
    filename
        Optional filename. If None, derives from ds.attrs["name"].
        If provided without suffix, `ext` is appended.
    ext
        Default ".nc".
    overwrite
        If False (default), refuse to overwrite.
    mkdir
        If True (default), create output_dir if needed.
    to_netcdf_kwargs
        Extra kwargs passed to `Dataset.to_netcdf`.

    Returns
    -------
    pathlib.Path
        The written file path.
    """
    ds: xr.Dataset = obj.ds if hasattr(obj, "ds") else obj
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"save_croparray expected CropArray/TrackArray/xr.Dataset, got {type(obj)!r}")

    out_dir = Path(output_dir)
    if mkdir:
        out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        stem = _concat_ultraminimal_stem(ds)
        if stem is None:
            # fallback: slugify the human-readable name
            stem = _slugify_filename(str(ds.attrs.get("name", "")).strip())
        filename = stem
    else:
        filename = str(filename).strip()

    # Ensure filename is just a name, not a path
    filename = os.path.basename(filename)

    # Always ensure filename ends with ext
    if not filename.lower().endswith(ext.lower()):
        filename = filename + ext
        
    out_path = out_dir / filename

    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Refusing to overwrite. Set overwrite=True to replace it."
        )

    # Base kwargs
    kwargs: dict[str, Any] = {}
    if to_netcdf_kwargs:
        kwargs.update(to_netcdf_kwargs)

    # Force correct mode:
    # - first save (file missing) must be "w"
    # - overwrite=True should be "w"
    if (not out_path.exists()) or overwrite:
        kwargs["mode"] = "w"
    else:
        # Should never hit because of the exists+not overwrite guard above,
        # but keep it explicit.
        kwargs.setdefault("mode", "w")

    # Ensure output directory exists (even if mkdir=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Work on a shallow copy so we don't mutate the in-memory object
    ds_to_save = ds.copy(deep=False)

    # Store filename metadata (NetCDF-safe string)
    ds_to_save.attrs["filename"] = out_path.name

    ds_to_save.to_netcdf(str(out_path), **kwargs)

    return out_path
