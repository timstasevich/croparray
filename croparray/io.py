
import xarray as xr


def open_croparray(path: str, as_object: bool = True, **kwargs):
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

    if as_object:
        # Local import avoids circular dependency
        from .crop_array_object import CropArray
        return CropArray(ds)

    return ds




def open_croparray_zarr(store: str, as_object: bool = True, **kwargs):
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
    ca = open_croparray(path, as_object=True, **kwargs)

    if drop_vars:
        ca.ds = ca.ds.drop_vars(list(drop_vars), errors=drop_errors)

    # Local import avoids circular dependency
    from . import crop_array_tools

    ta = crop_array_tools.track_array(ca, as_object=as_object)

    # Help GC in large pipelines
    del ca
    return ta
