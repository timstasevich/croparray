import numpy as np
import xarray as xr

__all__ = [
    # ... existing ...
    "tracklist",
]

def tracklist(
    ta,
    *,
    var: str | None = None,
    min_count: int = 1,
    return_mask: bool = False,
):
    """
    List track_id values that have any non-null data (or >= min_count non-nulls)
    in this TrackArray dataset.

    Parameters
    ----------
    ta : xarray.Dataset
        TrackArray dataset (must contain dimension/coordinate 'track_id').
    var : str or None
        If provided, use only this variable to determine whether a track is present.
        If None, consider all variables that include 'track_id'.
    min_count : int
        Minimum number of non-null values required to keep a track (default 1).
    return_mask : bool
        If True, return a boolean DataArray over 'track_id' instead of the id list.

    Returns
    -------
    np.ndarray or xarray.DataArray
    """
    if "track_id" not in ta.dims and "track_id" not in ta.coords:
        raise ValueError("tracklist: expected 'track_id' in dataset")

    track_coord = ta["track_id"] if "track_id" in ta.coords else xr.DataArray(
        np.arange(ta.sizes["track_id"]), dims=("track_id",)
    )

    def _mask(da: xr.DataArray) -> xr.DataArray:
        if "track_id" not in da.dims:
            return xr.zeros_like(track_coord, dtype=bool)
        reduce_dims = [d for d in da.dims if d != "track_id"]
        cnt = da.notnull().sum(dim=reduce_dims) if reduce_dims else da.notnull().astype(int)
        return cnt >= int(min_count)

    if var is not None:
        if var not in ta:
            raise KeyError(f"tracklist: var={var!r} not found. Available: {list(ta.data_vars)}")
        present = _mask(ta[var])
    else:
        present = None
        for da in ta.data_vars.values():
            if "track_id" not in da.dims:
                continue
            m = _mask(da)
            present = m if present is None else (present | m)
        if present is None:
            present = xr.zeros_like(track_coord, dtype=bool)

    if return_mask:
        return present

    return track_coord.values[present.values]
