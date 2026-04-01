import numpy as np
import xarray as xr

__all__ = [
    # ... existing ...
    "tracklist","track_length","step_size", "msd"
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

        valid = da.notnull() & (da != 0)

        cnt = valid.sum(dim=reduce_dims) if reduce_dims else valid.astype(int)
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

def track_length(
    ta,
    *,
    coord: str = "xc",
    out_name: str = "track_length",
    broadcast_like: str | None = "signal",
):
    """
    Compute per-track length as the number of timepoints where the track exists,
    independent of fluorescence intensity.

    Track existence is defined by `coord.notnull()`. By croparray convention,
    coordinates such as 'xc' are NaN at timepoints where no detection exists
    for that track.
    """
    # Accept either TrackArray wrapper or raw xarray.Dataset (generated accessors pass Dataset)
    ds = ta.ds if hasattr(ta, "ds") else ta

    if coord not in ds:
        raise KeyError(f"'{coord}' not found in dataset")

    present = ds[coord].notnull()
    track_length_track = present.sum(dim="t")

    if broadcast_like is not None and broadcast_like in ds:
        ds[out_name] = track_length_track.broadcast_like(ds[broadcast_like])
    else:
        ds[out_name] = track_length_track.broadcast_like(present)

    ds[out_name].attrs.update(
        {
            "long_name": "track length (timepoints present)",
            "source": coord,
            "definition": f"{coord}.notnull().sum(dim='t')",
            "units": "frames",
        }
    )

    return ta

def msd(
    self,
    xvar: str = "xc",
    yvar: str = "yc",
    zvar: str | None = None,
    name: str = "MSD",
    space: str = "units",
):
    """
    Compute mean squared displacement (MSD) vs lag for each track.

    Returns a DataArray with the same non-time dims as the position variables
    and dimension 't', where 't' is interpreted as lag time.

    Parameters
    ----------
    xvar, yvar : str
        Position variables, usually 'xc' and 'yc'.
    zvar : str or None
        Optional z position variable.
    name : str
        Name of returned DataArray.
    space : {'units', 'pixels'}
        If 'units', convert coordinates using dx before computing MSD.
        If 'pixels', use raw coordinate values.
    """
    import xarray as xr

    ds = self._obj if hasattr(self, "_obj") else self.ds if hasattr(self, "ds") else self

    if xvar not in ds or yvar not in ds:
        raise ValueError(f"Dataset must contain '{xvar}' and '{yvar}'")

    if space not in {"units", "pixels"}:
        raise ValueError("space must be 'units' or 'pixels'")

    x = ds[xvar]
    y = ds[yvar]
    z = ds[zvar] if zvar is not None else None

    if zvar is not None and zvar not in ds:
        raise ValueError(f"Dataset must contain '{zvar}'")

    if space == "units":
        if not hasattr(self, "dx"):
            raise ValueError("space='units' requested, but object has no attribute 'dx'")
        x = x * self.dx
        y = y * self.dx
        if z is not None:
            z = z * self.dx
        base_units = getattr(self, "xyz_units", None)
    else:
        base_units = "px"

    pos = [x, y]
    dim_label = "2D"

    if z is not None:
        pos.append(z)
        dim_label = "3D"

    n_t = ds.sizes["t"]
    msd_list = []

    for lag in range(n_t):
        diffsq = 0
        for p in pos:
            d = p.shift(t=-lag) - p
            diffsq = diffsq + d**2

        msd_lag = diffsq.isel(t=slice(0, n_t - lag)).mean("t", skipna=True)
        msd_list.append(msd_lag)

    out = xr.concat(msd_list, dim=ds["t"])
    out = out.rename(name)

    out.attrs["long_name"] = f"{dim_label} mean squared displacement"
    out.attrs["description"] = "MSD vs lag time; coordinate 't' is interpreted as lag"
    out.attrs["space"] = space

    if base_units is not None:
        out.attrs["units"] = f"{base_units}^2"

    return out

def step_size(
    self,
    xvar: str = "xc",
    yvar: str = "yc",
    zvar: str | None = None,
    name: str = "step_size",
    space: str = "units",
):
    """
    Compute frame-to-frame step size and return as a DataArray.

    Step size at time t is the Euclidean distance between positions at t and t-1.
    The first timepoint is NaN.

    Parameters
    ----------
    xvar, yvar : str
        Position variables, usually 'xc' and 'yc'.
    zvar : str or None
        Optional z position variable.
    name : str
        Name of returned DataArray.
    space : {'units', 'pixels'}
        If 'units', convert coordinates using dx before computing step size.
        If 'pixels', use raw coordinate values.
    """
    import numpy as np

    ds = self._obj if hasattr(self, "_obj") else self.ds if hasattr(self, "ds") else self

    if xvar not in ds or yvar not in ds:
        raise ValueError(f"Dataset must contain '{xvar}' and '{yvar}'")

    if space not in {"units", "pixels"}:
        raise ValueError("space must be 'units' or 'pixels'")

    x = ds[xvar]
    y = ds[yvar]
    z = ds[zvar] if zvar is not None else None

    if zvar is not None and zvar not in ds:
        raise ValueError(f"Dataset must contain '{zvar}'")

    if space == "units":
        if not hasattr(self, "dx"):
            raise ValueError("space='units' requested, but object has no attribute 'dx'")
        x = x * self.dx
        y = y * self.dx
        if z is not None:
            z = z * self.dx
        units = getattr(self, "xyz_units", None)
    else:
        units = "px"

    dx = x.diff("t")
    dy = y.diff("t")

    if z is not None:
        dz = z.diff("t")
        out = np.sqrt(dx**2 + dy**2 + dz**2)
        long_name = "3D step size"
    else:
        out = np.sqrt(dx**2 + dy**2)
        long_name = "2D step size"

    out = out.reindex(t=ds.t)
    out.name = name
    out.attrs["long_name"] = long_name
    if units is not None:
        out.attrs["units"] = units
    out.attrs["space"] = space

    return out