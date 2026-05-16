from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = [
    # ... existing ...
    "tracklist","track_length","step_size", "msd", "autocorr",
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

    da = ds[coord]
    if "ch" in da.dims:
        da = da.isel(ch=0)
    present = da.notnull()
    track_length_track = present.sum(dim="t")
    ds[out_name] = track_length_track

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
    drift_correct: bool = False,
    drift_dims: tuple[str, ...] | None = None,
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
        If 'units', convert coordinates using pixel sizes before computing MSD.
        If 'pixels', use raw coordinate values.
    drift_correct : bool
        If True, subtract per-timepoint field drift estimated from the mean
        position of all spots in each FOV.
    drift_dims : tuple[str, ...] or None
        Dimensions over which to average when estimating drift. If None,
        defaults to ('track_id',) when present, otherwise all dims except
        't', 'exp', and 'fov'.

    Notes
    -----
    Drift correction is applied separately within each FOV because the mean
    is not taken over 'exp' or 'fov'. The estimated drift trace is simply
    subtracted from each coordinate; any constant offset is irrelevant for MSD.
    """
    import xarray as xr

    ds = self._obj if hasattr(self, "_obj") else self.ds if hasattr(self, "ds") else self

    if xvar not in ds or yvar not in ds:
        raise ValueError(f"Dataset must contain '{xvar}' and '{yvar}'")

    if zvar is not None and zvar not in ds:
        raise ValueError(f"Dataset must contain '{zvar}'")

    if space not in {"units", "pixels"}:
        raise ValueError("space must be 'units' or 'pixels'")

    x = ds[xvar]
    y = ds[yvar]
    z = ds[zvar] if zvar is not None else None

    # unit conversion
    if space == "units":
        dx = getattr(self, "dx", None)
        dy = getattr(self, "dy", dx)
        dz = getattr(self, "dz", dx)

        if dx is None:
            raise ValueError("space='units' requested, but object has no attribute 'dx'")

        x = x * dx
        y = y * dy
        if z is not None:
            z = z * dz

        base_units = getattr(self, "xyz_units", None)
    else:
        base_units = "px"

    # optional drift correction
    if drift_correct:
        if drift_dims is None:
            if "track_id" in x.dims:
                drift_dims = ("track_id",)
            else:
                drift_dims = tuple(d for d in x.dims if d not in {"t", "exp", "fov"})

        missing = [d for d in drift_dims if d not in x.dims]
        if missing:
            raise ValueError(f"drift_dims not present in position arrays: {missing}")

        x_drift = x.mean(dim=drift_dims, skipna=True)
        y_drift = y.mean(dim=drift_dims, skipna=True)

        x = x - x_drift
        y = y - y_drift

        if z is not None:
            z_drift = z.mean(dim=drift_dims, skipna=True)
            z = z - z_drift

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
    out.attrs["drift_corrected"] = drift_correct

    if drift_correct and drift_dims is not None:
        out.attrs["drift_dims"] = tuple(drift_dims)

    if base_units is not None:
        out.attrs["units"] = f"{base_units}^2"

    return out

def _fft_autocorr_1d(series: np.ndarray, mask_zeros: bool = True, correct_missing: bool = True) -> np.ndarray:
    """FFT-based autocorrelation normalized by lag count and mean^2 (G(τ) convention).

    correct_missing : bool, default True
        If True, norm counts only pairs where both t and t+τ are valid — correct
        when tracks have missing timepoints.
        If False, norm counts valid t positions only (legacy behavior, equivalent
        to correct_missing=True for complete tracks with no NaN).
    """
    n = len(series)
    mask = ~np.isnan(series)
    if mask_zeros:
        mask &= series != 0
    series_clean = series[mask]
    if len(series_clean) < 2:
        return np.full(n, np.nan)

    mean_val = np.mean(series_clean)
    series_zm = np.nan_to_num(series - mean_val, nan=0.0)

    padded = np.zeros(2 * n)
    padded[:n] = series_zm
    fft_s = np.fft.fft(padded)
    autocorr = np.fft.ifft(fft_s * np.conj(fft_s)).real[:n]

    if correct_missing:
        norm = np.array([np.sum(mask[: n - tau] & mask[tau:]) for tau in range(n)], dtype=float)
    else:
        norm = np.array([np.sum(mask[: n - tau]) for tau in range(n)], dtype=float)
    norm[norm == 0] = np.nan
    autocorr = autocorr / norm

    if mean_val != 0:
        autocorr /= mean_val ** 2

    return autocorr


def _bleach_correct_series(series: np.ndarray) -> np.ndarray:
    """Divide signal by a fitted single-exponential decay A*exp(-k*t) + C."""
    from scipy.optimize import curve_fit

    n = len(series)
    t = np.arange(n, dtype=float)
    mask = ~np.isnan(series) & (series != 0)
    if mask.sum() < 5:
        return series.copy()

    t_v, s_v = t[mask], series[mask]
    try:
        popt, _ = curve_fit(
            lambda x, A, k, C: A * np.exp(-k * x) + C,
            t_v,
            s_v,
            p0=[s_v[0], 1.0 / n, s_v[-1]],
            maxfev=3000,
        )
        A, k, C = popt
        fit = A * np.exp(-k * t) + C
        fit = np.where(np.abs(fit) < 1e-10, np.nan, fit)
        corrected = series / fit
        corrected[~mask] = np.nan
        return corrected
    except Exception:
        return series.copy()


def autocorr(
    ta,
    *,
    var: str = "signal",
    out_name: str | None = None,
    max_lag: int | None = None,
    bleach_correct: bool = False,
    correct_missing: bool = True,
):
    """
    Compute per-track FFT autocorrelation G(Δt) and store it in the dataset.

    Works like `msd`: the `t` dimension of the output represents lag time (Δt),
    not absolute time. Returns the TrackArray so calls can be chained.

    Works for both channelled variables (dims include 'ch') and channel-less
    variables (e.g. 'ch0_mask__major_axis_length_px').

    Parameters
    ----------
    ta : TrackArray or xarray.Dataset
    var : str
        Source variable (e.g. 'signal', 'ch0_mask__major_axis_length_px').
    out_name : str or None
        Name for the new dataset variable. Defaults to ``f"{var}_autocorr"``.
    max_lag : int or None
        Maximum lag in frames. Defaults to half the time-axis length.
    bleach_correct : bool
        Fit and divide out a per-track exponential decay before correlating.
        Corrects for photobleaching or slow signal loss over time.

    Returns
    -------
    ta
        Same TrackArray (or Dataset) with ``out_name`` added in place.
    """
    ds = ta.ds if hasattr(ta, "ds") else ta

    if var not in ds:
        raise KeyError(f"Variable {var!r} not found. Available: {list(ds.data_vars)}")

    if out_name is None:
        out_name = f"{var}_autocorr"

    da = ds[var]
    has_ch = "ch" in da.dims

    n_t = ds.sizes["t"]
    if max_lag is None:
        max_lag = n_t // 2
    max_lag = min(max_lag, n_t - 1)

    t_coord = ds["t"].values[: max_lag + 1]

    group_dims = [d for d in da.dims if d not in ("t", "ch")]

    def _compute_for_slice(da_slice):
        da_stacked = da_slice.stack(obs=group_dims)
        n_obs = da_stacked.sizes["obs"]
        result = np.full((n_obs, max_lag + 1), np.nan)
        for i in range(n_obs):
            series = da_stacked.isel(obs=i).values.astype(float)
            if bleach_correct:
                series = _bleach_correct_series(series)
            g = _fft_autocorr_1d(series, correct_missing=correct_missing)[: max_lag + 1]
            result[i, : len(g)] = g
        out = xr.DataArray(
            result,
            dims=["obs", "t"],
            coords={"obs": da_stacked.coords["obs"], "t": t_coord},
        )
        return out.unstack("obs")

    if has_ch:
        ch_coord = da.coords["ch"] if "ch" in da.coords else np.arange(da.sizes["ch"])
        slices = [_compute_for_slice(da.isel(ch=ch_i, drop=True))
                  for ch_i in range(da.sizes["ch"])]
        out_da = xr.concat(slices, dim=xr.DataArray(ch_coord, dims="ch", name="ch"))
    else:
        out_da = _compute_for_slice(da)

    out_da.attrs.update({
        "long_name": f"autocorrelation of {var}",
        "source_var": var,
        "bleach_corrected": bleach_correct,
        "t_description": "lag time Δt (same units as dataset t coordinate); NOT absolute time",
    })

    ds[out_name] = out_da
    return ta


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