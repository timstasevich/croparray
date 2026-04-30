
from __future__ import annotations

from dataclasses import dataclass, field

import xarray as xr
from typing import Sequence, TYPE_CHECKING, Any
from .tracking import to_track_array as _to_track_array

if TYPE_CHECKING:
    from .accessors import (
        CropArrayIO,
        CropArrayBuild,
        CropArrayMeasure,
        CropArrayPlot,
        CropArrayView,
        CropArrayDF,
        CropArrayTrack,
        CropArrayOps,
        CropArrayNapari,
    )

@dataclass
class CropArray:
    """
    Object wrapper around the underlying xarray.Dataset produced by the original
    crop-array builder. Provides method-style API: ca.best_z_proj(...), etc.
    """
    ds: xr.Dataset
    _io: "CropArrayIO" = field(init=False, repr=False)
    _build: "CropArrayBuild" = field(init=False, repr=False)
    _measure: "CropArrayMeasure" = field(init=False, repr=False)
    _plot: "CropArrayPlot" = field(init=False, repr=False)
    _view: "CropArrayView" = field(init=False, repr=False)
    _df: "CropArrayDF" = field(init=False, repr=False)
    _track: "CropArrayTrack" = field(init=False, repr=False)
    _ops: "CropArrayOps" = field(init=False, repr=False)
    _napari: "CropArrayNapari" = field(init=False, repr=False)


    def __post_init__(self):
        """
        Attach namespaced accessors (io/build/measure/plot/view/df/track)
        so you can call e.g.:
            ca1.plot.montage(...)
            ca1.measure.best_z_proj(...)
            ca1.track.to_trackarray(...)
        """
        from .accessors import (
            CropArrayIO,
            CropArrayBuild,
            CropArrayMeasure,
            CropArrayPlot,
            CropArrayView,
            CropArrayDF,
            CropArrayTrack,
            CropArrayOps,
            CropArrayNapari,
        )
        object.__setattr__(self, "_napari", CropArrayNapari(self))
        object.__setattr__(self, "_io", CropArrayIO(self))
        object.__setattr__(self, "_build", CropArrayBuild(self))
        object.__setattr__(self, "_measure", CropArrayMeasure(self))
        object.__setattr__(self, "_plot", CropArrayPlot(self))
        object.__setattr__(self, "_view", CropArrayView(self))
        object.__setattr__(self, "_df", CropArrayDF(self))
        object.__setattr__(self, "_track", CropArrayTrack(self))
        object.__setattr__(self, "_ops", CropArrayOps(self))


    @property
    def io(self) -> "CropArrayIO":
        return self._io

    @property
    def build(self) -> "CropArrayBuild":
        return self._build

    @property
    def measure(self) -> "CropArrayMeasure":
        return self._measure

    @property
    def plot(self) -> "CropArrayPlot":
        return self._plot

    @property
    def view(self) -> "CropArrayView":
        return self._view

    @property
    def df(self) -> "CropArrayDF":
        return self._df

    @property
    def track(self) -> "CropArrayTrack":
        return self._track

    @property
    def ops(self) -> "CropArrayOps":
        return self._ops
    
    @property
    def napari(self):
        return self._napari
    
    @staticmethod
    def _scalar_from_var(da, name: str, cast):
        """
        Extract a scalar from a data variable that may have been broadcast into
        an array after concatenation. If all values are equal, return the scalar.
        If they differ, raise a clear error.
        """
        import numpy as np
        vals = np.asarray(da).ravel()
        unique = np.unique(vals[~np.isnan(vals.astype(float))] if np.issubdtype(vals.dtype, np.floating) else vals)
        if unique.size == 1:
            return cast(unique[0])
        raise ValueError(
            f"'{name}' differs across concatenated datasets: {unique}. "
            f"Datasets with different '{name}' values cannot be combined."
        )

    @property
    def xy_pad(self) -> int:
        if "xy_pad" in self.ds.attrs:
            return int(self.ds.attrs["xy_pad"])
        if "xy_pad" in self.ds:  # old files
            return self._scalar_from_var(self.ds["xy_pad"], "xy_pad", int)
        raise AttributeError("xy_pad not found")

    @property
    def dx(self) -> float:
        if "dx" in self.ds.attrs:
            return float(self.ds.attrs["dx"])
        if "dx" in self.ds:  # old files
            return self._scalar_from_var(self.ds["dx"], "dx", float)
        raise AttributeError("dx not found")

    @property
    def dy(self) -> float:
        if "dy" in self.ds.attrs:
            return float(self.ds.attrs["dy"])
        if "dy" in self.ds:  # old files
            return self._scalar_from_var(self.ds["dy"], "dy", float)
        raise AttributeError("dy not found")

    @property
    def dz(self) -> float:
        if "dz" in self.ds.attrs:
            return float(self.ds.attrs["dz"])
        if "dz" in self.ds:  # old files
            return self._scalar_from_var(self.ds["dz"], "dz", float)
        raise AttributeError("dz not found")

    @property
    def dt(self) -> float:
        if "dt" in self.ds.attrs:
            return float(self.ds.attrs["dt"])
        if "dt" in self.ds:  # old files
            return self._scalar_from_var(self.ds["dt"], "dt", float)
        raise AttributeError("dt not found")

    @property
    def units(self):
        if "units" in self.ds.attrs:
            return self.ds.attrs["units"]
        if "units" in self.ds:  # old files
            return self._scalar_from_var(self.ds["units"], "units", str)
        raise AttributeError("units not found")
        
    @property
    def signal_units(self) -> str:
        return str(self.ds.attrs.get("signal_units", "a.u."))

    @property
    def xyz_units(self) -> str:
        return str(self.ds.attrs.get("xyz_units", "space"))

    @property
    def t_units(self) -> str:
        return str(self.ds.attrs.get("t_units", "time"))


    def __setitem__(self, key: str, value):
        self.ds[key] = value

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.ds, name)

    def to_xarray(self):
        """Return the underlying xarray.Dataset."""
        return self.ds

    def best_z_proj(self, ref_ch: int = 0, disk_r: int = 1, roll_n: int = 1):
        """
        Compute the best-z projection of crop intensities and add/overwrite `ds['zc']`.

        Parameters
        ----------
        ref_ch : int or None, default 0
            Reference channel used to choose the best z-plane. If None, best-z is
            computed separately for each channel.
        disk_r : int, default 1
            Radius (pixels) of the centered XY disk used to score each z-plane.
        roll_n : int, default 1
            Number of z-slices used in a rolling-z max projection (min_periods=1).

        Returns
        -------
        xarray.DataArray
            Best-z projection with dims like (fov, n, f, y, x, ch). Also augments
            the underlying dataset by adding/overwriting `ds['zc']`.
        """
        from .measure import best_z_proj
        out = best_z_proj(self.ds, ref_ch=ref_ch, disk_r=disk_r, roll_n=roll_n)

        self.ds["best_z"] = out
        return out
    
    def measure_signal(
        self,
        ref_ch=None,
        disk_r: int = 1,
        disk_bg=None,
        roll_n: int = 1,
        **kwargs
    ):
        """
        Measure background-subtracted intensity signals for crops.

        Parameters
        ----------
        ref_ch : int or None, default None
            Channel used to choose best z for measurements. None uses all channels.
        disk_r : int, default 1
            Radius (pixels) for signal measurement disk.
        disk_bg : int or None, default None
            Radius (pixels) defining an outer ring (width 1 pixel) for background.
            If None, the function may default to `xy_pad` behavior.
        roll_n : int, default 1
            Rolling-z window used for z selection/projection.
        **kwargs
            Passed through to the underlying implementation.

        Returns
        -------
        CropArray
            Self, with `ds` augmented (e.g., adds `best_z`, `signal`, etc.).
        """
        from .measure import measure_signal
        ds2 = measure_signal(
            self.ds,
            ref_ch=ref_ch,
            disk_r=disk_r,
            disk_bg=disk_bg,
            roll_n=roll_n,
            **kwargs
        )

        if isinstance(ds2, xr.Dataset) and ds2 is not self.ds:
            self.ds = ds2
        return self
    
    def to_trackarray(
    self,
    channel_to_track: int = 0,
    min_track_length: int = 5,
    search_range: int = 10,
    memory: int = 1,
    ):
        """
        Track particles in this CropArray and return a TrackArray object.

        Notes
        -----
        This overwrites `self.ds['id']` to store track IDs (0 indicates untracked/filtered).
        The original `id` is preserved in `spot_id` the first time tracking is run.
        """
        return _to_track_array(
            self.ds,
            channel_to_track=channel_to_track,
            min_track_length=min_track_length,
            search_range=search_range,
            memory=memory,
        )
    
    def decompose_crops(
        self,
        voxel_size,
        *,
        reference_spot=None,
        spot_radius=None,
        sigma_yx=None,
        sigma_z=None,
        amplitude=None,
        background=None,
        channel: int = 0,
        source: str = "int",
        z_select = "best",
        denoise = False,
        mask_source = None,
        mask_dilate: int = 0,
        limit_gaussian: int = 1000,
        progress: bool = True,
    ):
        """
        Decompose clustered spots in each crop using gaussian mixture fitting.

        Gaussian parameters can be provided in two ways:

        1. **From a reference spot**: pass ``reference_spot`` and ``spot_radius``.
        2. **Directly**: pass ``sigma_yx``, ``amplitude``, ``background``
           (and ``sigma_z`` for 3D).

        Parameters
        ----------
        voxel_size : tuple of float
            Voxel size in nm: (y, x) for 2D or (z, y, x) for 3D.
        reference_spot : np.ndarray, optional
            A 2D or 3D diffraction-limited reference spot.
        spot_radius : tuple of float, optional
            Spot radius in nm. Required with ``reference_spot``.
        sigma_yx : float, optional
            Gaussian std dev in yx plane (nm).
        sigma_z : float, optional
            Gaussian std dev along z (nm, 3D only).
        amplitude : float, optional
            Gaussian amplitude.
        background : float, optional
            Background value.
        channel : int, default 0
            Channel index to decompose.
        source : str, default "int"
            Intensity variable name in the dataset.
        z_select : int, "best", or None, default "best"
            Z-slice selection for 2D decomposition. "best" uses best-z
            from ``zc``, int uses fixed z-index, None uses z=0.
        denoise : bool or int, default False
            Apply median filter before decomposition. True uses 3×3,
            int uses that as filter size.
        mask_source : str, optional
            Binary mask variable name (e.g. "ch0_mask"). Restricts
            decomposition to the masked region.
        mask_dilate : int, default 0
            Pixels to dilate the mask before applying.
        limit_gaussian : int, default 1000
            Maximum gaussians to fit per crop.
        progress : bool, default True
            Show progress bar.

        Returns
        -------
        df_molecules : pd.DataFrame
            Positions of all decomposed molecules with columns
            (fov, n, t, mol_idx, y_px, x_px) or (fov, n, t, mol_idx, z_px, y_px, x_px).
            ``n_molecules``, ``spot_length``, and ``decompose_map`` are
            added to the dataset in place.
        """
        from .crop_ops.decompose import decompose_crops
        ds, df = decompose_crops(
            self.ds,
            voxel_size=voxel_size,
            reference_spot=reference_spot,
            spot_radius=spot_radius,
            sigma_yx=sigma_yx,
            sigma_z=sigma_z,
            amplitude=amplitude,
            background=background,
            channel=channel,
            source=source,
            z_select=z_select,
            denoise=denoise,
            mask_source=mask_source,
            mask_dilate=mask_dilate,
            limit_gaussian=limit_gaussian,
            add_to_ds=True,
            progress=progress,
        )
        self.ds = ds
        return df

    # -----------------------------
    # xarray passthrough helpers
    # -----------------------------
    def _wrap_ds(self, ds: xr.Dataset):
        """Wrap an xarray.Dataset back into the same wrapper type (CropArray or TrackArray)."""
        return type(self)(ds)

    def _call(self, name: str, *args, **kwargs):
        """Call an xarray.Dataset method by name and re-wrap the result."""
        return self._wrap_ds(getattr(self.ds, name)(*args, **kwargs))

    # -----------------------------
    # selection / dropping
    # -----------------------------
    def sel(self, *args, **kwargs):
        """xarray-like selection that preserves the wrapper type."""
        return self._call("sel", *args, **kwargs)

    def isel(self, *args, **kwargs):
        """xarray-like index selection that preserves the wrapper type."""
        return self._call("isel", *args, **kwargs)

    def where(self, *args, **kwargs):
        """xarray-like where that preserves the wrapper type."""
        return self._call("where", *args, **kwargs)

    def drop_vars(self, *args, **kwargs):
        """Drop variables and preserve wrapper type."""
        return self._call("drop_vars", *args, **kwargs)

    def drop_sel(self, *args, **kwargs):
        """Drop selection and preserve wrapper type."""
        return self._call("drop_sel", *args, **kwargs)

    def drop_isel(self, *args, **kwargs):
        """Drop index selection and preserve wrapper type."""
        return self._call("drop_isel", *args, **kwargs)

    # -----------------------------
    # reductions / aggregations
    # -----------------------------
    def mean(self, *args, **kwargs):
        """
        xarray-like mean that preserves wrapper type.
        """
        return self._call("mean", *args, **kwargs)

    def sum(self, *args, **kwargs):
        return self._call("sum", *args, **kwargs)

    def median(self, *args, **kwargs):
        return self._call("median", *args, **kwargs)

    def std(self, *args, **kwargs):
        return self._call("std", *args, **kwargs)

    def max(self, *args, **kwargs):
        return self._call("max", *args, **kwargs)

    def min(self, *args, **kwargs):
        return self._call("min", *args, **kwargs)

    def quantile(self, *args, **kwargs):
        return self._call("quantile", *args, **kwargs)

    def count(self, *args, **kwargs):
        return self._call("count", *args, **kwargs)

    def any(self, *args, **kwargs):
        return self._call("any", *args, **kwargs)

    def all(self, *args, **kwargs):
        return self._call("all", *args, **kwargs)

    # -----------------------------
    # sorting / ranking
    # -----------------------------
    def sortby(self, *args, **kwargs):
        return self._call("sortby", *args, **kwargs)

    def rank(self, *args, **kwargs):
        return self._call("rank", *args, **kwargs)

    # -----------------------------
    # missing data
    # -----------------------------
    def fillna(self, *args, **kwargs):
        return self._call("fillna", *args, **kwargs)

    def dropna(self, *args, **kwargs):
        return self._call("dropna", *args, **kwargs)

    def interpolate_na(self, *args, **kwargs):
        return self._call("interpolate_na", *args, **kwargs)

    # -----------------------------
    # rolling
    # -----------------------------
    def rolling(self, *args, **kwargs):
        """
        Return xarray's rolling object.

        Note: rolling(...).mean() returns an xr.Dataset, so you'll lose the wrapper
        unless you use rolling_mean/rolling_sum below.
        """
        return self.ds.rolling(*args, **kwargs)

    def rolling_mean(self, *args, **kwargs):
        """Wrapper-preserving rolling mean: type(self)(self.ds.rolling(...).mean(...))."""
        return self._wrap_ds(self.ds.rolling(*args, **kwargs).mean())

    def rolling_sum(self, *args, **kwargs):
        """Wrapper-preserving rolling sum: type(self)(self.ds.rolling(...).sum(...))."""
        return self._wrap_ds(self.ds.rolling(*args, **kwargs).sum())

    def __getitem__(self, key):
        """xarray-like variable access: wrapper['var'] -> wrapper.ds['var']."""
        return self.ds[key]

    def __setitem__(self, key, value):
        """
        Allow wrapper["var"] = value assignment.
        """
        self.ds[key] = value



    @classmethod
    def concat(
        cls,
        cas: Sequence["CropArray"],
        *,
        dim: str = "Exp",
        labels: Sequence[str] | None = None,
        start_index: int = 1,
        join: str = "exact",
        compat: str = "equals",
        coords: str = "minimal",
        combine_attrs: str = "override",
    ) -> "CropArray":
        """
        Concatenate multiple CropArray objects along a new (or existing) dimension.

        Parameters
        ----------
        cas
            Sequence of CropArray objects to concatenate.
        dim
            Name of the concatenation dimension. Default "Exp". Common alternatives: "Rep".
        labels
            Coordinate values for `dim`. Must have length == len(cas).
            If None, labels are auto-generated as [f"{dim}{i}", ...] with i starting at start_index.
            Example: dim="Exp" -> ["Exp1","Exp2",...], dim="Rep" -> ["Rep1","Rep2",...].
        start_index
            Starting index for auto-generated labels (default 1).
        join, compat, coords, combine_attrs
            Passed to xarray.concat. Defaults are conservative ("exact", "equals") to fail fast on mismatch.

        Returns
        -------
        CropArray
            New CropArray with concatenated dataset and coordinate named `dim`.
        """
        if not cas:
            raise ValueError("cas must contain at least one CropArray.")

        if labels is None:
            labels = [f"{dim}{i}" for i in range(start_index, start_index + len(cas))]

        if len(labels) != len(cas):
            raise ValueError(
                f"labels length ({len(labels)}) must match number of CropArrays ({len(cas)})."
            )

        # Accept either CropArray-like objects (with .ds) or raw xarray.Datasets.
        ds_list: list[xr.Dataset] = []
        wrapper_type = None  # preserve wrapper type if possible

        for obj in cas:
            if hasattr(obj, "ds"):
                if wrapper_type is None:
                    wrapper_type = type(obj)
                ds_list.append(obj.ds)  # type: ignore[attr-defined]
            elif isinstance(obj, xr.Dataset):
                ds_list.append(obj)
            else:
                raise TypeError(
                    f"cas must contain CropArray/TrackArray objects or xarray.Dataset; got {type(obj)}"
                )

        dim_coord = xr.DataArray(list(labels), dims=(dim,), name=dim)

        ds_out = xr.concat(
            ds_list,
            dim=dim_coord,
            join=join,
            compat=compat,
            coords=coords,
            combine_attrs=combine_attrs,
        )

        # Preserve wrapper type when concatenating TrackArrays.
        # TrackArray inherits CropArray, so isinstance(..., CropArray) is True for both;
        # we need to check the concrete class.
        first_type = type(cas[0])
        if all(type(ca) is first_type for ca in cas):
            return first_type(ds_out)  # type: ignore[call-arg]

        # Mixed types: fall back to CropArray (conservative)
        return cls(ds_out)


