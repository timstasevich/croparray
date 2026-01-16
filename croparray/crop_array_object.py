
from dataclasses import dataclass

import xarray as xr
from typing import Sequence
from .tracking import to_track_array as _to_track_array


@dataclass
class CropArray:
    """
    Object wrapper around the underlying xarray.Dataset produced by the original
    crop-array builder. Provides method-style API: ca.best_z_proj(...), etc.
    """
    ds: xr.Dataset

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
        )

        object.__setattr__(self, "_io", CropArrayIO(self))
        object.__setattr__(self, "_build", CropArrayBuild(self))
        object.__setattr__(self, "_measure", CropArrayMeasure(self))
        object.__setattr__(self, "_plot", CropArrayPlot(self))
        object.__setattr__(self, "_view", CropArrayView(self))
        object.__setattr__(self, "_df", CropArrayDF(self))
        object.__setattr__(self, "_track", CropArrayTrack(self))
        object.__setattr__(self, "_ops", CropArrayOps(self))


    @property
    def io(self):
        return self._io

    @property
    def build(self):
        return self._build

    @property
    def measure(self):
        return self._measure

    @property
    def plot(self):
        return self._plot

    @property
    def view(self):
        return self._view

    @property
    def df(self):
        return self._df

    @property
    def track(self):
        return self._track

    @property
    def ops(self):
        return self._ops

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
    
    def sel(self, *args, **kwargs):
        """xarray-like selection that preserves the wrapper type."""
        return type(self)(self.ds.sel(*args, **kwargs))

    def isel(self, *args, **kwargs):
        """xarray-like index selection that preserves the wrapper type."""
        return type(self)(self.ds.isel(*args, **kwargs))

    def where(self, *args, **kwargs):
        """xarray-like where that preserves the wrapper type."""
        return type(self)(self.ds.where(*args, **kwargs))

    def drop_vars(self, *args, **kwargs):
        """Drop variables and preserve wrapper type."""
        return type(self)(self.ds.drop_vars(*args, **kwargs))

    def drop_sel(self, *args, **kwargs):
        """Drop selection and preserve wrapper type."""
        return type(self)(self.ds.drop_sel(*args, **kwargs))

    def drop_isel(self, *args, **kwargs):
        """Drop index selection and preserve wrapper type."""
        return type(self)(self.ds.drop_isel(*args, **kwargs))

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

        ds_list = [ca.ds for ca in cas]
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


