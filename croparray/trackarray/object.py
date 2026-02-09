from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union
import xarray as xr

from ..crop_array_object import CropArray

if TYPE_CHECKING:
    # Import only for type checking (avoids runtime circular imports)
    from ..accessors import TrackArrayPlot, TrackArrayView, TrackArrayDF, TrackArrayMeasure, TrackArrayNapari


@dataclass
class TrackArray(CropArray):
    """
    CropArray subclass for track-based datasets.

    Expects ds to have:
      - dimension: track_id
      - dimension: t
    and often:
      - dimension: fov (length 1 per track)
      - optional dims: z, y, x, ch
    """

    # Typed caches (init=False so dataclass doesn't expect them in the constructor)
    _tplot: "TrackArrayPlot" = field(init=False, repr=False)
    _tview: "TrackArrayView" = field(init=False, repr=False)
    _tdf: "TrackArrayDF" = field(init=False, repr=False)
    _tmeasure: "TrackArrayMeasure" = field(init=False, repr=False)
    _napari: "TrackArrayNapari" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Base initialization (attaches io/build/measure/plot/view/df/track/ops)
        super().__post_init__()

        # Track-specific validation (keep your current behavior)
        if not isinstance(self.ds, xr.Dataset):
            raise TypeError("TrackArray expects an xarray.Dataset")
        if "track_id" not in self.ds.dims:
            raise ValueError("TrackArray dataset must have a 'track_id' dimension")

        # Track-specific accessors: DO NOT overwrite base .plot/.view/.df
        from ..accessors import TrackArrayPlot, TrackArrayView, TrackArrayDF, TrackArrayMeasure, TrackArrayNapari

        # --- accessor caches (private to avoid property collisions) ---
        object.__setattr__(self, "_tplot", TrackArrayPlot(self))
        object.__setattr__(self, "_tview", TrackArrayView(self))
        object.__setattr__(self, "_tdf", TrackArrayDF(self))
        object.__setattr__(self, "_tmeasure", TrackArrayMeasure(self))
        object.__setattr__(self, "_napari", TrackArrayNapari(self))

    def __repr__(self) -> str:
        sizes = self.ds.sizes
        return f"TrackArray(tracks={sizes.get('track_id', '?')}, t={sizes.get('t', '?')})"

    # ---- Typed properties (critical for autocomplete / docstrings) ----
    @property
    def tplot(self) -> "TrackArrayPlot":
        return self._tplot

    @property
    def tview(self) -> "TrackArrayView":
        return self._tview

    @property
    def tdf(self) -> "TrackArrayDF":
        return self._tdf

    @property
    def tmeasure(self) -> "TrackArrayMeasure":
        return self._tmeasure
    
    @property
    def napari(self):
        return self._napari

    # ---- Convenience ----
    @property
    def track_ids(self):
        return self.ds.coords["track_id"].values

    def sel_track(self, track_id: Union[int, list[int]]):
        """Return a TrackArray for one or more track IDs."""
        return TrackArray(self.ds.sel(track_id=track_id))

    def to_xarray(self) -> xr.Dataset:
        """Return the underlying xarray.Dataset."""
        return self.ds

    def _repr_html_(self) -> str:
        try:
            return self.ds._repr_html_()
        except Exception:
            return f"<pre>{repr(self.ds)}</pre>"

    # ---- xarray-like methods that preserve wrapper type ----
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
