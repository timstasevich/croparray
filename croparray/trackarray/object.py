from __future__ import annotations

from dataclasses import dataclass
from typing import Union
import xarray as xr

from ..crop_array_object import CropArray


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

    def __post_init__(self):
        # Base initialization (attaches io/build/measure/plot/view/df/track/ops)
        super().__post_init__()

        # Track-specific validation (keep your current behavior)
        if not isinstance(self.ds, xr.Dataset):
            raise TypeError("TrackArray expects an xarray.Dataset")
        if "track_id" not in self.ds.dims:
            raise ValueError("TrackArray dataset must have a 'track_id' dimension")

        # Track-specific accessors: DO NOT overwrite base .plot/.view/.df
        from ..accessors import TrackArrayPlot, TrackArrayView, TrackArrayDF

        self.tplot = TrackArrayPlot(self)
        self.tview = TrackArrayView(self)
        self.tdf = TrackArrayDF(self)

    def __repr__(self) -> str:
        sizes = self.ds.sizes
        return (
            f"TrackArray("
            f"tracks={sizes.get('track_id', '?')}, "
            f"t={sizes.get('t', '?')})"
        )

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
