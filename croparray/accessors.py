from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Re-export the generated dataset-aware accessors
from ._accessors_generated import (
    _BaseAccessor,
    CropArrayPlot,
    CropArrayMeasure,
    CropArrayDF,
    CropArrayView,
    CropArrayTrack as _GenCropArrayTrack,
    TrackArrayPlot,
    TrackArrayMeasure,
    TrackArrayView,
    TrackArrayDF,
    install_generated_accessors,  
)

# ----------------------------
# Hand-written "thin" accessors that are NOT purely ds-first
# (keep these minimal; move to generated later only if you make them ds-first)
# ----------------------------

@dataclass
class CropArrayIO(_BaseAccessor):
    """I/O convenience methods that don’t follow the ds-first pattern."""
    def open(self, *args, **kwargs):
        from .io import open_croparray
        return open_croparray(*args, **kwargs)

    def open_zarr(self, *args, **kwargs):
        from .io import open_croparray_zarr
        return open_croparray_zarr(*args, **kwargs)


@dataclass
class CropArrayBuild(_BaseAccessor):
    """Builder convenience methods that don’t follow the ds-first pattern."""
    def create(self, *args, **kwargs):
        from .build import create_crop_array
        return create_crop_array(*args, **kwargs)


@dataclass
class CropArrayOps(_BaseAccessor):
    """Ops wrapper (kept hand-written; not strictly ds-first)."""
    def apply(self, func, source="best_z", *args, **kwargs):
        from .crop_ops.apply import apply_crop_op
        return apply_crop_op(self.ds, func, source=source, *args, **kwargs)


@dataclass
class CropArrayTrack(_GenCropArrayTrack):
    """
    Extend generated track accessor with one ergonomic helper.
    Keeps your prior behavior where conversion returns a TrackArray object.
    """
    def to_trackarray(self, *args, **kwargs):
        from .tracking import to_track_array
        return to_track_array(self.ds, *args, **kwargs)


def _install_all_accessors():
    # Local imports to avoid circular dependencies
    from .crop_array_object import CropArray
    from .trackarray.object import TrackArray
    install_generated_accessors(CropArray, TrackArray)

_install_all_accessors()
