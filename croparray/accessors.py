from __future__ import annotations

from typing import Any, Sequence

# Re-export the generated accessor classes and installer
from ._accessors_generated import (
    _BaseAccessor,
    CropArrayPlot,
    CropArrayMeasure,
    CropArrayDF,
    CropArrayView,
    CropArrayTrack,
    CropArrayOps,
    CropArrayNapari,
    CropArrayBuild,
    CropArrayIO,
    TrackArrayPlot,
    TrackArrayMeasure,
    TrackArrayView,
    TrackArrayDF,
    TrackArrayOps,
    TrackArrayNapari,
    install_generated_accessors,
)

# ----------------------------
# Seaborn figure-level wrappers (delegate to croparray.plot)
# These are convenience methods that accept seaborn-native facetting args
# and internally convert ds -> DataFrame via croparray.plot utilities.
# ----------------------------

def _relplot(
    self,
    /,
    *,
    vars: Sequence[str] | None = None,
    query: str | None = None,
    dropna: bool = True,
    **kwargs: Any,
):
    from . import plot as _plot
    return _plot.relplot(self.ds, vars=vars, query=query, dropna=dropna, **kwargs)


def _displot(
    self,
    /,
    *,
    vars: Sequence[str] | None = None,
    query: str | None = None,
    dropna: bool = True,
    **kwargs: Any,
):
    from . import plot as _plot
    return _plot.displot(self.ds, vars=vars, query=query, dropna=dropna, **kwargs)


def _catplot(
    self,
    /,
    *,
    vars: Sequence[str] | None = None,
    query: str | None = None,
    dropna: bool = True,
    **kwargs: Any,
):
    from . import plot as _plot
    return _plot.catplot(self.ds, vars=vars, query=query, dropna=dropna, **kwargs)


# Attach to both plot accessors (CropArray + TrackArray)
CropArrayPlot.relplot = _relplot
CropArrayPlot.displot = _displot
CropArrayPlot.catplot = _catplot

TrackArrayPlot.relplot = _relplot
TrackArrayPlot.displot = _displot
TrackArrayPlot.catplot = _catplot


def _install_all_accessors():
    # Local imports to avoid circular dependencies
    from .crop_array_object import CropArray
    from .trackarray.object import TrackArray
    install_generated_accessors(CropArray, TrackArray)


_install_all_accessors()
