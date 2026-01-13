from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import functools
import importlib
import inspect


def delegate_ds(mod_path: str, func_name: str, *, drop_first_arg: bool = True):
    """
    Create an accessor method that forwards self.ds into the target function while
    preserving docstring + signature for IDEs/Jupyter.

    This avoids duplicating docstrings in accessors while still allowing call tips.
    """

    # Import at definition time so the resulting attribute is a real function on the class.
    # NOTE: This can surface circular import issues if the target module imports accessors.
    module = importlib.import_module(mod_path, package=__package__)
    func = getattr(module, func_name)

    @functools.wraps(func)  # copies __doc__, __name__, __module__, etc.
    def method(self, *args, **kwargs):
        return func(self.ds, *args, **kwargs)

    # Make the signature match the underlying function but hide the injected ds argument.
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if drop_first_arg and params:
            params = params[1:]
        method.__signature__ = inspect.Signature(
            parameters=params,
            return_annotation=sig.return_annotation,
        )
        # Some frontends look at this for displaying call tips
        method.__text_signature__ = str(method.__signature__)
    except Exception:
        pass

    return method


@dataclass
class _BaseAccessor:
    parent: Any  # CropArray or TrackArray

    # Module path (relative to croparray package) used for delegation.
    # Each accessor overrides this, e.g. ".plot", ".measure", ".dataframe", etc.
    _delegate_module: Optional[str] = None

    @property
    def ds(self):
        # Both wrappers should expose .ds
        return self.parent.ds

    def __getattr__(self, name: str):
        """
        Delegate missing attributes to a module associated with this accessor.

        This enables a lightweight workflow:
          - Put generic helper functions into the corresponding module
            (e.g. croparray/plot.py, croparray/measure.py, croparray/dataframe.py)
          - Access them as ca1.plot.<helper>(...), ca1.measure.<helper>(...), etc.
            without adding one-line wrapper methods each time.

        Notes
        -----
        - Delegation only triggers if normal attribute lookup fails.
        - Private names (starting with "_") are not delegated.
        - For dataset-aware public API methods where you want Jupyter call tips,
          prefer defining explicit accessor methods via `delegate_ds(...)`.
        """
        if name.startswith("_"):
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

        mod_path = getattr(self, "_delegate_module", None)
        if not mod_path:
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

        module = importlib.import_module(mod_path, package=__package__)

        try:
            return getattr(module, name)
        except AttributeError as e:
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}") from e


# ----------------------------
# CropArray accessors
# ----------------------------

@dataclass
class CropArrayIO(_BaseAccessor):
    _delegate_module: Optional[str] = ".io"

    def open(self, *args, **kwargs):
        from .io import open_croparray
        return open_croparray(*args, **kwargs)

    def open_zarr(self, *args, **kwargs):
        from .io import open_croparray_zarr
        return open_croparray_zarr(*args, **kwargs)


@dataclass
class CropArrayBuild(_BaseAccessor):
    _delegate_module: Optional[str] = ".build"

    def create(self, *args, **kwargs):
        from .build import create_crop_array
        return create_crop_array(*args, **kwargs)


@dataclass
class CropArrayMeasure(_BaseAccessor):
    _delegate_module: Optional[str] = ".measure"

    # Public API: preserve docstring + signature from underlying functions
    best_z_proj = delegate_ds(".measure", "best_z_proj")
    signal = delegate_ds(".measure", "measure_signal")
    signal_raw = delegate_ds(".measure", "measure_signal_raw")
    mask_props = delegate_ds(".measure", "measure_mask_props")


@dataclass
class CropArrayOps(_BaseAccessor):
    # You can point this at a "front-door" ops module if you make one.
    # For now, delegate to the apply module (add more ops here later if desired).
    _delegate_module: Optional[str] = ".crop_ops.apply"

    def apply(self, func, source="best_z", *args, **kwargs):
        """
        Apply a single-crop function across the crop array using xr.apply_ufunc.

        Parameters
        ----------
        func : callable
            Function that operates on a single crop (e.g. (x,y) array) and returns
            either a scalar or an image.
        source : str
            Name of the DataArray in the dataset to operate on (default: "best_z").

        Other args/kwargs are forwarded to apply_crop_op.
        """
        from .crop_ops.apply import apply_crop_op
        return apply_crop_op(self.ds, func, source=source, *args, **kwargs)


@dataclass
class CropArrayPlot(_BaseAccessor):
    _delegate_module: Optional[str] = ".plot"

    montage = delegate_ds(".plot", "montage")


@dataclass
class CropArrayView(_BaseAccessor):
    _delegate_module: Optional[str] = ".napari_view"

    def montage(self, *args, **kwargs):
        from .napari_view import view_montage
        return view_montage(*args, **kwargs)  # expects montage dataset/array


@dataclass
class CropArrayDF(_BaseAccessor):
    _delegate_module: Optional[str] = ".dataframe"

    variables = delegate_ds(".dataframe", "variables_to_df")


@dataclass
class CropArrayTrack(_BaseAccessor):
    _delegate_module: Optional[str] = ".tracking"

    def to_trackarray(self, *args, **kwargs):
        # calls existing functional tracker; returns TrackArray because you set as_object=True there
        from .tracking import to_track_array
        return to_track_array(self.ds, *args, **kwargs)


# ----------------------------
# TrackArray accessors
# ----------------------------

@dataclass
class TrackArrayPlot(_BaseAccessor):
    """
    Plotting utilities for TrackArray datasets.

    Wraps functions implemented in croparray/trackarray/plot.py.
    """

    _delegate_module: Optional[str] = ".trackarray.plot"

    plot_trackarray_crops = delegate_ds(".trackarray.plot", "plot_trackarray_crops")
    plot_track_signal_traces = delegate_ds(".trackarray.plot", "plot_track_signal_traces")


@dataclass
class TrackArrayView(_BaseAccessor):
    # placeholder for napari viewers for trackarrays once you add them
    # Set this later when you create the module, e.g. ".napari_view"
    _delegate_module: Optional[str] = None


@dataclass
class TrackArrayDF(_BaseAccessor):
    _delegate_module: Optional[str] = ".dataframe"

    variables = delegate_ds(".dataframe", "variables_to_df")
