from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# This file is AUTO-GENERATED. Do not edit by hand.
# Rebuild with: python scripts/generate_accessors.py

@dataclass
class _BaseAccessor:
    parent: Any
    @property
    def ds(self):
        return self.parent.ds

from croparray.plot import montage as _impl_CropArrayPlot_montage

@dataclass
class CropArrayPlot(_BaseAccessor):
    """Generated accessor methods."""
    def montage(self, col='t', row='n', **kwargs):
        return _impl_CropArrayPlot_montage(self.ds, col=col, row=row, **kwargs)


CropArrayPlot.montage.__doc__ = _impl_CropArrayPlot_montage.__doc__


from croparray.measure import best_z_proj as _impl_CropArrayMeasure_best_z_proj
from croparray.measure import measure_signal as _impl_CropArrayMeasure_measure_signal
from croparray.measure import measure_signal_raw as _impl_CropArrayMeasure_measure_signal_raw
from croparray.measure import mask_props as _impl_CropArrayMeasure_mask_props
from croparray.measure import mask_skeleton_length as _impl_CropArrayMeasure_mask_skeleton_length

@dataclass
class CropArrayMeasure(_BaseAccessor):
    """Generated accessor methods."""
    def best_z_proj(self, **kwargs):
        return _impl_CropArrayMeasure_best_z_proj(self.ds, **kwargs)

    def measure_signal(self, **kwargs):
        return _impl_CropArrayMeasure_measure_signal(self.ds, **kwargs)

    def measure_signal_raw(self, **kwargs):
        return _impl_CropArrayMeasure_measure_signal_raw(self.ds, **kwargs)

    def mask_props(self, source, out_prefix=None, props=('area_px', 'eccentricity', 'solidity', 'perimeter_px', 'centroid_y_px', 'centroid_x_px'), connectivity=2, empty_value=float("nan")):
        return _impl_CropArrayMeasure_mask_props(self.ds, source=source, out_prefix=out_prefix, props=props, connectivity=connectivity, empty_value=empty_value)

    def mask_skeleton_length(self, source, out_prefix=None, method='longest_path', connectivity=2, empty_value=float("nan")):
        return _impl_CropArrayMeasure_mask_skeleton_length(self.ds, source=source, out_prefix=out_prefix, method=method, connectivity=connectivity, empty_value=empty_value)


CropArrayMeasure.best_z_proj.__doc__ = _impl_CropArrayMeasure_best_z_proj.__doc__
CropArrayMeasure.measure_signal.__doc__ = _impl_CropArrayMeasure_measure_signal.__doc__
CropArrayMeasure.measure_signal_raw.__doc__ = _impl_CropArrayMeasure_measure_signal_raw.__doc__
CropArrayMeasure.mask_props.__doc__ = _impl_CropArrayMeasure_mask_props.__doc__
CropArrayMeasure.mask_skeleton_length.__doc__ = _impl_CropArrayMeasure_mask_skeleton_length.__doc__


from croparray.dataframe import variables_to_df as _impl_CropArrayDF_variables_to_df

@dataclass
class CropArrayDF(_BaseAccessor):
    """Generated accessor methods."""
    def variables_to_df(self, var_names):
        return _impl_CropArrayDF_variables_to_df(self.ds, var_names)


CropArrayDF.variables_to_df.__doc__ = _impl_CropArrayDF_variables_to_df.__doc__


from croparray.napari_view import view_montage as _impl_CropArrayView_view_montage

@dataclass
class CropArrayView(_BaseAccessor):
    """Generated accessor methods."""
    def view_montage(self):
        return _impl_CropArrayView_view_montage(self.ds)


CropArrayView.view_montage.__doc__ = _impl_CropArrayView_view_montage.__doc__


@dataclass
class CropArrayTrack(_BaseAccessor):
    """Generated accessor methods."""


from croparray.trackarray.plot import plot_trackarray_crops as _impl_TrackArrayPlot_plot_trackarray_crops
from croparray.trackarray.plot import plot_track_signal_traces as _impl_TrackArrayPlot_plot_track_signal_traces

@dataclass
class TrackArrayPlot(_BaseAccessor):
    """Generated accessor methods."""
    def plot_trackarray_crops(self, layer='best_z', fov=0, track_ids=(1,), t=(0, 10, 3), rolling=1, quantile_range=(0.02, 0.99), show_grayscale=True, show_merge_chs=None, ch=None, suppress_labels=True, show_suptitle=True):
        return _impl_TrackArrayPlot_plot_trackarray_crops(self.ds, layer=layer, fov=fov, track_ids=track_ids, t=t, rolling=rolling, quantile_range=quantile_range, show_grayscale=show_grayscale, show_merge_chs=show_merge_chs, ch=ch, suppress_labels=suppress_labels, show_suptitle=show_suptitle)

    def plot_track_signal_traces(self, track_ids, var='signal', rgb=(1, 1, 1), colors=('#00f670', '#f67000', '#7000f6'), markers=('o', 's', 'D'), marker_size=6, scatter_size=25, markevery=5, figsize=(7, 2.8), ylim=None, xlim=None, col_wrap=3, y2=None, y2lim=None, y2_label=None, legend_loc='upper right', show_legend=True):
        return _impl_TrackArrayPlot_plot_track_signal_traces(self.ds, track_ids, var=var, rgb=rgb, colors=colors, markers=markers, marker_size=marker_size, scatter_size=scatter_size, markevery=markevery, figsize=figsize, ylim=ylim, xlim=xlim, col_wrap=col_wrap, y2=y2, y2lim=y2lim, y2_label=y2_label, legend_loc=legend_loc, show_legend=show_legend)


TrackArrayPlot.plot_trackarray_crops.__doc__ = _impl_TrackArrayPlot_plot_trackarray_crops.__doc__
TrackArrayPlot.plot_track_signal_traces.__doc__ = _impl_TrackArrayPlot_plot_track_signal_traces.__doc__


from croparray.trackarray.measure import tracklist as _impl_TrackArrayMeasure_tracklist

@dataclass
class TrackArrayMeasure(_BaseAccessor):
    """Generated accessor methods."""
    def tracklist(self, var=None, min_count=1, return_mask=False):
        return _impl_TrackArrayMeasure_tracklist(self.ds, var=var, min_count=min_count, return_mask=return_mask)


TrackArrayMeasure.tracklist.__doc__ = _impl_TrackArrayMeasure_tracklist.__doc__


@dataclass
class TrackArrayView(_BaseAccessor):
    """Generated accessor methods."""


from croparray.trackarray.dataframe import create_tracks_df as _impl_TrackArrayDF_create_tracks_df
from croparray.trackarray.dataframe import track_signals_to_df as _impl_TrackArrayDF_track_signals_to_df

@dataclass
class TrackArrayDF(_BaseAccessor):
    """Generated accessor methods."""
    def create_tracks_df(self):
        return _impl_TrackArrayDF_create_tracks_df(self.ds)

    def track_signals_to_df(self):
        return _impl_TrackArrayDF_track_signals_to_df(self.ds)


TrackArrayDF.create_tracks_df.__doc__ = _impl_TrackArrayDF_create_tracks_df.__doc__
TrackArrayDF.track_signals_to_df.__doc__ = _impl_TrackArrayDF_track_signals_to_df.__doc__




def install_generated_accessors(CropArray, TrackArray):
    """Attach generated accessors as @property on wrapper classes."""
    # CropArray accessors
    CropArray.plot    = property(lambda self, _A=CropArrayPlot: _A(self))
    CropArray.measure = property(lambda self, _A=CropArrayMeasure: _A(self))
    CropArray.view    = property(lambda self, _A=CropArrayView: _A(self))
    CropArray.df      = property(lambda self, _A=CropArrayDF: _A(self))
    CropArray.track   = property(lambda self, _A=CropArrayTrack: _A(self))

    # TrackArray accessors
    TrackArray.tplot    = property(lambda self, _A=TrackArrayPlot: _A(self))
    TrackArray.tmeasure = property(lambda self, _A=TrackArrayMeasure: _A(self))
    TrackArray.tview    = property(lambda self, _A=TrackArrayView: _A(self))
    TrackArray.tdf      = property(lambda self, _A=TrackArrayDF: _A(self))
