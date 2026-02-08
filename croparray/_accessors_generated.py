from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import inspect

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
        """Returns a montage of a crop array for easier visualization.

Parameters
----------
ds : xr.Dataset
    A crop array dataset.
col : str, optional
    Coordinate to arrange in columns, typically 'fov', 'n', or 't' (default 't').
row : str, optional
    Coordinate to arrange in rows, typically 'fov', 'n' (default 'n'), or 't'.

Returns
-------
xr.Dataset
    A reshaped crop array in which individual crops are arranged in a 2D array
    of dimensions row x col. If row and col are the same, a square montage is returned."""
        return _impl_CropArrayPlot_montage(self.ds, col=col, row=row, **kwargs)


CropArrayPlot.montage.__doc__ = _impl_CropArrayPlot_montage.__doc__
CropArrayPlot.montage.__wrapped__ = _impl_CropArrayPlot_montage
CropArrayPlot.montage.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayPlot_montage).parameters.values())[1:])


from croparray.measure import best_z_proj as _impl_CropArrayMeasure_best_z_proj
from croparray.measure import measure_signal as _impl_CropArrayMeasure_measure_signal
from croparray.measure import measure_signal_raw as _impl_CropArrayMeasure_measure_signal_raw
from croparray.measure import mask_props as _impl_CropArrayMeasure_mask_props
from croparray.measure import mask_skeleton_length as _impl_CropArrayMeasure_mask_skeleton_length

@dataclass
class CropArrayMeasure(_BaseAccessor):
    """Generated accessor methods."""
    def best_z_proj(self, ref_ch=0, disk_r=1, roll_n=1, use_zc=False):
        """Return a best-z projection of crop intensities.

Output is constructed from `ca.int` by:
  1) applying a centered rolling-z max projection of length `roll_n` (min_periods=1)
  2) selecting a single z index per crop.

Two modes are supported:

Mode A (default): use_zc=False
  - Compute per-crop best z by maximizing mean intensity within an xy disk of radius `disk_r`.
  - Stores the result as:
      * ca['z_pos_best'] : local z index into the stored `z` dimension (dims like fov,n,t[,ch])
      * ca['zc_best_pix'] : global best z pixel index (if ca['zc_pix'] exists)
      * ca['zc_best'] : global best z coordinate in z-index units (if ca['zc'] exists)
  - Does NOT overwrite ca['zc'] (which is treated as a global float coordinate).

Mode B: use_zc=True
  - Do not compute best z.
  - Use an existing selector to pick z after rolling-max:
      * prefers ca['z_pos'] if present (local index into stored `z`)
      * otherwise falls back to ca['zc'] (legacy only; must already be local)
  - If selector == -1, returns all zeros for those crops.

Parameters
----------
ca : Dataset-like
    Must contain `ca.int` with a `z` dimension and coords `x`, `y`.
ref_ch : int or None, default=0
    Used only when use_zc=False:
      - int: compute z_pos_best from that channel and apply to all channels
      - None: compute independently per channel (produces z_pos_best with ch dim)
disk_r : int, default=1
    Radius in pixels of xy disk for computing best-z signal.
roll_n : int, default=1
    Rolling window along z for rolling max projection.
use_zc : bool, default=False
    If True, use existing z selector (z_pos preferred; else zc).

Returns
-------
xarray.DataArray
    Best-z image with dims like (fov,n,t,y,x,ch)."""
        return _impl_CropArrayMeasure_best_z_proj(self.ds, ref_ch, disk_r, roll_n, use_zc)

    def measure_signal(self, ref_ch=None, disk_r=1, disk_bg=None, roll_n=1, use_zc=False, drop_int=False, drop_best_z=False):
        """Measure background-subtracted crop intensity signal using a best-z projection.

This function calls `best_z_proj(...)` to obtain a per-crop best-z image, then computes:
  - signal = (mean intensity in an inner disk) - (median intensity in an outer 1-pixel ring)

Parameters
----------
ca : xarray Dataset-like (CropArray)
    Must contain `ca.int` with dimensions including `z`, and coordinates `x`, `y`.
    If `use_zc=True`, must already contain `ca['zc']`.

ref_ch : int or None, default=None
    Passed through to `best_z_proj` when `use_zc=False`.
    - If int: compute best-z indices from this channel and apply to all channels.
    - If None: compute best-z indices separately per channel.
    If `use_zc=True`, this parameter is ignored by `best_z_proj`.

disk_r : int, default=1
    Radius (in pixels) of the inner disk used to compute the signal.

disk_bg : int, default=None
    Radius (in pixels) of the background ring (1 pixel thick). If None, defaults to `ca.xy_pad`.

roll_n : int, default=1
    Rolling window length along z used inside `best_z_proj` (rolling-z max projection).

use_zc : bool, default=False
    If True, `best_z_proj` will not recompute/overwrite `ca['zc']` and will instead use the
    existing `ca['zc']` to pick the z plane (after rolling-z max). Crops with `zc == -1` return
    all-zero best-z images, yielding zero signal after subtraction.

Returns
-------
ca : same type as input
    The input crop array augmented with:
      - `ca['best_z']` : best-z image after background subtraction, dims (fov,n,t,y,x,ch)
      - `ca['signal']` : background-subtracted signal per crop, dims (fov,n,t,ch)"""
        return _impl_CropArrayMeasure_measure_signal(self.ds, ref_ch, disk_r, disk_bg, roll_n, use_zc, drop_int, drop_best_z)

    def measure_signal_raw(self, ref_ch=None, disk_r=1, roll_n=1, use_zc=False, drop_int=False, drop_best_z_raw=False):
        """Measure raw (non-background-subtracted) crop intensity signal using a best-z projection.

This function calls `best_z_proj(...)` to obtain a per-crop best-z image, then computes a raw
signal as the sum of intensities within an inner disk.

Parameters
----------
ca : xarray Dataset-like (CropArray)
    Must contain `ca.int` with dimensions including `z`, and coordinates `x`, `y`.
    If `use_zc=True`, must already contain `ca['zc']`.

ref_ch : int or None, default=None
    Passed through to `best_z_proj` when `use_zc=False` (see its docs).
    If `use_zc=True`, this parameter is ignored by `best_z_proj`.

disk_r : int, default=1
    Radius (in pixels) of the disk used to compute the signal.

roll_n : int, default=1
    Rolling window length along z used inside `best_z_proj` (rolling-z max projection).

use_zc : bool, default=False
    If True, `best_z_proj` will not recompute/overwrite `ca['zc']` and will instead use the
    existing `ca['zc']` to pick the z plane (after rolling-z max). Crops with `zc == -1` return
    all-zero best-z images, yielding zero raw signal.

Returns
-------
ca : same type as input
    The input crop array augmented with:
      - `ca['best_z_raw']` : best-z image (no background subtraction), dims (fov,n,t,y,x,ch)
      - `ca['signal_raw']` : raw signal per crop, dims (fov,n,t,ch)"""
        return _impl_CropArrayMeasure_measure_signal_raw(self.ds, ref_ch, disk_r, roll_n, use_zc, drop_int, drop_best_z_raw)

    def mask_props(self, source, out_prefix=None, props=('area_px', 'eccentricity', 'solidity', 'perimeter_px', 'centroid_y_px', 'centroid_x_px'), connectivity=2, empty_value=float("nan")):
        """Measure morphology from a binary mask layer across the entire crop array and add scalar
measurement layers back onto `ca`.

Outputs are named: f"{out_prefix}__{prop}" and have dims equal to the non-(y,x) dims of `source`."""
        return _impl_CropArrayMeasure_mask_props(self.ds, source=source, out_prefix=out_prefix, props=props, connectivity=connectivity, empty_value=empty_value)

    def mask_skeleton_length(self, source, out_prefix=None, method='longest_path', connectivity=2, empty_value=float("nan")):
        """Compute a skeleton-based length (in pixels) from a binary mask layer across the entire
crop array and add a scalar measurement layer back onto `ca`.

This is designed for "comet-like" objects where you want a robust head-to-tail length.

Parameters
----------
ca : CropArray-like
    Object containing xarray DataArrays in `ca[data_var_name]` and supporting assignment
    `ca[new_name] = xr.DataArray(...)`.
source : str
    Name of the binary mask layer. Must have at least 2D (y,x) as its last two dims.
    Nonzero values are treated as True.
out_prefix : str | None
    Prefix for output layer name(s). If None, uses `source`.
method : {"longest_path","total"}
    - "longest_path": longest geodesic path along the skeleton graph (recommended for head-to-tail).
    - "total": total skeleton length (sum of unique skeleton edges).
connectivity : {1,2}
    1 uses 4-neighborhood (up/down/left/right). 2 uses 8-neighborhood (also diagonals).
empty_value : float
    Value used when the mask is empty (no True pixels), or skeletonization yields no nodes.

Outputs
-------
Adds one scalar layer to `ca`:
  - f"{out_prefix}__skeleton_longest_path_px" if method=="longest_path"
  - f"{out_prefix}__skeleton_total_length_px" if method=="total"

Notes
-----
- "longest_path" is generally the best proxy for head-to-tail length for curved tails.
- If the skeleton has no endpoints (e.g., a loop), "longest_path" falls back to a
  two-pass Dijkstra diameter estimate on the skeleton graph."""
        return _impl_CropArrayMeasure_mask_skeleton_length(self.ds, source=source, out_prefix=out_prefix, method=method, connectivity=connectivity, empty_value=empty_value)


CropArrayMeasure.best_z_proj.__doc__ = _impl_CropArrayMeasure_best_z_proj.__doc__
CropArrayMeasure.best_z_proj.__wrapped__ = _impl_CropArrayMeasure_best_z_proj
CropArrayMeasure.best_z_proj.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayMeasure_best_z_proj).parameters.values())[1:])
CropArrayMeasure.measure_signal.__doc__ = _impl_CropArrayMeasure_measure_signal.__doc__
CropArrayMeasure.measure_signal.__wrapped__ = _impl_CropArrayMeasure_measure_signal
CropArrayMeasure.measure_signal.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayMeasure_measure_signal).parameters.values())[1:])
CropArrayMeasure.measure_signal_raw.__doc__ = _impl_CropArrayMeasure_measure_signal_raw.__doc__
CropArrayMeasure.measure_signal_raw.__wrapped__ = _impl_CropArrayMeasure_measure_signal_raw
CropArrayMeasure.measure_signal_raw.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayMeasure_measure_signal_raw).parameters.values())[1:])
CropArrayMeasure.mask_props.__doc__ = _impl_CropArrayMeasure_mask_props.__doc__
CropArrayMeasure.mask_props.__wrapped__ = _impl_CropArrayMeasure_mask_props
CropArrayMeasure.mask_props.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayMeasure_mask_props).parameters.values())[1:])
CropArrayMeasure.mask_skeleton_length.__doc__ = _impl_CropArrayMeasure_mask_skeleton_length.__doc__
CropArrayMeasure.mask_skeleton_length.__wrapped__ = _impl_CropArrayMeasure_mask_skeleton_length
CropArrayMeasure.mask_skeleton_length.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayMeasure_mask_skeleton_length).parameters.values())[1:])


from croparray.dataframe import variables_to_df as _impl_CropArrayDF_variables_to_df

@dataclass
class CropArrayDF(_BaseAccessor):
    """Generated accessor methods."""
    def variables_to_df(self, var_names):
        """Creates a pandas DataFrame from selected variables of a CropArray.

Parameters
----------
ca : xarray.Dataset
    The CropArray dataset.
var_names : Sequence[str]
    Names of variables in the dataset to include as columns.

Returns
-------
pandas.DataFrame
    A concatenated DataFrame containing the requested variables, with any
    MultiIndex flattened (similar to ``xr.Dataset.to_dataframe()`` but with a
    flattened index)."""
        return _impl_CropArrayDF_variables_to_df(self.ds, var_names)


CropArrayDF.variables_to_df.__doc__ = _impl_CropArrayDF_variables_to_df.__doc__
CropArrayDF.variables_to_df.__wrapped__ = _impl_CropArrayDF_variables_to_df
CropArrayDF.variables_to_df.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayDF_variables_to_df).parameters.values())[1:])


from croparray.napari_view import view_montage as _impl_CropArrayView_view_montage

@dataclass
class CropArrayView(_BaseAccessor):
    """Generated accessor methods."""
    def view_montage(self):
        """Display a montage of images in Napari based on the number of channels.

Parameters:
my_ca_montage (xarray.DataArray or xarray.Dataset): The input dataset with channel information."""
        return _impl_CropArrayView_view_montage(self.ds)


CropArrayView.view_montage.__doc__ = _impl_CropArrayView_view_montage.__doc__
CropArrayView.view_montage.__wrapped__ = _impl_CropArrayView_view_montage
CropArrayView.view_montage.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayView_view_montage).parameters.values())[1:])


@dataclass
class CropArrayTrack(_BaseAccessor):
    """Generated accessor methods."""


from croparray.trackarray.plot import plot_trackarray_crops as _impl_TrackArrayPlot_plot_trackarray_crops
from croparray.trackarray.plot import plot_track_signal_traces as _impl_TrackArrayPlot_plot_track_signal_traces

@dataclass
class TrackArrayPlot(_BaseAccessor):
    """Generated accessor methods."""
    def plot_trackarray_crops(self, layer='best_z', fov=0, track_ids=(1,), t=(0, 10, 3), rolling=1, quantile_range=(0.02, 0.99), show_grayscale=True, show_merge_chs=None, ch=None, suppress_labels=True, show_suptitle=True):
        """Plot track-centered image crops across time using xarray.plot.imshow.

Behavior
--------
- If the data has no `ch` dimension: plot grayscale.
- If the data has `ch` and `ch is None`: plot each channel in grayscale (default).
  If `show_merge_chs` is provided, also plot an RGB composite using that mapping.
- If `ch` is provided: plot only that channel in grayscale and skip merge.

Parameters
----------
show_merge_chs
    Optional mapping (r_src, g_src, b_src) using *positional* channel indices.
    Example for two channels: (0, 1, 0) -> ch0 in R/B, ch1 in G.

Returns
-------
dict[int, xr.DataArray]
    Mapping track_id -> normalized DataArray used for plotting."""
        return _impl_TrackArrayPlot_plot_trackarray_crops(self.ds, layer=layer, fov=fov, track_ids=track_ids, t=t, rolling=rolling, quantile_range=quantile_range, show_grayscale=show_grayscale, show_merge_chs=show_merge_chs, ch=ch, suppress_labels=suppress_labels, show_suptitle=show_suptitle)

    def plot_track_signal_traces(self, track_ids, var='signal', rgb=(1, 1, 1), colors=('#00f670', '#f67000', '#7000f6'), markers=('o', 's', 'D'), marker_size=6, scatter_size=25, markevery=5, figsize=(7, 2.8), ylim=None, xlim=None, col_wrap=3, y2=None, y2lim=None, y2_label=None, legend_loc='upper right', show_legend=True):
        """Plot per-track traces for a chosen variable (default: 'signal').

Works for both:
- channelled variables (dims include 'ch')
- channel-less variables (no usable 'ch')

Parameters
----------
ta_dataset
    TrackArray dataset containing `var` with dims including (track_id, t) and optionally (ch).
track_ids
    Track IDs to plot.
var
    Variable name to plot.
rgb
    Channel inclusion mask for the left axis (unless a channel is assigned to `y2`).
    If None, plot all channels. Ignored for channel-less variables.
colors, markers
    Per-channel colors/markers (cycled by channel index).
marker_size, scatter_size, markevery
    Marker and scatter display controls.
figsize
    Base size per subplot; actual figure size scales with `col_wrap` and number of tracks.
ylim, xlim
    Axis limits for the left axis.
col_wrap
    Number of columns in the subplot grid.
y2
    If not None, place that channel index on a secondary (right) y-axis.
    Only applies to channelled variables.
y2lim, y2_label
    Right-axis limits/label.
legend_loc
    'upper right', 'best', etc. Use 'outside' to place legend outside axes.
show_legend
    Toggle legend on/off.

Returns
-------
None
    Displays the figure via matplotlib."""
        return _impl_TrackArrayPlot_plot_track_signal_traces(self.ds, track_ids, var=var, rgb=rgb, colors=colors, markers=markers, marker_size=marker_size, scatter_size=scatter_size, markevery=markevery, figsize=figsize, ylim=ylim, xlim=xlim, col_wrap=col_wrap, y2=y2, y2lim=y2lim, y2_label=y2_label, legend_loc=legend_loc, show_legend=show_legend)


TrackArrayPlot.plot_trackarray_crops.__doc__ = _impl_TrackArrayPlot_plot_trackarray_crops.__doc__
TrackArrayPlot.plot_trackarray_crops.__wrapped__ = _impl_TrackArrayPlot_plot_trackarray_crops
TrackArrayPlot.plot_trackarray_crops.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayPlot_plot_trackarray_crops).parameters.values())[1:])
TrackArrayPlot.plot_track_signal_traces.__doc__ = _impl_TrackArrayPlot_plot_track_signal_traces.__doc__
TrackArrayPlot.plot_track_signal_traces.__wrapped__ = _impl_TrackArrayPlot_plot_track_signal_traces
TrackArrayPlot.plot_track_signal_traces.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayPlot_plot_track_signal_traces).parameters.values())[1:])


from croparray.trackarray.measure import tracklist as _impl_TrackArrayMeasure_tracklist
from croparray.trackarray.measure import track_length as _impl_TrackArrayMeasure_track_length

@dataclass
class TrackArrayMeasure(_BaseAccessor):
    """Generated accessor methods."""
    def tracklist(self, var=None, min_count=1, return_mask=False):
        """List track_id values that have any non-null data (or >= min_count non-nulls)
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
np.ndarray or xarray.DataArray"""
        return _impl_TrackArrayMeasure_tracklist(self.ds, var=var, min_count=min_count, return_mask=return_mask)

    def track_length(self, coord='xc', out_name='track_length', broadcast_like='signal'):
        """Compute per-track length as the number of timepoints where the track exists,
independent of fluorescence intensity.

Track existence is defined by `coord.notnull()`. By croparray convention,
coordinates such as 'xc' are NaN at timepoints where no detection exists
for that track.

Parameters
----------
ta : TrackArray
    Input TrackArray.
coord : str, default "xc"
    Coordinate-like data variable used to define presence (NaN = absent).
    Typical choices: "xc", "yc", "zc".
out_name : str, default "track_length"
    Name of the output data variable.
broadcast_like : str | None, default "signal"
    Variable to broadcast the per-track length onto so that `out_name`
    matches its dims (e.g. track_id × t × ...). If None or missing, the
    output is broadcast to match the presence mask derived from `coord`.

Returns
-------
TrackArray
    The same TrackArray with a new `out_name` layer added."""
        return _impl_TrackArrayMeasure_track_length(self.ds, coord=coord, out_name=out_name, broadcast_like=broadcast_like)


TrackArrayMeasure.tracklist.__doc__ = _impl_TrackArrayMeasure_tracklist.__doc__
TrackArrayMeasure.tracklist.__wrapped__ = _impl_TrackArrayMeasure_tracklist
TrackArrayMeasure.tracklist.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayMeasure_tracklist).parameters.values())[1:])
TrackArrayMeasure.track_length.__doc__ = _impl_TrackArrayMeasure_track_length.__doc__
TrackArrayMeasure.track_length.__wrapped__ = _impl_TrackArrayMeasure_track_length
TrackArrayMeasure.track_length.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayMeasure_track_length).parameters.values())[1:])


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
        """Combine signal and signal_raw data from each channel into a single DataFrame.

Parameters:
- my_ta: xarray.Dataset containing 'signal' and 'signal_raw' data for each channel.

Returns:
- DataFrame: Combined DataFrame with columns for each signal and signal_raw."""
        return _impl_TrackArrayDF_track_signals_to_df(self.ds)


TrackArrayDF.create_tracks_df.__doc__ = _impl_TrackArrayDF_create_tracks_df.__doc__
TrackArrayDF.create_tracks_df.__wrapped__ = _impl_TrackArrayDF_create_tracks_df
TrackArrayDF.create_tracks_df.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayDF_create_tracks_df).parameters.values())[1:])
TrackArrayDF.track_signals_to_df.__doc__ = _impl_TrackArrayDF_track_signals_to_df.__doc__
TrackArrayDF.track_signals_to_df.__wrapped__ = _impl_TrackArrayDF_track_signals_to_df
TrackArrayDF.track_signals_to_df.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayDF_track_signals_to_df).parameters.values())[1:])




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
