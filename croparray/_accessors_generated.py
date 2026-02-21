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
        """Returns a montage of a crop array for easier visualization *and* robust manual
tile-annotation in napari.

Key guarantees (for downstream click-to-label tools)
----------------------------------------------------
This function always produces:
  - integer tile axes:   montage_row, montage_col
  - pixel axes:          y, x
  - stacked pixel dims:  r=(montage_row, y), c=(montage_col, x)

It also preserves the identity of each tile:
  - If row != col:
        tile_row_id[montage_row] = original coordinate values for `row`
        tile_col_id[montage_col] = original coordinate values for `col`
  - If row == col (square packing):
        tile_id[montage_row, montage_col] = original coordinate value for that tile
        (padded tiles get a fill value; see `pad_value`)

Notes
-----
- This is designed so you can click a tile in napari and map it back to the
  underlying `row` / `col` identity, even for square montages where identity
  used to be dropped.
- Backward-compat: callers may pass col/row via kwargs.

Parameters
----------
ds : xr.Dataset
    A crop array dataset.
col : str, optional
    Coordinate/dimension to arrange in columns.
row : str, optional
    Coordinate/dimension to arrange in rows.
pad_value : scalar, optional (kwarg only)
    Value used to fill padded tiles (default: 0).

Returns
-------
xr.Dataset
    A reshaped dataset arranged in a montage."""
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
        """Creates a pandas DataFrame from selected variables/coords of a CropArray.

Unlike xr.Dataset.to_dataframe(), this lets you select specific variables
and will broadcast lower-dimensional variables to a common target grid
when possible (e.g. per-track scalars to (track_id, t)).

Parameters
----------
ca : xarray.Dataset
    The CropArray dataset.
var_names : Sequence[str]
    Names of variables/coords in the dataset to include as columns.

Returns
-------
pandas.DataFrame
    DataFrame containing requested columns, with shared index columns
    (dims/coords) included once."""
        return _impl_CropArrayDF_variables_to_df(self.ds, var_names)


CropArrayDF.variables_to_df.__doc__ = _impl_CropArrayDF_variables_to_df.__doc__
CropArrayDF.variables_to_df.__wrapped__ = _impl_CropArrayDF_variables_to_df
CropArrayDF.variables_to_df.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayDF_variables_to_df).parameters.values())[1:])


from croparray.napari_view import montage_viewer as _impl_CropArrayView_montage_viewer
from croparray.napari_view import manual_filter_montage as _impl_CropArrayView_manual_filter_montage

@dataclass
class CropArrayView(_BaseAccessor):
    """Generated accessor methods."""
    def montage_viewer(self, row, col, show=('best_z', 'ch0_mask'), ch=0, z_index=0, viewer=None, image_contrast=None, tile_overlay_contrast=None, tile_overlay_opacity=0.35, default_blending='additive', colormaps=None, show_tile_text=True, tile_text_name='tile_text', tile_text_size=14, tile_text_color='white'):
        """Create a napari viewer for a montage with smart defaults:
  - mask-like layers => add_labels
  - image-like layers (has r,c) => add_image with robust contrast
  - non-xy layers (e.g., (track_id,t) or (n,t)) => drawn as per-tile overlays

Parameters
----------
ca_or_ta : xr.Dataset
    The CropArray/TrackArray dataset.
row, col : str
    The montage tiling dims you want (e.g. row="track_id", col="track_id"; row="n", col="t").
show : iterable[str]
    Variable names from the montage dataset to display.
ch : int
    Channel to show for image-like layers that have a 'ch' dimension.
z_index : int
    z-slice to show if z is present.
viewer : napari.Viewer, optional
    If provided, add layers into this viewer; otherwise create a new one.
image_contrast : 
    Percentiles for image layers (default 0–99.5). 
tile_overlay_contrast : 
    Percentiles for tile overlays (default 5–95).
tile_overlay_opacity : float
    Opacity used for tile overlays and labels.
default_blending : str
    Napari blending mode for image layers.
colormaps : dict[str,str], optional
    Colormap per layer name; falls back to 'gray' for images and 'magenta' for overlays.

Returns
-------
viewer, layers_dict
    layers_dict maps requested layer name -> napari layer object."""
        return _impl_CropArrayView_montage_viewer(self.ds, row=row, col=col, show=show, ch=ch, z_index=z_index, viewer=viewer, image_contrast=image_contrast, tile_overlay_contrast=tile_overlay_contrast, tile_overlay_opacity=tile_overlay_opacity, default_blending=default_blending, colormaps=colormaps, show_tile_text=show_tile_text, tile_text_name=tile_text_name, tile_text_size=tile_text_size, tile_text_color=tile_text_color)

    def manual_filter_montage(self, output_dir, row, col, filter_name='manual_filter', show=('best_z', 'ch0_mask'), ch=0, z_index=0, viewer=None, write_back=True, overlay_opacity=0.35, single_click_delay_ms=100, colormaps=None, label_colors=None, show_click_info=False, show_tile_text=False):
        return _impl_CropArrayView_manual_filter_montage(self.ds, output_dir=output_dir, row=row, col=col, filter_name=filter_name, show=show, ch=ch, z_index=z_index, viewer=viewer, write_back=write_back, overlay_opacity=overlay_opacity, single_click_delay_ms=single_click_delay_ms, colormaps=colormaps, label_colors=label_colors, show_click_info=show_click_info, show_tile_text=show_tile_text)


CropArrayView.montage_viewer.__doc__ = _impl_CropArrayView_montage_viewer.__doc__
CropArrayView.montage_viewer.__wrapped__ = _impl_CropArrayView_montage_viewer
CropArrayView.montage_viewer.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayView_montage_viewer).parameters.values())[1:])
CropArrayView.manual_filter_montage.__doc__ = _impl_CropArrayView_manual_filter_montage.__doc__
CropArrayView.manual_filter_montage.__wrapped__ = _impl_CropArrayView_manual_filter_montage
CropArrayView.manual_filter_montage.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayView_manual_filter_montage).parameters.values())[1:])


from croparray.tracking import perform_tracking_with_exclusions as _impl_CropArrayTrack_perform_tracking_with_exclusions
from croparray.tracking import to_track_array as _impl_CropArrayTrack_to_track_array

@dataclass
class CropArrayTrack(_BaseAccessor):
    """Generated accessor methods."""
    def perform_tracking_with_exclusions(self, search_range=10, memory=1):
        """Perform particle tracking on DataFrame with option to exclude certain data points 
and assign them a default track ID.

Parameters:
----------
df: DataFrame
    DataFrame containing particle coordinates and potentially other data.
search_range: int
    Maximum distance particles can move between frames.
memory: int
    Maximum number of frames during which a particle can vanish, then reappear, and still be considered the same particle.

Returns:
-------
DataFrame with tracked particles, including excluded ones assigned a default track ID."""
        return _impl_CropArrayTrack_perform_tracking_with_exclusions(self.ds, search_range, memory)

    def to_track_array(self, channel_to_track=0, min_track_length=5, search_range=10, memory=1):
        """Track particles in a given croparray dataset and update the croparray dataset with new track IDs,
filtering out short tracks.

Parameters:
- ca: The dataset array
- channel_to_track: Channel index used for tracking particles
- min_track_length: Minimum length required for a track to be kept
- search_range: Search range for linking particles to form tracks
- memory: Number of frames a track can skip"""
        return _impl_CropArrayTrack_to_track_array(self.ds, channel_to_track, min_track_length, search_range, memory)


CropArrayTrack.perform_tracking_with_exclusions.__doc__ = _impl_CropArrayTrack_perform_tracking_with_exclusions.__doc__
CropArrayTrack.perform_tracking_with_exclusions.__wrapped__ = _impl_CropArrayTrack_perform_tracking_with_exclusions
CropArrayTrack.perform_tracking_with_exclusions.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayTrack_perform_tracking_with_exclusions).parameters.values())[1:])
CropArrayTrack.to_track_array.__doc__ = _impl_CropArrayTrack_to_track_array.__doc__
CropArrayTrack.to_track_array.__wrapped__ = _impl_CropArrayTrack_to_track_array
CropArrayTrack.to_track_array.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayTrack_to_track_array).parameters.values())[1:])


from croparray.crop_ops.apply import apply_crop_op as _impl_CropArrayOps_apply_crop_op
from croparray.crop_ops.apply import apply_crop_op_legacy as _impl_CropArrayOps_apply_crop_op_legacy
from croparray.crop_ops.apply import apply as _impl_CropArrayOps_apply
from croparray.crop_ops.apply import apply_legacy as _impl_CropArrayOps_apply_legacy

@dataclass
class CropArrayOps(_BaseAccessor):
    """Generated accessor methods."""
    def apply_crop_op(self, func, source='best_z', group_dims=None, group_exclude_dims=('track_id', 't', 'ch', 'x', 'y', 'z'), channels=None, channel_dim='ch', input_core_dims=('x', 'y'), output_core_dims=('x', 'y'), out_name='ch{ch}_op', func_kwargs=None, compute_sum_xy=False, sum_name='{out}_sig', sum_dims=('x', 'y'), add_to_ds=True):
        """Apply a per-crop image operation, automatically grouping over non-core dims.

Default behavior:
  - Slices ds over all dims except group_exclude_dims (e.g. fov/exp/cell/rep)
  - Runs the legacy vectorized implementation on each slice (drop=True)
  - Concats slices back

This preserves the *exact* semantics of apply_crop_op_legacy within each slice,
which is critical for quantile/threshold based operations.

Power users: call apply_crop_op_legacy(...) directly to bypass grouping."""
        return _impl_CropArrayOps_apply_crop_op(self.ds, func, source, group_dims=group_dims, group_exclude_dims=group_exclude_dims, channels=channels, channel_dim=channel_dim, input_core_dims=input_core_dims, output_core_dims=output_core_dims, out_name=out_name, func_kwargs=func_kwargs, compute_sum_xy=compute_sum_xy, sum_name=sum_name, sum_dims=sum_dims, add_to_ds=add_to_ds)

    def apply_crop_op_legacy(self, func, source='best_z', channels=None, channel_dim='ch', input_core_dims=('x', 'y'), output_core_dims=('x', 'y'), out_name='ch{ch}_op', func_kwargs=None, compute_sum_xy=False, sum_name='{out}_sig', sum_dims=('x', 'y'), add_to_ds=True):
        """Apply a per-crop image operation across a crop dataset using `xr.apply_ufunc`,
optionally iterating over channels, and optionally computing a per-crop scalar
summary (e.g., summed signal).

Conceptually, this function executes the “known-good” pattern:

    xr.apply_ufunc(
        func,
        da.sel(ch=<one channel>),
        input_core_dims=[["x", "y"]],
        output_core_dims=[["x", "y"]],
        kwargs=...,
        vectorize=True,
    )

and does so for every crop (and for each requested channel, if present).

Parameters
----------
ds
    The dataset containing crop data variables (e.g., a stack of crops across time/track/crop_id).
func
    A function to apply to each crop. It must accept an array-like input matching
    `input_core_dims` (typically a 2D image) and return an array-like output matching
    `output_core_dims`. Additional keyword arguments may be provided via `func_kwargs`.

    Typical examples: spot detection / QC, filtering, background subtraction, etc.
source
    Either the name of a DataArray in `ds` (default: `"best_z"`) or an explicit
    `xr.DataArray` to process.
channels
    Channel selection behavior when `channel_dim` exists in the source DataArray:

    - `None` (default): process **all** channels present in `da.coords[channel_dim]`.
    - `int`: process only that channel (e.g., `channels=0`).
    - `Sequence[int]`: process only those channels (e.g., `channels=[0]` or `[0, 1]`).

    If the source DataArray does **not** contain `channel_dim`, this argument is ignored
    and the operation is applied once (i.e., “single-channel” mode).
channel_dim
    Name of the channel dimension in the source DataArray (default: `"ch"`).
input_core_dims
    Core dimensions consumed by `func` (default: `("x", "y")`). These are passed to
    `xr.apply_ufunc(..., input_core_dims=[...])`.
output_core_dims
    Core dimensions produced by `func` (default: `("x", "y")`). These are passed to
    `xr.apply_ufunc(..., output_core_dims=[...])`.
out_name
    Output variable name template. If channel iteration is used, `{ch}` is formatted
    with the channel index (e.g., `"ch0_spots"`). If no channel dimension exists,
    `{ch}` is formatted as `"NA"` by default.
func_kwargs
    Optional keyword arguments forwarded to `func`. If `None`, treated as `{}`.
compute_sum_xy
    If True, compute an additional summary signal per crop by summing `out` over
    `sum_dims` and store it as a second output variable.
sum_name
    Name template for the summed-signal output, formatted with `{out}` set to the
    generated `out_name` (default: `"{out}_sig"`).
sum_dims
    Dimensions to sum over when `compute_sum_xy=True` (default: `("x", "y")`).
    Use this to control whether you sum only spatial dims, or include others.
add_to_ds
    If True (default), add outputs back into `ds` and return the updated dataset.
    If False, return a dict mapping output variable names to DataArrays.

Returns
-------
xr.Dataset or Dict[str, xr.DataArray]
    - If `add_to_ds=True`: the input dataset `ds`, augmented with new variables.
    - If `add_to_ds=False`: a dict of created outputs (and optional summaries).

Notes
-----
- This function relies on `vectorize=True` in `xr.apply_ufunc`, meaning `func` is
  applied independently across non-core dimensions (crop index, time, track_id, etc.).
- Channel handling is done via `da.sel({channel_dim: ch})` per channel.
- If you request channels that are not present in the DataArray coordinates,
  a clear `ValueError` is raised.

Examples
--------
Process all channels in `best_z`:

    ds = apply_crop_op(ds, spot_detect_and_qc, source="best_z",
                       out_name="ch{ch}_spots",
                       func_kwargs={"minmass": 150, "size": 3})

Process only channel 0:

    ds = apply_crop_op(ds, spot_detect_and_qc, source="best_z",
                       channels=[0],
                       out_name="ch{ch}_spots",
                       func_kwargs={"minmass": 150, "size": 3})

Process channels 0 and 1 and also compute summed signal per crop:

    ds = apply_crop_op(ds, spot_detect_and_qc, source="best_z",
                       channels=[0, 1],
                       out_name="ch{ch}_spots",
                       func_kwargs={"minmass": 150, "size": 3},
                       compute_sum_xy=True,
                       sum_name="{out}_sig")"""
        return _impl_CropArrayOps_apply_crop_op_legacy(self.ds, func, source, channels=channels, channel_dim=channel_dim, input_core_dims=input_core_dims, output_core_dims=output_core_dims, out_name=out_name, func_kwargs=func_kwargs, compute_sum_xy=compute_sum_xy, sum_name=sum_name, sum_dims=sum_dims, add_to_ds=add_to_ds)

    def apply(self, func, source='best_z', group_dims=None, group_exclude_dims=('track_id', 't', 'ch', 'x', 'y', 'z'), channels=None, channel_dim='ch', input_core_dims=('x', 'y'), output_core_dims=('x', 'y'), out_name='ch{ch}_op', func_kwargs=None, compute_sum_xy=False, sum_name='{out}_sig', sum_dims=('x', 'y'), add_to_ds=True):
        """Apply a per-crop image operation, automatically grouping over non-core dims.

Default behavior:
  - Slices ds over all dims except group_exclude_dims (e.g. fov/exp/cell/rep)
  - Runs the legacy vectorized implementation on each slice (drop=True)
  - Concats slices back

This preserves the *exact* semantics of apply_crop_op_legacy within each slice,
which is critical for quantile/threshold based operations.

Power users: call apply_crop_op_legacy(...) directly to bypass grouping."""
        return _impl_CropArrayOps_apply(self.ds, func, source, group_dims=group_dims, group_exclude_dims=group_exclude_dims, channels=channels, channel_dim=channel_dim, input_core_dims=input_core_dims, output_core_dims=output_core_dims, out_name=out_name, func_kwargs=func_kwargs, compute_sum_xy=compute_sum_xy, sum_name=sum_name, sum_dims=sum_dims, add_to_ds=add_to_ds)

    def apply_legacy(self, func, source='best_z', channels=None, channel_dim='ch', input_core_dims=('x', 'y'), output_core_dims=('x', 'y'), out_name='ch{ch}_op', func_kwargs=None, compute_sum_xy=False, sum_name='{out}_sig', sum_dims=('x', 'y'), add_to_ds=True):
        """Apply a per-crop image operation across a crop dataset using `xr.apply_ufunc`,
optionally iterating over channels, and optionally computing a per-crop scalar
summary (e.g., summed signal).

Conceptually, this function executes the “known-good” pattern:

    xr.apply_ufunc(
        func,
        da.sel(ch=<one channel>),
        input_core_dims=[["x", "y"]],
        output_core_dims=[["x", "y"]],
        kwargs=...,
        vectorize=True,
    )

and does so for every crop (and for each requested channel, if present).

Parameters
----------
ds
    The dataset containing crop data variables (e.g., a stack of crops across time/track/crop_id).
func
    A function to apply to each crop. It must accept an array-like input matching
    `input_core_dims` (typically a 2D image) and return an array-like output matching
    `output_core_dims`. Additional keyword arguments may be provided via `func_kwargs`.

    Typical examples: spot detection / QC, filtering, background subtraction, etc.
source
    Either the name of a DataArray in `ds` (default: `"best_z"`) or an explicit
    `xr.DataArray` to process.
channels
    Channel selection behavior when `channel_dim` exists in the source DataArray:

    - `None` (default): process **all** channels present in `da.coords[channel_dim]`.
    - `int`: process only that channel (e.g., `channels=0`).
    - `Sequence[int]`: process only those channels (e.g., `channels=[0]` or `[0, 1]`).

    If the source DataArray does **not** contain `channel_dim`, this argument is ignored
    and the operation is applied once (i.e., “single-channel” mode).
channel_dim
    Name of the channel dimension in the source DataArray (default: `"ch"`).
input_core_dims
    Core dimensions consumed by `func` (default: `("x", "y")`). These are passed to
    `xr.apply_ufunc(..., input_core_dims=[...])`.
output_core_dims
    Core dimensions produced by `func` (default: `("x", "y")`). These are passed to
    `xr.apply_ufunc(..., output_core_dims=[...])`.
out_name
    Output variable name template. If channel iteration is used, `{ch}` is formatted
    with the channel index (e.g., `"ch0_spots"`). If no channel dimension exists,
    `{ch}` is formatted as `"NA"` by default.
func_kwargs
    Optional keyword arguments forwarded to `func`. If `None`, treated as `{}`.
compute_sum_xy
    If True, compute an additional summary signal per crop by summing `out` over
    `sum_dims` and store it as a second output variable.
sum_name
    Name template for the summed-signal output, formatted with `{out}` set to the
    generated `out_name` (default: `"{out}_sig"`).
sum_dims
    Dimensions to sum over when `compute_sum_xy=True` (default: `("x", "y")`).
    Use this to control whether you sum only spatial dims, or include others.
add_to_ds
    If True (default), add outputs back into `ds` and return the updated dataset.
    If False, return a dict mapping output variable names to DataArrays.

Returns
-------
xr.Dataset or Dict[str, xr.DataArray]
    - If `add_to_ds=True`: the input dataset `ds`, augmented with new variables.
    - If `add_to_ds=False`: a dict of created outputs (and optional summaries).

Notes
-----
- This function relies on `vectorize=True` in `xr.apply_ufunc`, meaning `func` is
  applied independently across non-core dimensions (crop index, time, track_id, etc.).
- Channel handling is done via `da.sel({channel_dim: ch})` per channel.
- If you request channels that are not present in the DataArray coordinates,
  a clear `ValueError` is raised.

Examples
--------
Process all channels in `best_z`:

    ds = apply_crop_op(ds, spot_detect_and_qc, source="best_z",
                       out_name="ch{ch}_spots",
                       func_kwargs={"minmass": 150, "size": 3})

Process only channel 0:

    ds = apply_crop_op(ds, spot_detect_and_qc, source="best_z",
                       channels=[0],
                       out_name="ch{ch}_spots",
                       func_kwargs={"minmass": 150, "size": 3})

Process channels 0 and 1 and also compute summed signal per crop:

    ds = apply_crop_op(ds, spot_detect_and_qc, source="best_z",
                       channels=[0, 1],
                       out_name="ch{ch}_spots",
                       func_kwargs={"minmass": 150, "size": 3},
                       compute_sum_xy=True,
                       sum_name="{out}_sig")"""
        return _impl_CropArrayOps_apply_legacy(self.ds, func, source, channels=channels, channel_dim=channel_dim, input_core_dims=input_core_dims, output_core_dims=output_core_dims, out_name=out_name, func_kwargs=func_kwargs, compute_sum_xy=compute_sum_xy, sum_name=sum_name, sum_dims=sum_dims, add_to_ds=add_to_ds)


CropArrayOps.apply_crop_op.__doc__ = _impl_CropArrayOps_apply_crop_op.__doc__
CropArrayOps.apply_crop_op.__wrapped__ = _impl_CropArrayOps_apply_crop_op
CropArrayOps.apply_crop_op.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayOps_apply_crop_op).parameters.values())[1:])
CropArrayOps.apply_crop_op_legacy.__doc__ = _impl_CropArrayOps_apply_crop_op_legacy.__doc__
CropArrayOps.apply_crop_op_legacy.__wrapped__ = _impl_CropArrayOps_apply_crop_op_legacy
CropArrayOps.apply_crop_op_legacy.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayOps_apply_crop_op_legacy).parameters.values())[1:])
CropArrayOps.apply.__doc__ = _impl_CropArrayOps_apply.__doc__
CropArrayOps.apply.__wrapped__ = _impl_CropArrayOps_apply
CropArrayOps.apply.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayOps_apply).parameters.values())[1:])
CropArrayOps.apply_legacy.__doc__ = _impl_CropArrayOps_apply_legacy.__doc__
CropArrayOps.apply_legacy.__wrapped__ = _impl_CropArrayOps_apply_legacy
CropArrayOps.apply_legacy.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayOps_apply_legacy).parameters.values())[1:])


from croparray.napari_view import montage_viewer as _impl_CropArrayNapari_montage_viewer
from croparray.napari_view import manual_filter_montage as _impl_CropArrayNapari_manual_filter_montage

@dataclass
class CropArrayNapari(_BaseAccessor):
    """Generated accessor methods."""
    def montage_viewer(self, row, col, show=('best_z', 'ch0_mask'), ch=0, z_index=0, viewer=None, image_contrast=None, tile_overlay_contrast=None, tile_overlay_opacity=0.35, default_blending='additive', colormaps=None, show_tile_text=True, tile_text_name='tile_text', tile_text_size=14, tile_text_color='white'):
        """Create a napari viewer for a montage with smart defaults:
  - mask-like layers => add_labels
  - image-like layers (has r,c) => add_image with robust contrast
  - non-xy layers (e.g., (track_id,t) or (n,t)) => drawn as per-tile overlays

Parameters
----------
ca_or_ta : xr.Dataset
    The CropArray/TrackArray dataset.
row, col : str
    The montage tiling dims you want (e.g. row="track_id", col="track_id"; row="n", col="t").
show : iterable[str]
    Variable names from the montage dataset to display.
ch : int
    Channel to show for image-like layers that have a 'ch' dimension.
z_index : int
    z-slice to show if z is present.
viewer : napari.Viewer, optional
    If provided, add layers into this viewer; otherwise create a new one.
image_contrast : 
    Percentiles for image layers (default 0–99.5). 
tile_overlay_contrast : 
    Percentiles for tile overlays (default 5–95).
tile_overlay_opacity : float
    Opacity used for tile overlays and labels.
default_blending : str
    Napari blending mode for image layers.
colormaps : dict[str,str], optional
    Colormap per layer name; falls back to 'gray' for images and 'magenta' for overlays.

Returns
-------
viewer, layers_dict
    layers_dict maps requested layer name -> napari layer object."""
        return _impl_CropArrayNapari_montage_viewer(self.ds, row=row, col=col, show=show, ch=ch, z_index=z_index, viewer=viewer, image_contrast=image_contrast, tile_overlay_contrast=tile_overlay_contrast, tile_overlay_opacity=tile_overlay_opacity, default_blending=default_blending, colormaps=colormaps, show_tile_text=show_tile_text, tile_text_name=tile_text_name, tile_text_size=tile_text_size, tile_text_color=tile_text_color)

    def manual_filter_montage(self, output_dir, row, col, filter_name='manual_filter', show=('best_z', 'ch0_mask'), ch=0, z_index=0, viewer=None, write_back=True, overlay_opacity=0.35, single_click_delay_ms=100, colormaps=None, label_colors=None, show_click_info=False, show_tile_text=False):
        return _impl_CropArrayNapari_manual_filter_montage(self.ds, output_dir=output_dir, row=row, col=col, filter_name=filter_name, show=show, ch=ch, z_index=z_index, viewer=viewer, write_back=write_back, overlay_opacity=overlay_opacity, single_click_delay_ms=single_click_delay_ms, colormaps=colormaps, label_colors=label_colors, show_click_info=show_click_info, show_tile_text=show_tile_text)


CropArrayNapari.montage_viewer.__doc__ = _impl_CropArrayNapari_montage_viewer.__doc__
CropArrayNapari.montage_viewer.__wrapped__ = _impl_CropArrayNapari_montage_viewer
CropArrayNapari.montage_viewer.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayNapari_montage_viewer).parameters.values())[1:])
CropArrayNapari.manual_filter_montage.__doc__ = _impl_CropArrayNapari_manual_filter_montage.__doc__
CropArrayNapari.manual_filter_montage.__wrapped__ = _impl_CropArrayNapari_manual_filter_montage
CropArrayNapari.manual_filter_montage.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayNapari_manual_filter_montage).parameters.values())[1:])


from croparray.build import standardize_video_axes as _impl_CropArrayBuild_standardize_video_axes
from croparray.build import standardize_spots as _impl_CropArrayBuild_standardize_spots
from croparray.build import create_crop_array as _impl_CropArrayBuild_create_crop_array
from croparray.build import make_croparrays as _impl_CropArrayBuild_make_croparrays
from croparray.build import open_measure_concat as _impl_CropArrayBuild_open_measure_concat

@dataclass
class CropArrayBuild(_BaseAccessor):
    """Generated accessor methods."""
    def standardize_video_axes(self, video, axes, add_missing_singletons=True):
        """Reorder raw video into croparray canonical order: (fov, f, z, y, x, ch).

`axes` must describe the CURRENT order of `video`, e.g. res.axes from ca.gui.label_video_axes()."""
        return _impl_CropArrayBuild_standardize_video_axes(video, axes, add_missing_singletons=add_missing_singletons)

    def standardize_spots(self, source, tracker='auto', fov=0, xy_um_per_px=None, z_um_per_plane=None, keep_cols=None, require_calibration_if_needed=True):
        """Convert tracker output into a croparray-ready spots dataframe.

Output (always includes):
  - fov, id, f, yc, xc, zc
Output (optional):
  - track_id (if present in the input)
Plus:
  - extra columns requested by `keep_cols` (or default intensity columns)

Unit handling (TrackMate)
-------------------------
TrackMate exports POSITION_X/Y/Z in the *image spatial units* at export time
(often micron/um; sometimes pixel). We read the units row from the CSV and:

  - if units indicate pixel/px: treat values as already in pixels/planes
  - if units indicate micron/um: convert to pixels/planes using xy_um_per_px / z_um_per_plane
  - if units indicate nm: convert to pixels/planes using xy_um_per_px / z_um_per_plane (with nm scaling)

If conversion is needed but calibration is missing:
  - if require_calibration_if_needed=True (default), raise ValueError
  - else, fall back to treating coordinates as already pixel/plane units

Parameters
----------
source
    Path to a CSV file (TrackMate/TrackPy/etc.) or a pandas DataFrame.
tracker
    "auto" | "trackmate" | "trackpy" | "croparray"
fov
    Field-of-view index to assign to all rows in this spots table.
xy_um_per_px
    Microns per pixel (used when TrackMate units are micron/um/nm).
z_um_per_plane
    Microns per z-plane (used when TrackMate units are micron/um/nm).
keep_cols
    Extra columns to retain in the output. If None, keeps intensity-like columns by default.
require_calibration_if_needed
    If True, raise when TrackMate units require conversion but calibration is missing.

Returns
-------
pd.DataFrame"""
        return _impl_CropArrayBuild_standardize_spots(source, tracker=tracker, fov=fov, xy_um_per_px=xy_um_per_px, z_um_per_plane=z_um_per_plane, keep_cols=keep_cols, require_calibration_if_needed=require_calibration_if_needed)

    def create_crop_array(self, video, df, axes=None, as_object=True, **kwargs):
        """Build a crop-array from raw inputs.

Parameters
----------
video
    Input video array. If `axes` is None, this must already be ordered
    (fov, f, z, y, x, ch). If `axes` is provided, `video` may be in any
    order described by `axes` (e.g. from `ca.gui.label_video_axes(...).axes`).
df
    Spot / crop definition table.
axes : sequence of str, optional
    Axis labels describing the CURRENT order of `video`.
    If provided, `video` will be reordered into croparray canonical order:
    (fov, f, z, y, x, ch) via `standardize_video_axes(video, axes)`.
    Accepted labels: fov, f, z, y, x, ch (synonyms: t->f, c->ch).
as_object : bool, default True
    If True, return a CropArray wrapper (method-style API).
    If False, return the raw xarray.Dataset (legacy behavior).
**kwargs
    Passed through to `_create_crop_array_dataset`.

Returns
-------
CropArray or xarray.Dataset


---


Create a crop-array xarray.Dataset from a 6D video and a dataframe of detected spots.

This function extracts fixed-size crops around detected spots from a time-lapse
3D (z-stack) video and organizes them into a structured xarray Dataset (“crop array”).
Crops are always centered in the lateral (x,y) directions; axial (z) handling depends
on whether z positions are provided.

Parameters
----------
video : numpy.ndarray
    A 6D numpy array containing the raw image data with dimensions ordered as:
        (fov, f, z, y, x, ch)

    where:
        fov : field of view
        f   : frame (time)
        z   : axial plane index
        y   : lateral y coordinate
        x   : lateral x coordinate
        ch  : imaging channel

df : pandas.DataFrame
    DataFrame describing the detected spots to be cropped. At minimum, the following
    columns are required:

        - 'fov' : field-of-view index (integer or filename-like identifier)
        - 'f'   : frame index (integer, starting at 0)
        - 'xc'  : x-position of the spot center in **movie pixel coordinates**
        - 'yc'  : y-position of the spot center in **movie pixel coordinates**

    Optional columns:

        - 'zc' : axial (z) position of the spot center in **movie z-index units**.
                May be float (sub-plane precision). If provided together with
                `z_pad > 0`, crops will be extracted from a z-slab centered on this
                position.
        - 'id' : integer spot identifier. If missing, a unique id is generated.
        - 'track_id' : integer track identifier (-1 indicates untracked).
        - Any additional numeric columns will be converted into per-crop xarray
        variables with dimensions (fov, n, t).

xy_pad : int, optional
    Number of pixels to pad on either side of the crop center in the x and y directions.
    Each crop will have size (2*xy_pad + 1, 2*xy_pad + 1) in (y, x).

z_pad : int, optional
    Number of z-planes to include on either side of the provided z center.
    If `z_pad > 0` and df contains a 'zc' column, crops are extracted as a z-slab of
    depth (2*z_pad + 1) centered on the rounded z index.
    If `z_pad == 0` or 'zc' is not provided, all z planes are retained.

dx, dy, dz : float, optional
    Physical size of a pixel in the x, y, and z directions, respectively.
    These values are stored as metadata and used for coordinate construction, but do
    not affect cropping.

dt : float, optional
    Time interval between consecutive frames. Stored as metadata.

homography : list of numpy.ndarray, optional
    A list of 3×3 homography matrices, one per channel, used to correct lateral (x,y)
    misalignments between channels. Homographies are applied to the *float* (xc, yc)
    coordinates prior to cropping.

Returns
-------
xarray.Dataset
    A crop-array dataset with coordinates:
        - fov : field of view
        - n   : crop index (spot counter per frame per fov)
        - t   : time
        - z   : axial coordinate (full stack or slab-relative)
        - y   : lateral y coordinate (centered on 0)
        - x   : lateral x coordinate (centered on 0)
        - ch  : channel

    Core data variables include:

    1. int : (fov, n, t, z, y, x, ch)
        Cropped intensity data.

    2. xc, yc, zc : (fov, n, t, ch)
        Global **movie-coordinate** spot positions stored as floats.
        These are suitable for trajectory analysis, MSD calculations, and spatial
        measurements.

    3. xc_pix, yc_pix, zc_pix : (fov, n, t, ch)
        Rounded integer pixel indices corresponding to the global positions.
        These are used internally for indexing and cropping.

    4. xc_pad, yc_pad : (fov, n, t, ch)
        Pixel indices into the *padded* video used during crop extraction.

    5. z_pos : (fov, n, t, ch)
        Local z index into the stored z dimension used for best-z selection.
        A value of -1 indicates an invalid or unknown z position.
        This variable is consumed by `best_z_proj(use_z_pos=True)` and should not
        be interpreted as a physical coordinate.

    6. id : (fov, n, t)
        Spot identifier.

    7. track_id : (fov, n, t)
        Track assignment (-1 = untracked).

    Additional numeric columns in `df` are converted into per-crop variables with
    dimensions (fov, n, t).

Notes
-----
- Global coordinates (`xc`, `yc`, `zc`) are never rounded and retain subpixel precision.
- Pixel index variables (`*_pix`, `*_pad`, `z_pos`) are used strictly for array indexing.
- When z-slab mode is active, the z coordinate of the dataset is slab-relative and
centered at zero; the original global z position is preserved in `zc`.
- Crops that cannot be extracted due to out-of-bounds coordinates remain zero-filled."""
        return _impl_CropArrayBuild_create_crop_array(video, df, axes=axes, as_object=as_object, **kwargs)

    def make_croparrays(self, videos, spots, out_dir=None, out_ext='.nc', axes=None, define_axes=True, axes_source_index=0, tracker='auto', xy_um_per_px=None, z_um_per_plane=None, keep_cols=None, xy_pad=10, z_pad=1, dx=None, dy=None, dz=None, dt=None, units=None, date=None, as_object=False, skip_existing=True, progress=True, notes='A croparray built from a tracking file and a video.'):
        """High-level convenience builder that accepts either:
  - single (video, spots) OR
  - lists of (videos, spots)

It standardizes video axes, standardizes spots tables, builds croparrays,
and optionally writes .nc files.

Behavior:
  - Never overwrites existing outputs.
  - If out_dir is provided and skip_existing=True, existing .nc files are skipped.

Returns:
  - If out_dir is None: returns a single dataset/object (single input) or list of them (batch)
  - If out_dir is provided: returns list of output paths written (skipped files omitted)"""
        return _impl_CropArrayBuild_make_croparrays(videos, spots, out_dir=out_dir, out_ext=out_ext, axes=axes, define_axes=define_axes, axes_source_index=axes_source_index, tracker=tracker, xy_um_per_px=xy_um_per_px, z_um_per_plane=z_um_per_plane, keep_cols=keep_cols, xy_pad=xy_pad, z_pad=z_pad, dx=dx, dy=dy, dz=dz, dt=dt, units=units, date=date, as_object=as_object, skip_existing=skip_existing, progress=progress, notes=notes)

    def open_measure_concat(self, groups, dims, labels=None, measure_kwargs=None, drop_vars=None, join='outer', attach_provenance=True):
        """Open tracked croparrays, run measure_signal, and concatenate across nested groupings.

This is a high-level *workflow constructor* that hides the common boilerplate:

  open_as_trackarray → measure_signal → concat along dims (outer→inner)

Parameters
----------
groups
    A nested list structure whose depth equals len(dims). The leaf level must be
    a list of file paths (str/Path). Example for dims=["rep","exp","fov"]:

        groups = [
            [files_rep1_exp1, files_rep1_exp2],   # rep1: exp groups
            [files_rep2_exp1, files_rep2_exp2],   # rep2
        ]

    where each `files_repX_expY` is a list of .nc file paths (one per fov).

dims
    Grouping dimensions from outermost to innermost.
    The final dim (dims[-1]) is used to concatenate the file list at each leaf
    (e.g., "fov" or "cell").

labels
    Optional labels per dimension (length must equal len(dims)).
    Each entry can be:
      - None: auto-label at that level
          * leaf level: filename stems
          * higher levels: f"{dim}{i}"
      - list[str]: explicit labels for that level

    Example: labels=[["rep1","rep2"], ["-ZNF598","+ZNF598"], None]

measure_kwargs
    Keyword args forwarded to `.measure_signal(**measure_kwargs)` for each file.

drop_vars
    Passed to `ca.tools.open_as_trackarray(..., drop_vars=drop_vars)`.

join
    Passed to concat at each level (often "outer" in your workflows).

attach_provenance
    If True, attaches JSON metadata to ds.attrs["provenance_json"].

Returns
-------
TrackArray
    Concatenated TrackArray wrapper.

Notes
-----
- This function intentionally concatenates *hierarchically* to keep memory
  reasonable and to mirror the experiment structure.
- If you want different behavior at the leaf (e.g., leaf dim is "cell"),
  just change dims[-1] and provide a corresponding leaf file list."""
        return _impl_CropArrayBuild_open_measure_concat(groups=groups, dims=dims, labels=labels, measure_kwargs=measure_kwargs, drop_vars=drop_vars, join=join, attach_provenance=attach_provenance)


CropArrayBuild.standardize_video_axes.__doc__ = _impl_CropArrayBuild_standardize_video_axes.__doc__
CropArrayBuild.standardize_video_axes.__wrapped__ = _impl_CropArrayBuild_standardize_video_axes
CropArrayBuild.standardize_video_axes.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayBuild_standardize_video_axes).parameters.values()))
CropArrayBuild.standardize_spots.__doc__ = _impl_CropArrayBuild_standardize_spots.__doc__
CropArrayBuild.standardize_spots.__wrapped__ = _impl_CropArrayBuild_standardize_spots
CropArrayBuild.standardize_spots.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayBuild_standardize_spots).parameters.values()))
CropArrayBuild.create_crop_array.__doc__ = _impl_CropArrayBuild_create_crop_array.__doc__
CropArrayBuild.create_crop_array.__wrapped__ = _impl_CropArrayBuild_create_crop_array
CropArrayBuild.create_crop_array.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayBuild_create_crop_array).parameters.values()))
CropArrayBuild.make_croparrays.__doc__ = _impl_CropArrayBuild_make_croparrays.__doc__
CropArrayBuild.make_croparrays.__wrapped__ = _impl_CropArrayBuild_make_croparrays
CropArrayBuild.make_croparrays.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayBuild_make_croparrays).parameters.values()))
CropArrayBuild.open_measure_concat.__doc__ = _impl_CropArrayBuild_open_measure_concat.__doc__
CropArrayBuild.open_measure_concat.__wrapped__ = _impl_CropArrayBuild_open_measure_concat
CropArrayBuild.open_measure_concat.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayBuild_open_measure_concat).parameters.values()))


from croparray.io import save_croparray as _impl_CropArrayIO_save_croparray
from croparray.io import open_trackarray as _impl_CropArrayIO_open_trackarray
from croparray.io import open_as_trackarray as _impl_CropArrayIO_open_as_trackarray
from croparray.io import open_croparray as _impl_CropArrayIO_open_croparray
from croparray.io import open_croparray_zarr as _impl_CropArrayIO_open_croparray_zarr

@dataclass
class CropArrayIO(_BaseAccessor):
    """Generated accessor methods."""
    def save_croparray(self, obj, output_dir, filename=None, ext='.nc', overwrite=False, mkdir=True, to_netcdf_kwargs=None):
        """Save a CropArray/TrackArray (or raw xarray.Dataset) to NetCDF.

If `filename` is not provided, uses ds.attrs["name"] (slugified) as the stem.

Parameters
----------
obj
    CropArray, TrackArray, or xarray.Dataset. If object has `.ds`, that dataset is saved.
output_dir
    Directory to write into.
filename
    Optional filename. If None, derives from ds.attrs["name"].
    If provided without suffix, `ext` is appended.
ext
    Default ".nc".
overwrite
    If False (default), refuse to overwrite.
mkdir
    If True (default), create output_dir if needed.
to_netcdf_kwargs
    Extra kwargs passed to `Dataset.to_netcdf`.

Returns
-------
pathlib.Path
    The written file path."""
        return _impl_CropArrayIO_save_croparray(obj, output_dir=output_dir, filename=filename, ext=ext, overwrite=overwrite, mkdir=mkdir, to_netcdf_kwargs=to_netcdf_kwargs)

    def open_trackarray(self, path, as_object=True, load_manual_filters=True, **kwargs):
        return _impl_CropArrayIO_open_trackarray(path, as_object=as_object, load_manual_filters=load_manual_filters, **kwargs)

    def open_as_trackarray(self, path, drop_vars=('int',), drop_errors='ignore', as_object=True, load_manual_filters=True, **kwargs):
        """Open a CropArray dataset and immediately convert it to a TrackArray.

Parameters
----------
path : str
    Path to a dataset readable by ``open_croparray`` (e.g. NetCDF).
drop_vars : sequence of str or None, default ("int",)
    Variables to drop from the underlying Dataset before track conversion.
    Set to None or empty to disable dropping.
drop_errors : {"ignore","raise"}, default "ignore"
    Passed to ``Dataset.drop_vars``.
as_object : bool, default True
    If True, return a TrackArray wrapper. If False, return the raw Dataset.
**kwargs
    Additional keyword arguments forwarded to ``open_croparray``.

Returns
-------
TrackArray or xarray.Dataset"""
        return _impl_CropArrayIO_open_as_trackarray(path, drop_vars=drop_vars, drop_errors=drop_errors, as_object=as_object, load_manual_filters=load_manual_filters, **kwargs)

    def open_croparray(self, path, as_object=True, load_manual_filters=True, **kwargs):
        """Open a saved CropArray dataset and optionally wrap it as a CropArray object.

This function uses ``xarray.open_dataset`` and is suitable for CropArrays
stored in NetCDF or other formats supported by xarray. For large or
chunked datasets, consider using :func:`open_croparray_zarr`.

Parameters
----------
path : str
    Path to a dataset readable by ``xarray.open_dataset`` (e.g. NetCDF).
as_object : bool, default True
    If True, return a ``CropArray`` wrapper providing the method-style API
    (e.g. ``ca.best_z_proj()``, ``ca.measure_signal()``).
    If False, return the raw ``xarray.Dataset``.
**kwargs
    Additional keyword arguments passed directly to
    ``xarray.open_dataset`` (e.g. ``engine``, ``chunks``).

Returns
-------
CropArray or xarray.Dataset
    If ``as_object=True``, returns a ``CropArray`` wrapping the opened
    dataset. Otherwise, returns the underlying ``xarray.Dataset``.

Notes
-----
Datasets opened with ``open_croparray`` may be loaded eagerly or lazily
depending on the file format and the arguments passed to
``xarray.open_dataset``.

Examples
--------
Open a CropArray from a NetCDF file and compute best-z projections::

    from croparray import open_croparray

    ca = open_croparray("my_croparray.nc")
    ca.best_z_proj()
    ca.measure_signal()

Open the raw Dataset without wrapping::

    ds = open_croparray("my_croparray.nc", as_object=False)"""
        return _impl_CropArrayIO_open_croparray(path, as_object=as_object, load_manual_filters=load_manual_filters, **kwargs)

    def open_croparray_zarr(self, store, as_object=True, **kwargs):
        """Open a saved CropArray stored in Zarr format and optionally wrap it as a
CropArray object.

This function mirrors :func:`open_croparray`, but uses
``xarray.open_zarr`` instead of ``xarray.open_dataset``. It is intended
for large crop arrays that benefit from chunked, lazy loading.

Parameters
----------
store : str
    Path to the Zarr store (directory or consolidated Zarr archive).
as_object : bool, default True
    If True, return a ``CropArray`` wrapper providing the method-style API
    (e.g. ``ca.best_z_proj()``, ``ca.measure_signal()``).
    If False, return the raw ``xarray.Dataset``.
**kwargs
    Additional keyword arguments passed directly to
    ``xarray.open_zarr`` (e.g. ``consolidated=True``).

Returns
-------
CropArray or xarray.Dataset
    If ``as_object=True``, returns a ``CropArray`` wrapping the opened
    dataset. Otherwise, returns the underlying ``xarray.Dataset``.

Notes
-----
Zarr-backed CropArrays are loaded lazily; data are read from disk only
when required for computation. This makes Zarr the preferred storage
format for large or multi-FOV crop arrays.

Examples
--------
Open a Zarr-backed CropArray and compute best-z projections::

    from croparray import open_croparray_zarr

    ca = open_croparray_zarr("my_croparray.zarr")
    ca.best_z_proj()
    ca.measure_signal()

Open the raw Dataset without wrapping::

    ds = open_croparray_zarr("my_croparray.zarr", as_object=False)"""
        return _impl_CropArrayIO_open_croparray_zarr(store, as_object=as_object, **kwargs)


CropArrayIO.save_croparray.__doc__ = _impl_CropArrayIO_save_croparray.__doc__
CropArrayIO.save_croparray.__wrapped__ = _impl_CropArrayIO_save_croparray
CropArrayIO.save_croparray.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayIO_save_croparray).parameters.values()))
CropArrayIO.open_trackarray.__doc__ = _impl_CropArrayIO_open_trackarray.__doc__
CropArrayIO.open_trackarray.__wrapped__ = _impl_CropArrayIO_open_trackarray
CropArrayIO.open_trackarray.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayIO_open_trackarray).parameters.values()))
CropArrayIO.open_as_trackarray.__doc__ = _impl_CropArrayIO_open_as_trackarray.__doc__
CropArrayIO.open_as_trackarray.__wrapped__ = _impl_CropArrayIO_open_as_trackarray
CropArrayIO.open_as_trackarray.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayIO_open_as_trackarray).parameters.values()))
CropArrayIO.open_croparray.__doc__ = _impl_CropArrayIO_open_croparray.__doc__
CropArrayIO.open_croparray.__wrapped__ = _impl_CropArrayIO_open_croparray
CropArrayIO.open_croparray.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayIO_open_croparray).parameters.values()))
CropArrayIO.open_croparray_zarr.__doc__ = _impl_CropArrayIO_open_croparray_zarr.__doc__
CropArrayIO.open_croparray_zarr.__wrapped__ = _impl_CropArrayIO_open_croparray_zarr
CropArrayIO.open_croparray_zarr.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_CropArrayIO_open_croparray_zarr).parameters.values()))


from croparray.plot import montage as _impl_TrackArrayPlot_montage
from croparray.trackarray.plot import plot_trackarray_crops as _impl_TrackArrayPlot_plot_trackarray_crops
from croparray.trackarray.plot import plot_track_signal_traces as _impl_TrackArrayPlot_plot_track_signal_traces

@dataclass
class TrackArrayPlot(_BaseAccessor):
    """Generated accessor methods."""
    def montage(self, col='t', row='n', **kwargs):
        """Returns a montage of a crop array for easier visualization *and* robust manual
tile-annotation in napari.

Key guarantees (for downstream click-to-label tools)
----------------------------------------------------
This function always produces:
  - integer tile axes:   montage_row, montage_col
  - pixel axes:          y, x
  - stacked pixel dims:  r=(montage_row, y), c=(montage_col, x)

It also preserves the identity of each tile:
  - If row != col:
        tile_row_id[montage_row] = original coordinate values for `row`
        tile_col_id[montage_col] = original coordinate values for `col`
  - If row == col (square packing):
        tile_id[montage_row, montage_col] = original coordinate value for that tile
        (padded tiles get a fill value; see `pad_value`)

Notes
-----
- This is designed so you can click a tile in napari and map it back to the
  underlying `row` / `col` identity, even for square montages where identity
  used to be dropped.
- Backward-compat: callers may pass col/row via kwargs.

Parameters
----------
ds : xr.Dataset
    A crop array dataset.
col : str, optional
    Coordinate/dimension to arrange in columns.
row : str, optional
    Coordinate/dimension to arrange in rows.
pad_value : scalar, optional (kwarg only)
    Value used to fill padded tiles (default: 0).

Returns
-------
xr.Dataset
    A reshaped dataset arranged in a montage."""
        return _impl_TrackArrayPlot_montage(self.ds, col=col, row=row, **kwargs)

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


TrackArrayPlot.montage.__doc__ = _impl_TrackArrayPlot_montage.__doc__
TrackArrayPlot.montage.__wrapped__ = _impl_TrackArrayPlot_montage
TrackArrayPlot.montage.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayPlot_montage).parameters.values())[1:])
TrackArrayPlot.plot_trackarray_crops.__doc__ = _impl_TrackArrayPlot_plot_trackarray_crops.__doc__
TrackArrayPlot.plot_trackarray_crops.__wrapped__ = _impl_TrackArrayPlot_plot_trackarray_crops
TrackArrayPlot.plot_trackarray_crops.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayPlot_plot_trackarray_crops).parameters.values())[1:])
TrackArrayPlot.plot_track_signal_traces.__doc__ = _impl_TrackArrayPlot_plot_track_signal_traces.__doc__
TrackArrayPlot.plot_track_signal_traces.__wrapped__ = _impl_TrackArrayPlot_plot_track_signal_traces
TrackArrayPlot.plot_track_signal_traces.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayPlot_plot_track_signal_traces).parameters.values())[1:])


from croparray.measure import best_z_proj as _impl_TrackArrayMeasure_best_z_proj
from croparray.measure import measure_signal as _impl_TrackArrayMeasure_measure_signal
from croparray.measure import measure_signal_raw as _impl_TrackArrayMeasure_measure_signal_raw
from croparray.measure import mask_props as _impl_TrackArrayMeasure_mask_props
from croparray.measure import mask_skeleton_length as _impl_TrackArrayMeasure_mask_skeleton_length
from croparray.trackarray.measure import tracklist as _impl_TrackArrayMeasure_tracklist
from croparray.trackarray.measure import track_length as _impl_TrackArrayMeasure_track_length

@dataclass
class TrackArrayMeasure(_BaseAccessor):
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
        return _impl_TrackArrayMeasure_best_z_proj(self.ds, ref_ch, disk_r, roll_n, use_zc)

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
        return _impl_TrackArrayMeasure_measure_signal(self.ds, ref_ch, disk_r, disk_bg, roll_n, use_zc, drop_int, drop_best_z)

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
        return _impl_TrackArrayMeasure_measure_signal_raw(self.ds, ref_ch, disk_r, roll_n, use_zc, drop_int, drop_best_z_raw)

    def mask_props(self, source, out_prefix=None, props=('area_px', 'eccentricity', 'solidity', 'perimeter_px', 'centroid_y_px', 'centroid_x_px'), connectivity=2, empty_value=float("nan")):
        """Measure morphology from a binary mask layer across the entire crop array and add scalar
measurement layers back onto `ca`.

Outputs are named: f"{out_prefix}__{prop}" and have dims equal to the non-(y,x) dims of `source`."""
        return _impl_TrackArrayMeasure_mask_props(self.ds, source=source, out_prefix=out_prefix, props=props, connectivity=connectivity, empty_value=empty_value)

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
        return _impl_TrackArrayMeasure_mask_skeleton_length(self.ds, source=source, out_prefix=out_prefix, method=method, connectivity=connectivity, empty_value=empty_value)

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
for that track."""
        return _impl_TrackArrayMeasure_track_length(self.ds, coord=coord, out_name=out_name, broadcast_like=broadcast_like)


TrackArrayMeasure.best_z_proj.__doc__ = _impl_TrackArrayMeasure_best_z_proj.__doc__
TrackArrayMeasure.best_z_proj.__wrapped__ = _impl_TrackArrayMeasure_best_z_proj
TrackArrayMeasure.best_z_proj.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayMeasure_best_z_proj).parameters.values())[1:])
TrackArrayMeasure.measure_signal.__doc__ = _impl_TrackArrayMeasure_measure_signal.__doc__
TrackArrayMeasure.measure_signal.__wrapped__ = _impl_TrackArrayMeasure_measure_signal
TrackArrayMeasure.measure_signal.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayMeasure_measure_signal).parameters.values())[1:])
TrackArrayMeasure.measure_signal_raw.__doc__ = _impl_TrackArrayMeasure_measure_signal_raw.__doc__
TrackArrayMeasure.measure_signal_raw.__wrapped__ = _impl_TrackArrayMeasure_measure_signal_raw
TrackArrayMeasure.measure_signal_raw.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayMeasure_measure_signal_raw).parameters.values())[1:])
TrackArrayMeasure.mask_props.__doc__ = _impl_TrackArrayMeasure_mask_props.__doc__
TrackArrayMeasure.mask_props.__wrapped__ = _impl_TrackArrayMeasure_mask_props
TrackArrayMeasure.mask_props.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayMeasure_mask_props).parameters.values())[1:])
TrackArrayMeasure.mask_skeleton_length.__doc__ = _impl_TrackArrayMeasure_mask_skeleton_length.__doc__
TrackArrayMeasure.mask_skeleton_length.__wrapped__ = _impl_TrackArrayMeasure_mask_skeleton_length
TrackArrayMeasure.mask_skeleton_length.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayMeasure_mask_skeleton_length).parameters.values())[1:])
TrackArrayMeasure.tracklist.__doc__ = _impl_TrackArrayMeasure_tracklist.__doc__
TrackArrayMeasure.tracklist.__wrapped__ = _impl_TrackArrayMeasure_tracklist
TrackArrayMeasure.tracklist.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayMeasure_tracklist).parameters.values())[1:])
TrackArrayMeasure.track_length.__doc__ = _impl_TrackArrayMeasure_track_length.__doc__
TrackArrayMeasure.track_length.__wrapped__ = _impl_TrackArrayMeasure_track_length
TrackArrayMeasure.track_length.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayMeasure_track_length).parameters.values())[1:])


from croparray.napari_view import montage_viewer as _impl_TrackArrayView_montage_viewer
from croparray.napari_view import manual_filter_montage as _impl_TrackArrayView_manual_filter_montage
from croparray.trackarray.napari_view import display_cell_and_tracks as _impl_TrackArrayView_display_cell_and_tracks

@dataclass
class TrackArrayView(_BaseAccessor):
    """Generated accessor methods."""
    def montage_viewer(self, row, col, show=('best_z', 'ch0_mask'), ch=0, z_index=0, viewer=None, image_contrast=None, tile_overlay_contrast=None, tile_overlay_opacity=0.35, default_blending='additive', colormaps=None, show_tile_text=True, tile_text_name='tile_text', tile_text_size=14, tile_text_color='white'):
        """Create a napari viewer for a montage with smart defaults:
  - mask-like layers => add_labels
  - image-like layers (has r,c) => add_image with robust contrast
  - non-xy layers (e.g., (track_id,t) or (n,t)) => drawn as per-tile overlays

Parameters
----------
ca_or_ta : xr.Dataset
    The CropArray/TrackArray dataset.
row, col : str
    The montage tiling dims you want (e.g. row="track_id", col="track_id"; row="n", col="t").
show : iterable[str]
    Variable names from the montage dataset to display.
ch : int
    Channel to show for image-like layers that have a 'ch' dimension.
z_index : int
    z-slice to show if z is present.
viewer : napari.Viewer, optional
    If provided, add layers into this viewer; otherwise create a new one.
image_contrast : 
    Percentiles for image layers (default 0–99.5). 
tile_overlay_contrast : 
    Percentiles for tile overlays (default 5–95).
tile_overlay_opacity : float
    Opacity used for tile overlays and labels.
default_blending : str
    Napari blending mode for image layers.
colormaps : dict[str,str], optional
    Colormap per layer name; falls back to 'gray' for images and 'magenta' for overlays.

Returns
-------
viewer, layers_dict
    layers_dict maps requested layer name -> napari layer object."""
        return _impl_TrackArrayView_montage_viewer(self.ds, row=row, col=col, show=show, ch=ch, z_index=z_index, viewer=viewer, image_contrast=image_contrast, tile_overlay_contrast=tile_overlay_contrast, tile_overlay_opacity=tile_overlay_opacity, default_blending=default_blending, colormaps=colormaps, show_tile_text=show_tile_text, tile_text_name=tile_text_name, tile_text_size=tile_text_size, tile_text_color=tile_text_color)

    def manual_filter_montage(self, output_dir, row, col, filter_name='manual_filter', show=('best_z', 'ch0_mask'), ch=0, z_index=0, viewer=None, write_back=True, overlay_opacity=0.35, single_click_delay_ms=100, colormaps=None, label_colors=None, show_click_info=False, show_tile_text=False):
        return _impl_TrackArrayView_manual_filter_montage(self.ds, output_dir=output_dir, row=row, col=col, filter_name=filter_name, show=show, ch=ch, z_index=z_index, viewer=viewer, write_back=write_back, overlay_opacity=overlay_opacity, single_click_delay_ms=single_click_delay_ms, colormaps=colormaps, label_colors=label_colors, show_click_info=show_click_info, show_tile_text=show_tile_text)

    def display_cell_and_tracks(self, tracks_df):
        """Display the maximum intensity projection of the images and the tracks in Napari.

Parameters:
img_croparray (numpy.ndarray): Array containing image data with dimensions (fov, t, z, x, y, ch).
tracks_df (pandas.DataFrame): DataFrame containing track information.

Returns:
napari.Viewer: The viewer instance with the images and tracks added."""
        return _impl_TrackArrayView_display_cell_and_tracks(self.ds, tracks_df)


TrackArrayView.montage_viewer.__doc__ = _impl_TrackArrayView_montage_viewer.__doc__
TrackArrayView.montage_viewer.__wrapped__ = _impl_TrackArrayView_montage_viewer
TrackArrayView.montage_viewer.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayView_montage_viewer).parameters.values())[1:])
TrackArrayView.manual_filter_montage.__doc__ = _impl_TrackArrayView_manual_filter_montage.__doc__
TrackArrayView.manual_filter_montage.__wrapped__ = _impl_TrackArrayView_manual_filter_montage
TrackArrayView.manual_filter_montage.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayView_manual_filter_montage).parameters.values())[1:])
TrackArrayView.display_cell_and_tracks.__doc__ = _impl_TrackArrayView_display_cell_and_tracks.__doc__
TrackArrayView.display_cell_and_tracks.__wrapped__ = _impl_TrackArrayView_display_cell_and_tracks
TrackArrayView.display_cell_and_tracks.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayView_display_cell_and_tracks).parameters.values())[1:])


from croparray.dataframe import variables_to_df as _impl_TrackArrayDF_variables_to_df
from croparray.trackarray.dataframe import create_tracks_df as _impl_TrackArrayDF_create_tracks_df
from croparray.trackarray.dataframe import track_signals_to_df as _impl_TrackArrayDF_track_signals_to_df

@dataclass
class TrackArrayDF(_BaseAccessor):
    """Generated accessor methods."""
    def variables_to_df(self, var_names):
        """Creates a pandas DataFrame from selected variables/coords of a CropArray.

Unlike xr.Dataset.to_dataframe(), this lets you select specific variables
and will broadcast lower-dimensional variables to a common target grid
when possible (e.g. per-track scalars to (track_id, t)).

Parameters
----------
ca : xarray.Dataset
    The CropArray dataset.
var_names : Sequence[str]
    Names of variables/coords in the dataset to include as columns.

Returns
-------
pandas.DataFrame
    DataFrame containing requested columns, with shared index columns
    (dims/coords) included once."""
        return _impl_TrackArrayDF_variables_to_df(self.ds, var_names)

    def create_tracks_df(self):
        return _impl_TrackArrayDF_create_tracks_df(self.ds)

    def track_signals_to_df(self):
        """Combine signal and signal_raw data from each channel into a single DataFrame.

Parameters:
- my_ta: xarray.Dataset containing 'signal' and 'signal_raw' data for each channel.

Returns:
- DataFrame: Combined DataFrame with columns for each signal and signal_raw."""
        return _impl_TrackArrayDF_track_signals_to_df(self.ds)


TrackArrayDF.variables_to_df.__doc__ = _impl_TrackArrayDF_variables_to_df.__doc__
TrackArrayDF.variables_to_df.__wrapped__ = _impl_TrackArrayDF_variables_to_df
TrackArrayDF.variables_to_df.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayDF_variables_to_df).parameters.values())[1:])
TrackArrayDF.create_tracks_df.__doc__ = _impl_TrackArrayDF_create_tracks_df.__doc__
TrackArrayDF.create_tracks_df.__wrapped__ = _impl_TrackArrayDF_create_tracks_df
TrackArrayDF.create_tracks_df.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayDF_create_tracks_df).parameters.values())[1:])
TrackArrayDF.track_signals_to_df.__doc__ = _impl_TrackArrayDF_track_signals_to_df.__doc__
TrackArrayDF.track_signals_to_df.__wrapped__ = _impl_TrackArrayDF_track_signals_to_df
TrackArrayDF.track_signals_to_df.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayDF_track_signals_to_df).parameters.values())[1:])


from croparray.crop_ops.apply import apply_crop_op as _impl_TrackArrayOps_apply_crop_op
from croparray.crop_ops.apply import apply_crop_op_legacy as _impl_TrackArrayOps_apply_crop_op_legacy
from croparray.crop_ops.apply import apply as _impl_TrackArrayOps_apply
from croparray.crop_ops.apply import apply_legacy as _impl_TrackArrayOps_apply_legacy

@dataclass
class TrackArrayOps(_BaseAccessor):
    """Generated accessor methods."""
    def apply_crop_op(self, func, source='best_z', group_dims=None, group_exclude_dims=('track_id', 't', 'ch', 'x', 'y', 'z'), channels=None, channel_dim='ch', input_core_dims=('x', 'y'), output_core_dims=('x', 'y'), out_name='ch{ch}_op', func_kwargs=None, compute_sum_xy=False, sum_name='{out}_sig', sum_dims=('x', 'y'), add_to_ds=True):
        """Apply a per-crop image operation, automatically grouping over non-core dims.

Default behavior:
  - Slices ds over all dims except group_exclude_dims (e.g. fov/exp/cell/rep)
  - Runs the legacy vectorized implementation on each slice (drop=True)
  - Concats slices back

This preserves the *exact* semantics of apply_crop_op_legacy within each slice,
which is critical for quantile/threshold based operations.

Power users: call apply_crop_op_legacy(...) directly to bypass grouping."""
        return _impl_TrackArrayOps_apply_crop_op(self.ds, func, source, group_dims=group_dims, group_exclude_dims=group_exclude_dims, channels=channels, channel_dim=channel_dim, input_core_dims=input_core_dims, output_core_dims=output_core_dims, out_name=out_name, func_kwargs=func_kwargs, compute_sum_xy=compute_sum_xy, sum_name=sum_name, sum_dims=sum_dims, add_to_ds=add_to_ds)

    def apply_crop_op_legacy(self, func, source='best_z', channels=None, channel_dim='ch', input_core_dims=('x', 'y'), output_core_dims=('x', 'y'), out_name='ch{ch}_op', func_kwargs=None, compute_sum_xy=False, sum_name='{out}_sig', sum_dims=('x', 'y'), add_to_ds=True):
        """Apply a per-crop image operation across a crop dataset using `xr.apply_ufunc`,
optionally iterating over channels, and optionally computing a per-crop scalar
summary (e.g., summed signal).

Conceptually, this function executes the “known-good” pattern:

    xr.apply_ufunc(
        func,
        da.sel(ch=<one channel>),
        input_core_dims=[["x", "y"]],
        output_core_dims=[["x", "y"]],
        kwargs=...,
        vectorize=True,
    )

and does so for every crop (and for each requested channel, if present).

Parameters
----------
ds
    The dataset containing crop data variables (e.g., a stack of crops across time/track/crop_id).
func
    A function to apply to each crop. It must accept an array-like input matching
    `input_core_dims` (typically a 2D image) and return an array-like output matching
    `output_core_dims`. Additional keyword arguments may be provided via `func_kwargs`.

    Typical examples: spot detection / QC, filtering, background subtraction, etc.
source
    Either the name of a DataArray in `ds` (default: `"best_z"`) or an explicit
    `xr.DataArray` to process.
channels
    Channel selection behavior when `channel_dim` exists in the source DataArray:

    - `None` (default): process **all** channels present in `da.coords[channel_dim]`.
    - `int`: process only that channel (e.g., `channels=0`).
    - `Sequence[int]`: process only those channels (e.g., `channels=[0]` or `[0, 1]`).

    If the source DataArray does **not** contain `channel_dim`, this argument is ignored
    and the operation is applied once (i.e., “single-channel” mode).
channel_dim
    Name of the channel dimension in the source DataArray (default: `"ch"`).
input_core_dims
    Core dimensions consumed by `func` (default: `("x", "y")`). These are passed to
    `xr.apply_ufunc(..., input_core_dims=[...])`.
output_core_dims
    Core dimensions produced by `func` (default: `("x", "y")`). These are passed to
    `xr.apply_ufunc(..., output_core_dims=[...])`.
out_name
    Output variable name template. If channel iteration is used, `{ch}` is formatted
    with the channel index (e.g., `"ch0_spots"`). If no channel dimension exists,
    `{ch}` is formatted as `"NA"` by default.
func_kwargs
    Optional keyword arguments forwarded to `func`. If `None`, treated as `{}`.
compute_sum_xy
    If True, compute an additional summary signal per crop by summing `out` over
    `sum_dims` and store it as a second output variable.
sum_name
    Name template for the summed-signal output, formatted with `{out}` set to the
    generated `out_name` (default: `"{out}_sig"`).
sum_dims
    Dimensions to sum over when `compute_sum_xy=True` (default: `("x", "y")`).
    Use this to control whether you sum only spatial dims, or include others.
add_to_ds
    If True (default), add outputs back into `ds` and return the updated dataset.
    If False, return a dict mapping output variable names to DataArrays.

Returns
-------
xr.Dataset or Dict[str, xr.DataArray]
    - If `add_to_ds=True`: the input dataset `ds`, augmented with new variables.
    - If `add_to_ds=False`: a dict of created outputs (and optional summaries).

Notes
-----
- This function relies on `vectorize=True` in `xr.apply_ufunc`, meaning `func` is
  applied independently across non-core dimensions (crop index, time, track_id, etc.).
- Channel handling is done via `da.sel({channel_dim: ch})` per channel.
- If you request channels that are not present in the DataArray coordinates,
  a clear `ValueError` is raised.

Examples
--------
Process all channels in `best_z`:

    ds = apply_crop_op(ds, spot_detect_and_qc, source="best_z",
                       out_name="ch{ch}_spots",
                       func_kwargs={"minmass": 150, "size": 3})

Process only channel 0:

    ds = apply_crop_op(ds, spot_detect_and_qc, source="best_z",
                       channels=[0],
                       out_name="ch{ch}_spots",
                       func_kwargs={"minmass": 150, "size": 3})

Process channels 0 and 1 and also compute summed signal per crop:

    ds = apply_crop_op(ds, spot_detect_and_qc, source="best_z",
                       channels=[0, 1],
                       out_name="ch{ch}_spots",
                       func_kwargs={"minmass": 150, "size": 3},
                       compute_sum_xy=True,
                       sum_name="{out}_sig")"""
        return _impl_TrackArrayOps_apply_crop_op_legacy(self.ds, func, source, channels=channels, channel_dim=channel_dim, input_core_dims=input_core_dims, output_core_dims=output_core_dims, out_name=out_name, func_kwargs=func_kwargs, compute_sum_xy=compute_sum_xy, sum_name=sum_name, sum_dims=sum_dims, add_to_ds=add_to_ds)

    def apply(self, func, source='best_z', group_dims=None, group_exclude_dims=('track_id', 't', 'ch', 'x', 'y', 'z'), channels=None, channel_dim='ch', input_core_dims=('x', 'y'), output_core_dims=('x', 'y'), out_name='ch{ch}_op', func_kwargs=None, compute_sum_xy=False, sum_name='{out}_sig', sum_dims=('x', 'y'), add_to_ds=True):
        """Apply a per-crop image operation, automatically grouping over non-core dims.

Default behavior:
  - Slices ds over all dims except group_exclude_dims (e.g. fov/exp/cell/rep)
  - Runs the legacy vectorized implementation on each slice (drop=True)
  - Concats slices back

This preserves the *exact* semantics of apply_crop_op_legacy within each slice,
which is critical for quantile/threshold based operations.

Power users: call apply_crop_op_legacy(...) directly to bypass grouping."""
        return _impl_TrackArrayOps_apply(self.ds, func, source, group_dims=group_dims, group_exclude_dims=group_exclude_dims, channels=channels, channel_dim=channel_dim, input_core_dims=input_core_dims, output_core_dims=output_core_dims, out_name=out_name, func_kwargs=func_kwargs, compute_sum_xy=compute_sum_xy, sum_name=sum_name, sum_dims=sum_dims, add_to_ds=add_to_ds)

    def apply_legacy(self, func, source='best_z', channels=None, channel_dim='ch', input_core_dims=('x', 'y'), output_core_dims=('x', 'y'), out_name='ch{ch}_op', func_kwargs=None, compute_sum_xy=False, sum_name='{out}_sig', sum_dims=('x', 'y'), add_to_ds=True):
        """Apply a per-crop image operation across a crop dataset using `xr.apply_ufunc`,
optionally iterating over channels, and optionally computing a per-crop scalar
summary (e.g., summed signal).

Conceptually, this function executes the “known-good” pattern:

    xr.apply_ufunc(
        func,
        da.sel(ch=<one channel>),
        input_core_dims=[["x", "y"]],
        output_core_dims=[["x", "y"]],
        kwargs=...,
        vectorize=True,
    )

and does so for every crop (and for each requested channel, if present).

Parameters
----------
ds
    The dataset containing crop data variables (e.g., a stack of crops across time/track/crop_id).
func
    A function to apply to each crop. It must accept an array-like input matching
    `input_core_dims` (typically a 2D image) and return an array-like output matching
    `output_core_dims`. Additional keyword arguments may be provided via `func_kwargs`.

    Typical examples: spot detection / QC, filtering, background subtraction, etc.
source
    Either the name of a DataArray in `ds` (default: `"best_z"`) or an explicit
    `xr.DataArray` to process.
channels
    Channel selection behavior when `channel_dim` exists in the source DataArray:

    - `None` (default): process **all** channels present in `da.coords[channel_dim]`.
    - `int`: process only that channel (e.g., `channels=0`).
    - `Sequence[int]`: process only those channels (e.g., `channels=[0]` or `[0, 1]`).

    If the source DataArray does **not** contain `channel_dim`, this argument is ignored
    and the operation is applied once (i.e., “single-channel” mode).
channel_dim
    Name of the channel dimension in the source DataArray (default: `"ch"`).
input_core_dims
    Core dimensions consumed by `func` (default: `("x", "y")`). These are passed to
    `xr.apply_ufunc(..., input_core_dims=[...])`.
output_core_dims
    Core dimensions produced by `func` (default: `("x", "y")`). These are passed to
    `xr.apply_ufunc(..., output_core_dims=[...])`.
out_name
    Output variable name template. If channel iteration is used, `{ch}` is formatted
    with the channel index (e.g., `"ch0_spots"`). If no channel dimension exists,
    `{ch}` is formatted as `"NA"` by default.
func_kwargs
    Optional keyword arguments forwarded to `func`. If `None`, treated as `{}`.
compute_sum_xy
    If True, compute an additional summary signal per crop by summing `out` over
    `sum_dims` and store it as a second output variable.
sum_name
    Name template for the summed-signal output, formatted with `{out}` set to the
    generated `out_name` (default: `"{out}_sig"`).
sum_dims
    Dimensions to sum over when `compute_sum_xy=True` (default: `("x", "y")`).
    Use this to control whether you sum only spatial dims, or include others.
add_to_ds
    If True (default), add outputs back into `ds` and return the updated dataset.
    If False, return a dict mapping output variable names to DataArrays.

Returns
-------
xr.Dataset or Dict[str, xr.DataArray]
    - If `add_to_ds=True`: the input dataset `ds`, augmented with new variables.
    - If `add_to_ds=False`: a dict of created outputs (and optional summaries).

Notes
-----
- This function relies on `vectorize=True` in `xr.apply_ufunc`, meaning `func` is
  applied independently across non-core dimensions (crop index, time, track_id, etc.).
- Channel handling is done via `da.sel({channel_dim: ch})` per channel.
- If you request channels that are not present in the DataArray coordinates,
  a clear `ValueError` is raised.

Examples
--------
Process all channels in `best_z`:

    ds = apply_crop_op(ds, spot_detect_and_qc, source="best_z",
                       out_name="ch{ch}_spots",
                       func_kwargs={"minmass": 150, "size": 3})

Process only channel 0:

    ds = apply_crop_op(ds, spot_detect_and_qc, source="best_z",
                       channels=[0],
                       out_name="ch{ch}_spots",
                       func_kwargs={"minmass": 150, "size": 3})

Process channels 0 and 1 and also compute summed signal per crop:

    ds = apply_crop_op(ds, spot_detect_and_qc, source="best_z",
                       channels=[0, 1],
                       out_name="ch{ch}_spots",
                       func_kwargs={"minmass": 150, "size": 3},
                       compute_sum_xy=True,
                       sum_name="{out}_sig")"""
        return _impl_TrackArrayOps_apply_legacy(self.ds, func, source, channels=channels, channel_dim=channel_dim, input_core_dims=input_core_dims, output_core_dims=output_core_dims, out_name=out_name, func_kwargs=func_kwargs, compute_sum_xy=compute_sum_xy, sum_name=sum_name, sum_dims=sum_dims, add_to_ds=add_to_ds)


TrackArrayOps.apply_crop_op.__doc__ = _impl_TrackArrayOps_apply_crop_op.__doc__
TrackArrayOps.apply_crop_op.__wrapped__ = _impl_TrackArrayOps_apply_crop_op
TrackArrayOps.apply_crop_op.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayOps_apply_crop_op).parameters.values())[1:])
TrackArrayOps.apply_crop_op_legacy.__doc__ = _impl_TrackArrayOps_apply_crop_op_legacy.__doc__
TrackArrayOps.apply_crop_op_legacy.__wrapped__ = _impl_TrackArrayOps_apply_crop_op_legacy
TrackArrayOps.apply_crop_op_legacy.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayOps_apply_crop_op_legacy).parameters.values())[1:])
TrackArrayOps.apply.__doc__ = _impl_TrackArrayOps_apply.__doc__
TrackArrayOps.apply.__wrapped__ = _impl_TrackArrayOps_apply
TrackArrayOps.apply.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayOps_apply).parameters.values())[1:])
TrackArrayOps.apply_legacy.__doc__ = _impl_TrackArrayOps_apply_legacy.__doc__
TrackArrayOps.apply_legacy.__wrapped__ = _impl_TrackArrayOps_apply_legacy
TrackArrayOps.apply_legacy.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayOps_apply_legacy).parameters.values())[1:])


from croparray.napari_view import montage_viewer as _impl_TrackArrayNapari_montage_viewer
from croparray.napari_view import manual_filter_montage as _impl_TrackArrayNapari_manual_filter_montage
from croparray.trackarray.napari_view import display_cell_and_tracks as _impl_TrackArrayNapari_display_cell_and_tracks

@dataclass
class TrackArrayNapari(_BaseAccessor):
    """Generated accessor methods."""
    def montage_viewer(self, row, col, show=('best_z', 'ch0_mask'), ch=0, z_index=0, viewer=None, image_contrast=None, tile_overlay_contrast=None, tile_overlay_opacity=0.35, default_blending='additive', colormaps=None, show_tile_text=True, tile_text_name='tile_text', tile_text_size=14, tile_text_color='white'):
        """Create a napari viewer for a montage with smart defaults:
  - mask-like layers => add_labels
  - image-like layers (has r,c) => add_image with robust contrast
  - non-xy layers (e.g., (track_id,t) or (n,t)) => drawn as per-tile overlays

Parameters
----------
ca_or_ta : xr.Dataset
    The CropArray/TrackArray dataset.
row, col : str
    The montage tiling dims you want (e.g. row="track_id", col="track_id"; row="n", col="t").
show : iterable[str]
    Variable names from the montage dataset to display.
ch : int
    Channel to show for image-like layers that have a 'ch' dimension.
z_index : int
    z-slice to show if z is present.
viewer : napari.Viewer, optional
    If provided, add layers into this viewer; otherwise create a new one.
image_contrast : 
    Percentiles for image layers (default 0–99.5). 
tile_overlay_contrast : 
    Percentiles for tile overlays (default 5–95).
tile_overlay_opacity : float
    Opacity used for tile overlays and labels.
default_blending : str
    Napari blending mode for image layers.
colormaps : dict[str,str], optional
    Colormap per layer name; falls back to 'gray' for images and 'magenta' for overlays.

Returns
-------
viewer, layers_dict
    layers_dict maps requested layer name -> napari layer object."""
        return _impl_TrackArrayNapari_montage_viewer(self.ds, row=row, col=col, show=show, ch=ch, z_index=z_index, viewer=viewer, image_contrast=image_contrast, tile_overlay_contrast=tile_overlay_contrast, tile_overlay_opacity=tile_overlay_opacity, default_blending=default_blending, colormaps=colormaps, show_tile_text=show_tile_text, tile_text_name=tile_text_name, tile_text_size=tile_text_size, tile_text_color=tile_text_color)

    def manual_filter_montage(self, output_dir, row, col, filter_name='manual_filter', show=('best_z', 'ch0_mask'), ch=0, z_index=0, viewer=None, write_back=True, overlay_opacity=0.35, single_click_delay_ms=100, colormaps=None, label_colors=None, show_click_info=False, show_tile_text=False):
        return _impl_TrackArrayNapari_manual_filter_montage(self.ds, output_dir=output_dir, row=row, col=col, filter_name=filter_name, show=show, ch=ch, z_index=z_index, viewer=viewer, write_back=write_back, overlay_opacity=overlay_opacity, single_click_delay_ms=single_click_delay_ms, colormaps=colormaps, label_colors=label_colors, show_click_info=show_click_info, show_tile_text=show_tile_text)

    def display_cell_and_tracks(self, tracks_df):
        """Display the maximum intensity projection of the images and the tracks in Napari.

Parameters:
img_croparray (numpy.ndarray): Array containing image data with dimensions (fov, t, z, x, y, ch).
tracks_df (pandas.DataFrame): DataFrame containing track information.

Returns:
napari.Viewer: The viewer instance with the images and tracks added."""
        return _impl_TrackArrayNapari_display_cell_and_tracks(self.ds, tracks_df)


TrackArrayNapari.montage_viewer.__doc__ = _impl_TrackArrayNapari_montage_viewer.__doc__
TrackArrayNapari.montage_viewer.__wrapped__ = _impl_TrackArrayNapari_montage_viewer
TrackArrayNapari.montage_viewer.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayNapari_montage_viewer).parameters.values())[1:])
TrackArrayNapari.manual_filter_montage.__doc__ = _impl_TrackArrayNapari_manual_filter_montage.__doc__
TrackArrayNapari.manual_filter_montage.__wrapped__ = _impl_TrackArrayNapari_manual_filter_montage
TrackArrayNapari.manual_filter_montage.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayNapari_manual_filter_montage).parameters.values())[1:])
TrackArrayNapari.display_cell_and_tracks.__doc__ = _impl_TrackArrayNapari_display_cell_and_tracks.__doc__
TrackArrayNapari.display_cell_and_tracks.__wrapped__ = _impl_TrackArrayNapari_display_cell_and_tracks
TrackArrayNapari.display_cell_and_tracks.__signature__ = inspect.Signature(parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(inspect.signature(_impl_TrackArrayNapari_display_cell_and_tracks).parameters.values())[1:])



def install_generated_accessors(CropArray, TrackArray):
    """Attach generated accessors as @property on wrapper classes."""
    # CropArray accessors
    CropArray.plot = property(lambda self, _A=CropArrayPlot: _A(self))
    CropArray.measure = property(lambda self, _A=CropArrayMeasure: _A(self))
    CropArray.df = property(lambda self, _A=CropArrayDF: _A(self))
    CropArray.view = property(lambda self, _A=CropArrayView: _A(self))
    CropArray.track = property(lambda self, _A=CropArrayTrack: _A(self))
    CropArray.ops = property(lambda self, _A=CropArrayOps: _A(self))
    CropArray.napari = property(lambda self, _A=CropArrayNapari: _A(self))
    CropArray.build = property(lambda self, _A=CropArrayBuild: _A(self))
    CropArray.io = property(lambda self, _A=CropArrayIO: _A(self))

    # TrackArray accessors
    TrackArray.tplot = property(lambda self, _A=TrackArrayPlot: _A(self))
    TrackArray.tmeasure = property(lambda self, _A=TrackArrayMeasure: _A(self))
    TrackArray.tview = property(lambda self, _A=TrackArrayView: _A(self))
    TrackArray.tdf = property(lambda self, _A=TrackArrayDF: _A(self))
    TrackArray.ops = property(lambda self, _A=TrackArrayOps: _A(self))
    TrackArray.napari = property(lambda self, _A=TrackArrayNapari: _A(self))

