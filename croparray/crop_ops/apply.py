from typing import Any, Callable, Dict, Optional, Sequence, Union, Iterable, List
import xarray as xr
import itertools
from functools import wraps

__all__ = [
    "apply_crop_op",          # NEW default (grouped wrapper)
    "apply_crop_op_legacy",   # old behavior (power users)
    "apply",                  # alias for apply_crop_op
    "apply_legacy",            # alias for apply_legacy
]

def _infer_group_dims(
    da: xr.DataArray,
    *,
    group_dims: Optional[Sequence[str]],
    group_exclude_dims: Sequence[str],
) -> List[str]:
    """Choose which dims to slice over."""
    if group_dims is not None:
        return [d for d in group_dims if d in da.dims]
    return [d for d in da.dims if d not in set(group_exclude_dims)]

def apply_crop_op(
    ds: xr.Dataset,
    func: Callable,
    source: Union[str, xr.DataArray] = "best_z",
    *,
    # grouping knobs
    group_dims: Optional[Sequence[str]] = None,
    group_exclude_dims: Sequence[str] = ("track_id", "t", "ch", "x", "y", "z"),
    # --- same API as legacy ---
    channels: Optional[Union[int, Sequence[int]]] = None,
    channel_dim: str = "ch",
    input_core_dims: Sequence[str] = ("x", "y"),
    output_core_dims: Sequence[str] = ("x", "y"),
    out_name: str = "ch{ch}_op",
    func_kwargs: Optional[Dict[str, Any]] = None,
    compute_sum_xy: bool = False,
    sum_name: str = "{out}_sig",
    sum_dims: Sequence[str] = ("x", "y"),
    add_to_ds: bool = True,
) -> Union[xr.Dataset, Dict[str, xr.DataArray]]:
    """
    Apply a per-crop image operation, automatically grouping over non-core dims.

    Default behavior:
      - Slices ds over all dims except group_exclude_dims (e.g. fov/exp/cell/rep)
      - Runs the legacy vectorized implementation on each slice (drop=True)
      - Concats slices back

    This preserves the *exact* semantics of apply_crop_op_legacy within each slice,
    which is critical for quantile/threshold based operations.

    Power users: call apply_crop_op_legacy(...) directly to bypass grouping.
    """
    func_kwargs = {} if func_kwargs is None else dict(func_kwargs)

    da = ds[source] if isinstance(source, str) else source
    group_dims_eff = _infer_group_dims(da, group_dims=group_dims, group_exclude_dims=group_exclude_dims)

    # No grouping needed -> just run legacy once
    if not group_dims_eff:
        out = apply_crop_op_legacy(
            ds,
            func,
            source=source,
            channels=channels,
            channel_dim=channel_dim,
            input_core_dims=input_core_dims,
            output_core_dims=output_core_dims,
            out_name=out_name,
            func_kwargs=func_kwargs,
            compute_sum_xy=compute_sum_xy,
            sum_name=sum_name,
            sum_dims=sum_dims,
            add_to_ds=add_to_ds,
        )
        return out

    # Build coordinate grid over group dims
    coords_list = [list(ds.coords[d].values) for d in group_dims_eff]
    slices: List[xr.Dataset] = []

    for idx in itertools.product(*coords_list):
        sel = dict(zip(group_dims_eff, idx))

        # IMPORTANT: drop=True (default) to match legacy semantics
        sub = ds.sel(sel)

        sub2 = apply_crop_op_legacy(
            sub,
            func,
            source=source,
            channels=channels,
            channel_dim=channel_dim,
            input_core_dims=input_core_dims,
            output_core_dims=output_core_dims,
            out_name=out_name,
            func_kwargs=func_kwargs,
            compute_sum_xy=compute_sum_xy,
            sum_name=sum_name,
            sum_dims=sum_dims,
            add_to_ds=True,  # always add inside slice
        )
        slices.append(sub2)

    # Concat back along group dims (preserve original coord order)
    # We do it one dim at a time for clarity/stability.
    out_ds = slices
    for d_i, d in enumerate(group_dims_eff[::-1]):
        labels = list(ds.coords[d].values)
        step = len(labels)

        new_list = []
        for j in range(0, len(out_ds), step):
            block = xr.concat(out_ds[j : j + step], dim=d).assign_coords({d: labels})
            new_list.append(block)
        out_ds = new_list

    merged = out_ds[0]

    if add_to_ds:
        # mutate in place like legacy apply() expects
        ds.update(merged)
        return ds

    # If user asked for dict outputs, mimic legacy: return created vars only
    created = {k: merged[k] for k in merged.data_vars if k not in ds.data_vars}    
    return created

def apply_crop_op_legacy(
    ds: xr.Dataset,
    func: Callable,
    source: Union[str, xr.DataArray] = "best_z",
    *,
    channels: Optional[Union[int, Sequence[int]]] = None,
    channel_dim: str = "ch",
    input_core_dims: Sequence[str] = ("x", "y"),
    output_core_dims: Sequence[str] = ("x", "y"),
    out_name: str = "ch{ch}_op",
    func_kwargs: Optional[Dict[str, Any]] = None,
    compute_sum_xy: bool = False,
    sum_name: str = "{out}_sig",
    sum_dims: Sequence[str] = ("x", "y"),
    add_to_ds: bool = True,
) -> Union[xr.Dataset, Dict[str, xr.DataArray]]:
    """
    Apply a per-crop image operation across a crop dataset using `xr.apply_ufunc`,
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
                           sum_name="{out}_sig")
    """
    func_kwargs = {} if func_kwargs is None else dict(func_kwargs)

    da = ds[source] if isinstance(source, str) else source

    # Normalize channel selection.
    def _normalize_channels(
        da_: xr.DataArray,
        channels_: Optional[Union[int, Sequence[int]]],
        channel_dim_: str,
    ) -> Sequence[Optional[int]]:
        # No channel dimension => run once.
        if channel_dim_ not in da_.dims:
            return [None]

        available = list(da_.coords[channel_dim_].values)

        # Default: all channels.
        if channels_ is None:
            # Coerce to python ints where reasonable (for clean formatting later).
            return [int(c) for c in available]

        # Accept a single int.
        if isinstance(channels_, (int,)):
            requested = [int(channels_)]
        else:
            requested = [int(c) for c in channels_]

        missing = [c for c in requested if c not in list(map(int, available))]
        if missing:
            raise ValueError(
                f"Requested channels {missing} not found in {channel_dim_} coords. "
                f"Available: {list(map(int, available))}"
            )

        return requested

    channel_list = _normalize_channels(da, channels, channel_dim)

    created: Dict[str, xr.DataArray] = {}

    for ch in channel_list:
        da_in = da.sel({channel_dim: ch}) if ch is not None else da

        out = xr.apply_ufunc(
            func,
            da_in,
            input_core_dims=[list(input_core_dims)],
            output_core_dims=[list(output_core_dims)],
            kwargs=func_kwargs,
            vectorize=True,
        )

        ch_label = "NA" if ch is None else int(ch)
        out_var = out_name.format(ch=ch_label)
        out.name = out_var
        created[out_var] = out

        if compute_sum_xy:
            sig = out.sum(list(sum_dims))
            sig_name = sum_name.format(out=out_var)
            sig.name = sig_name
            created[sig_name] = sig

    if add_to_ds:
        for k, v in created.items():
            ds[k] = v
        return ds

    return created

@wraps(apply_crop_op)
def apply(ds, *args, **kwargs):
    return apply_crop_op(ds, *args, **kwargs)

@wraps(apply_crop_op_legacy)
def apply_legacy(ds, *args, **kwargs):
    return apply_crop_op_legacy(ds, *args, **kwargs)