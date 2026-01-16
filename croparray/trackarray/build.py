import numpy as np
import pandas as pd
import xarray as xr
import itertools

# Make track_id dimension in crop array (that has been tracked)
def track_array_single(ca, as_object: bool = False):
    """
    Create a track-array dataset from a tracked crop-array dataset by grouping
    entries by unique track IDs (stored in ca['id']).

    Note: In the croparray pipeline, ca['id'] is treated as the active identifier.
    After tracking, ca['id'] is overwritten to store track IDs (0 indicates untracked /
    filtered) and the original 'id' is moved to 'spot_id'.

    Parameters
    ----------
    ca : xarray.Dataset
        Crop-array dataset that contains an `id` variable with per-(fov,n,t,...) track IDs.

    Returns
    -------
    xarray.Dataset
        Track-array dataset with dimension `track_id` and (typically) `fov`
        as a real dimension. Variables are aligned across tracks with fill_value=0.
    """
    # Accept either CropArray object or raw xarray.Dataset
    if hasattr(ca, "ds"):  # CropArray wrapper
        ca = ca.ds

    my_ids = np.unique(ca["id"].values)
    my_ids = my_ids[pd.notnull(my_ids)]
    my_ids = my_ids[my_ids != 0]

    if len(my_ids) == 0:
        empty = xr.Dataset()
        if as_object:
            from .object import TrackArray
            return TrackArray(empty)
        return empty

    my_das = []
    for tid in my_ids:
        g = ca.groupby("id")[tid]
        stacked_dim = next((d for d in g.dims if d.startswith("stacked_")), None)
        if stacked_dim is None:
            raise ValueError("Could not find stacked dimension created by groupby('id').")

        temp = (
            g.reset_index(stacked_dim)
             .sortby("t")
             .reset_coords("n", drop=True)
             .set_index({stacked_dim: "t"})
             .rename({stacked_dim: "t"})
        )

        if "fov" in temp.coords:
            fovs = np.unique(temp["fov"].values)
            if len(fovs) != 1:
                raise ValueError(f"Track {tid} spans multiple FOVs: {fovs}")
            fov0 = int(fovs[0])
            temp = temp.drop_vars("fov").expand_dims(fov=[fov0])

        my_das.append(temp)

    my_taz = xr.concat(my_das, dim=pd.Index(my_ids, name="track_id"), fill_value=0)
    my_taz = my_taz.transpose("track_id", "fov", "t", "z", "y", "x", "ch", missing_dims="ignore")

    if as_object:
        from .object import TrackArray
        return TrackArray(my_taz)

    return my_taz



def track_array(
    ca_in,
    as_object: bool = False,
    *,
    base_dims: tuple[str, ...] = ("t", "z", "y", "x", "ch", "n"),
):
    """
    Create a track-array dataset from a (possibly concatenated) crop-array dataset.

    This function is the generalized, "safe" entry point for building TrackArrays
    from CropArrays that may have additional dimensions beyond the standard imaging
    dimensions. In croparray workflows it is common to concatenate CropArrays
    across conditions or samples, creating new dimensions such as:

        - Exp
        - Cell
        - Batch
        - Replicate
        - Treatment
        - (and others)

    Track IDs stored in `ca_in["id"]` are typically assigned *within* an individual
    acquisition (e.g., within one Cell / FOV / movie). Therefore, running a single
    `groupby("id")` across a concatenated dataset can accidentally merge tracks
    from different Cells/Exps/etc. whenever track IDs overlap numerically (which
    is common when IDs start from 1 in each acquisition).

    Strategy:
        1) Discover all "grouping dimensions": every dimension in `ca_in.dims`
           except the base imaging/time dimensions (default: t, z, y, x, ch) and `n`.
        2) Iterate over all unique combinations of those grouping dimensions.
        3) For each slice, call `track_array_single()` (your original implementation).
        4) Concatenate the resulting TrackArrays back together, preserving grouping
           dimensions in the output.

    Parameters
    ----------
    ca_in : xarray.Dataset
        Crop-array dataset containing an `id` variable with per-(fov,n,t,...) track IDs.
        This dataset may be a "native" crop-array, or a concatenation across new
        dimensions such as Exp/Cell/Batch/etc.

    as_object : bool, optional
        If True, return a TrackArray object wrapper. If False, return an xarray.Dataset.
        Default: False.

    base_dims : tuple[str, ...], optional
        Dimensions treated as "intrinsic imaging/time" dimensions that should *not*
        be used for grouping. Default: ("t","z","y","x","ch","n").
        Notes:
          - We exclude `n` from grouping because `track_array_single()` intentionally
            drops `n` in track view.
          - `fov` is *not* in base_dims by default, so it will be discovered and grouped
            over if it exists (which matches your requirement).

    Returns
    -------
    xarray.Dataset or TrackArray
        Track-array dataset with dimension `track_id` and with any discovered grouping
        dims (e.g., Exp, Cell, fov, Batch, ...) preserved as real dimensions.
        Variables are aligned across tracks with fill_value=0 during concatenation.

    Notes
    -----
    - Missing IDs:
        This function drops entries where `id` is NaN to avoid `KeyError: nan` during groupby.
    - Untracked IDs:
        By convention your pipeline uses 0 for "untracked/filtered". Those are excluded
        inside `track_array_single()` (and can also be filtered there).
    - Performance:
        This loops over all groups (Exp×Cell×fov×...). In typical use this is still
        efficient because each slice is modest and your per-slice `track_array_single()`
        is already vectorized with xarray operations.
    """
    # Accept either CropArray object or raw xarray.Dataset
    if hasattr(ca_in, "ds"):  # CropArray wrapper
        ca_in = ca_in.ds

    # Discover grouping dimensions automatically.
    # This includes fov (if present) and any "concat dims" such as Exp/Cell/Batch/etc.
    group_dims = [d for d in ca_in.dims if d not in base_dims]

    # track_array_single() already preserves fov as a dimension; don't group over it
    survive_dims = {"fov"}  # add others here if needed
    group_dims = [d for d in group_dims if d not in survive_dims]


    # If there are no extra dimensions to group over, fall back to single behavior.
    if not group_dims:
        return track_array_single(ca_in, as_object=as_object)

    tas = []
    keys = []

    # Iterate over every combination of grouping coordinate values.
    for vals in itertools.product(*[ca_in[d].values for d in group_dims]):
        sel = dict(zip(group_dims, vals))
        ca_sub = ca_in.sel(sel)

        # Drop missing track IDs (prevents KeyError: nan during groupby)
        if "id" in ca_sub:
            ca_sub = ca_sub.where(ca_sub["id"].notnull(), drop=True)

        # Build TrackArray for this slice using the original implementation.
        ta = track_array_single(ca_sub, as_object=False)

        # Skip empty results
        if ta.sizes.get("track_id", 0) == 0:
            continue

        # Preserve grouping dims on output by promoting them to real dimensions.
        # (Some dims may already exist, e.g., fov often survives as a dimension.)
        for d, v in sel.items():
            if d not in ta.dims:
                ta = ta.expand_dims({d: [v]})

        tas.append(ta)
        keys.append(vals)

    # If nothing produced tracks, return an empty dataset/object.
    if not tas:
        empty = xr.Dataset()
        if as_object:
            from .object import TrackArray
            return TrackArray(empty)
        return empty

    # Concatenate all per-group TrackArrays along an explicitly named dimension,
    # attach the MultiIndex as coordinates, then unstack back into group dims.
    mi = pd.MultiIndex.from_tuples(keys, names=group_dims)
    stack_dim = "__group__"

    out = xr.concat(tas, dim=stack_dim, fill_value=0)
    out = out.assign_coords({stack_dim: mi})
    out = out.unstack(stack_dim)


    # Prefer a consistent dimension order
    out = out.transpose(*group_dims, "track_id", "fov", "t", "z", "y", "x", "ch", missing_dims="ignore")

    if as_object:
        from .object import TrackArray
        return TrackArray(out)
    return out
