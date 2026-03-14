from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import itertools

__all__ = [
    # ... existing ...
    "track_array_single, track_array"
]

def track_array_single(ca, as_object: bool = False):
    import numpy as np
    import pandas as pd
    import xarray as xr

    if hasattr(ca, "ds"):
        ca = ca.ds

    if "track_id" not in ca:
        raise ValueError("track_array_single requires 'track_id' in dataset")

    if "track_id" in ca.coords and "track_id" not in ca.data_vars:
        ca = ca.reset_coords("track_id")

    ids = np.unique(ca["track_id"].values)
    ids = ids[pd.notnull(ids)]
    ids = ids[ids >= 0]

    if len(ids) == 0:
        empty = xr.Dataset()
        if as_object:
            from .object import TrackArray
            return TrackArray(empty)
        return empty

    gb = ca.groupby("track_id")
    tracks = []

    for tid in ids:

        g = gb[tid]

        stacked_dim = next((d for d in g.dims if d.startswith("stacked_")), None)
        if stacked_dim is None:
            raise ValueError("Expected stacked dimension from groupby")

        temp = (
            g.reset_index(stacked_dim)
             .sortby("t")
             .drop_vars("n", errors="ignore")
        )

        temp = temp.drop_vars("track_id", errors="ignore")
        if "track_id" in temp.coords:
            temp = temp.reset_coords("track_id", drop=True)

        if "t" in temp:
            tvals = np.asarray(temp["t"].values)
            _, idx = np.unique(tvals, return_index=True)
            temp = temp.isel({stacked_dim: np.sort(idx)})

        temp = (
            temp.set_index({stacked_dim: "t"})
                .rename({stacked_dim: "t"})
        )

        tracks.append(temp)

    out = xr.concat(tracks, dim=pd.Index(ids, name="track_id"), fill_value=0)

    out = out.transpose(
        "track_id",
        "t",
        "z",
        "y",
        "x",
        "ch",
        missing_dims="ignore",
    )

    if as_object:
        from .object import TrackArray
        return TrackArray(out)

    return out

def track_array(
    ca_in,
    as_object: bool = False,
    *,
    base_dims=("t", "z", "y", "x", "ch", "n"),
):

    import itertools
    import pandas as pd
    import xarray as xr

    if hasattr(ca_in, "ds"):
        ca_in = ca_in.ds

    split_dims = [d for d in ca_in.dims if d not in base_dims]

    if not split_dims:
        return track_array_single(ca_in, as_object=as_object)

    tas = []
    keys = []
    stack_dim = "__group__"

    for vals in itertools.product(*[ca_in[d].values for d in split_dims]):

        sel = dict(zip(split_dims, vals))
        ca_sub = ca_in.sel(sel)

        if "track_id" in ca_sub:
            track_id = ca_sub["track_id"]

            if "n" not in track_id.dims:
                raise ValueError(f"track_id must include 'n'; got dims={track_id.dims}")

            # Reduce across every dimension except n
            reduce_dims = tuple(d for d in track_id.dims if d != "n")
            valid = track_id.notnull()
            if reduce_dims:
                valid = valid.any(dim=reduce_dims)

            ca_sub = ca_sub.sel(n=valid)

        ta = track_array_single(ca_sub, as_object=False)

        if ta.sizes.get("track_id", 0) == 0:
            continue

        for d in split_dims:
            if d in ta.dims:
                if ta.sizes[d] != 1:
                    raise ValueError(
                        f"Expected singleton dim {d}, got {ta.sizes[d]}"
                    )
                ta = ta.isel({d: 0}, drop=True)

        drop_coords = [d for d in split_dims if d in ta.coords and d not in ta.dims]
        if drop_coords:
            ta = ta.reset_coords(drop_coords, drop=True)

        tas.append(ta)
        keys.append(tuple(sel[d] for d in split_dims))

    if not tas:
        empty = xr.Dataset()
        if as_object:
            from .object import TrackArray
            return TrackArray(empty)
        return empty

    out = xr.concat(
        tas,
        dim=pd.Index(range(len(tas)), name=stack_dim),
        fill_value=0,
    )

    mi = pd.MultiIndex.from_tuples(keys, names=split_dims)

    out = out.assign_coords({stack_dim: mi})
    out = out.unstack(stack_dim)

    order = [*split_dims, "track_id", "t", "z", "y", "x", "ch"]

    out = out.transpose(*order, missing_dims="ignore")

    if as_object:
        from .object import TrackArray
        return TrackArray(out)

    return out


