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

    # --------------------------------------------
    # Split variables into those that actually use n
    # vs metadata/static vars that should NOT acquire
    # a track_id dimension.
    # --------------------------------------------
    track_vars = [name for name, da in ca.data_vars.items() if "n" in da.dims]
    meta_vars = [name for name, da in ca.data_vars.items() if "n" not in da.dims]

    ca_tracks = ca[track_vars]
    ca_meta = ca[meta_vars] if meta_vars else xr.Dataset()

    ids = np.unique(ca_tracks["track_id"].values)
    ids = ids[pd.notnull(ids)]
    ids = ids[ids >= 0]

    if len(ids) == 0:
        empty = xr.merge([xr.Dataset(), ca_meta], compat="override")
        if as_object:
            from .object import TrackArray
            return TrackArray(empty)
        return empty

    gb = ca_tracks.groupby("track_id")
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

    # Keep missing values as NaN rather than forcing zeros everywhere
    out_tracks = xr.concat(
        tracks,
        dim=pd.Index(ids, name="track_id"),
    )

    out_tracks = out_tracks.transpose(
        "track_id",
        "t",
        "z",
        "y",
        "x",
        "ch",
        missing_dims="ignore",
    )

    # Merge static / metadata vars back unchanged
    out = xr.merge([out_tracks, ca_meta], compat="override")

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

    tas_tracks = []   # vars that contain track_id
    metas = []        # vars that do NOT contain track_id
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

        # Drop any singleton split dims that survived unexpectedly
        for d in split_dims:
            if d in ta.dims:
                if ta.sizes[d] != 1:
                    raise ValueError(f"Expected singleton dim {d}, got {ta.sizes[d]}")
                ta = ta.isel({d: 0}, drop=True)

        drop_coords = [d for d in split_dims if d in ta.coords and d not in ta.dims]
        if drop_coords:
            ta = ta.reset_coords(drop_coords, drop=True)

        # Split into variables that should carry track_id vs metadata that should not
        track_vars = [k for k, v in ta.data_vars.items() if "track_id" in v.dims]
        meta_vars = [k for k, v in ta.data_vars.items() if "track_id" not in v.dims]

        ta_tracks = ta[track_vars] if track_vars else xr.Dataset()
        ta_meta = ta[meta_vars] if meta_vars else xr.Dataset()

        tas_tracks.append(ta_tracks)
        metas.append(ta_meta)
        keys.append(tuple(sel[d] for d in split_dims))

    if not tas_tracks:
        empty = xr.Dataset()
        if as_object:
            from .object import TrackArray
            return TrackArray(empty)
        return empty

    # -----------------------------
    # Build tracked variables only
    # -----------------------------
    out_tracks = xr.concat(
        tas_tracks,
        dim=pd.Index(range(len(tas_tracks)), name=stack_dim),
        fill_value=0,
    )

    mi = pd.MultiIndex.from_tuples(keys, names=split_dims)
    out_tracks = out_tracks.assign_coords({stack_dim: mi})
    out_tracks = out_tracks.unstack(stack_dim)

    # -----------------------------
    # Build metadata separately
    # -----------------------------
    # Metadata vars do not have track_id and should remain indexed only by split_dims
    meta_rows = []
    for meta in metas:
        row = {}
        for name, da in meta.data_vars.items():
            # after track_array_single on a single split slice, these should be scalars
            # or at least not depend on track_id
            if da.ndim == 0:
                row[name] = da.item()
            else:
                # keep full values if they still have dimensions; xr will handle them below
                row[name] = da
        meta_rows.append(row)

    # Build a dataset from metadata pieces
    # We do this variable-by-variable so non-scalar metadata is preserved cleanly.
    meta_ds_vars = {}
    if metas:
        all_meta_names = set()
        for meta in metas:
            all_meta_names.update(meta.data_vars)

        for name in sorted(all_meta_names):
            arrs = [meta[name] for meta in metas if name in meta]

            # every meta variable should be present in every slice
            if len(arrs) != len(metas):
                raise ValueError(f"Metadata variable {name!r} missing from some groups")

            first = arrs[0]

            if first.ndim == 0:
                meta_ds_vars[name] = xr.DataArray(
                    [a.item() for a in arrs],
                    dims=(stack_dim,),
                )
            else:
                tmp = xr.concat(
                    arrs,
                    dim=pd.Index(range(len(arrs)), name=stack_dim),
                )
                meta_ds_vars[name] = tmp

    out_meta = xr.Dataset(meta_ds_vars)
    out_meta = out_meta.assign_coords({stack_dim: mi})
    out_meta = out_meta.unstack(stack_dim)

    # Merge tracked variables + metadata
    out = xr.merge([out_tracks, out_meta], compat="override")

    order = [*split_dims, "track_id", "t", "z", "y", "x", "ch"]
    out = out.transpose(*order, missing_dims="ignore")

    if as_object:
        from .object import TrackArray
        return TrackArray(out)

    return out
