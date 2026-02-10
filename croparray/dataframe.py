from __future__ import annotations

from typing import Sequence
import pandas as pd


__all__ = ["variables_to_df"]


def variables_to_df(ca, var_names: Sequence[str]):
    """
    Creates a pandas DataFrame from selected variables/coords of a CropArray.

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
        (dims/coords) included once.
    """
    # Check existence (variables OR coords)
    for name in var_names:
        if name not in ca:
            raise ValueError(f"'{name}' not found in the provided xarray dataset (var or coord).")

    # Pick a target dimensionality: ordered union of dims across requested vars
    target_dims: list[str] = []
    for name in var_names:
        for d in ca[name].dims:
            if d not in target_dims:
                target_dims.append(d)

    # Build a template DataArray with those dims so we can broadcast_like it.
    # We only need coords for these dims; if some dim has no coord, xarray
    # will still broadcast by length.
    template = ca[var_names[0]]
    # Expand template to include any missing target dims using broadcasting against coords
    # by creating a minimal "shape-only" template:
    for d in target_dims:
        if d not in template.dims:
            # create a length-only coordinate if dataset has it, else use range
            if d in ca.dims:
                coord = ca.coords[d] if d in ca.coords else pd.RangeIndex(ca.dims[d], name=d)
            else:
                # Shouldn't happen: dims should be in ca.dims if they appear in da.dims
                coord = pd.RangeIndex(template.sizes.get(d, 0), name=d)
            template = template.expand_dims({d: coord})

    template = template.transpose(*target_dims)

    dfs = []
    for name in var_names:
        da = ca[name]

        # Validate broadcastability: da.dims must be subset of target_dims
        if any(d not in target_dims for d in da.dims):
            raise ValueError(
                f"Cannot align '{name}' with target dims {tuple(target_dims)}. "
                f"'{name}' has dims {da.dims} which are not a subset."
            )

        # Broadcast + transpose into canonical order
        da = da.broadcast_like(template).transpose(*target_dims)

#        df = da.to_dataframe(name=name).reset_index()
        df0 = da.to_dataframe(name=name)

        # If any index level names already exist as columns, drop the column copies
        idx_names = [n for n in df0.index.names if n is not None]
        collisions = [n for n in idx_names if n in df0.columns]
        if collisions:
            df0 = df0.drop(columns=collisions)

        df = df0.reset_index()

        dfs.append(df)

    final_df = pd.concat(dfs, axis=1)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    return final_df
