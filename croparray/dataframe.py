from __future__ import annotations

import pandas as pd

__all__ = ["variables_to_df"]

def variables_to_df(ca, var_names):
    """
    Creates a pandas DataFrame from selected variables of a CropArray.

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
        flattened index).
    """
    # Check if variables exist in the dataset
    for var in var_names:
        if var not in ca:
            raise ValueError(f"'{var}' not found in the provided xarray dataset.")

    # Require same dims as a SET (order-insensitive), then normalize order
    canonical_dims = ca[var_names[0]].dims
    canonical_set = set(canonical_dims)

    for var in var_names[1:]:
        if set(ca[var].dims) != canonical_set:
            raise ValueError(
                "Variables do not have matching dimensions (ignoring order). "
                f"{var_names[0]} has dimensions {canonical_dims} while {var} has "
                f"dimensions {ca[var].dims}"
            )

    # Convert each variable to a dataframe after transposing to canonical order
    dfs = []
    for var in var_names:
        da = ca[var].transpose(*canonical_dims)
        df = da.to_dataframe(name=var).reset_index()
        dfs.append(df)

    # Concatenate side-by-side and drop duplicate index columns
    final_df = pd.concat(dfs, axis=1)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    return final_df
