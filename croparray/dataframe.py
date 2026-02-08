from __future__ import annotations

import pandas as pd

__all__ = ["variables_to_df"]

# Pull out variables in a crop array to a dataframe
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

    # Check if the variables have the same dimensions
    dims = ca[var_names[0]].dims
    for var in var_names[1:]:
        if ca[var].dims != dims:
            raise ValueError(f"Variables do not have matching dimensions. {var_names[0]} has dimensions {dims} while {var} has dimensions {ca[var].dims}")

    # Convert each variable to a dataframe and concatenate them
    dfs = [ca[var].to_dataframe().reset_index(level=list(range(len(dims)))) for var in var_names]
    
    final_df = pd.concat(dfs, axis=1)
    # Drop duplicate columns if any arise due to the reset index operation
    final_df = final_df.loc[:,~final_df.columns.duplicated()]
    
    return final_df