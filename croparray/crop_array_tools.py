# !/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Python code to create, manipulate, and analyze an array of crops from TIF images or videos.
Created: Summer of 2020 (updated Jan 2026 by Tim Stasevich)
Authors: Tim Stasevich & Luis Aguilera.
'''

# Conventions.
# module_name, package_name, ClassName, method_name,
# ExceptionName, function_name, GLOBAL_CONSTANT_NAME,
# global_var_name, instance_var_name, function_parameter_name, local_var_name.

from __future__ import annotations  

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
import inspect
from typing import Sequence

from .crop_array_object import CropArray

from .io import open_croparray, open_croparray_zarr
from .build import _create_crop_array_dataset, create_crop_array
from .measure import best_z_proj, measure_signal, measure_signal_raw, mask_props
from .plot import montage
from .napari_view import view_montage
from .dataframe import variables_to_df
from .tracking import (
    perform_tracking_with_exclusions,
    to_track_array,
)
from .trackarray.plot import plot_trackarray_crops, plot_track_signal_traces
from .trackarray.build import track_array,track_array_single
from .trackarray.dataframe import create_tracks_df, track_signals_to_df
from .trackarray.napari_view import display_cell_and_tracks
from .crop_ops.measure import spot_detect_and_qc, binarize_crop, binarize_crop_manual
from .crop_ops.apply import apply_crop_op
from .raw.detect import detecting_spots
from .raw.track import tracking_spots



##### Modules
def print_banner():
    print(" \n"
        "░█████╗░██████╗░░█████╗░██████╗░░█████╗░██████╗░██████╗░░█████╗░██╗░░░██╗\n"
        "██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗╚██╗░██╔╝\n"
        "██║░░╚═╝██████╔╝██║░░██║██████╔╝███████║██████╔╝██████╔╝███████║░╚████╔╝░\n"
        "██║░░██╗██╔══██╗██║░░██║██╔═══╝░██╔══██║██╔══██╗██╔══██╗██╔══██║░░╚██╔╝░░\n"
        "╚█████╔╝██║░░██║╚█████╔╝██║░░░░░██║░░██║██║░░██║██║░░██║██║░░██║░░░██║░░░\n"
        "░╚════╝░╚═╝░░╚═╝░╚════╝░╚═╝░░░░░╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚═╝╚═╝░░╚═╝░░░╚═╝░░░\n"
        "                                     by : Luis Aguilera and Tim Stasevich \n\n"        )


def concat(
    *,
    cas: Sequence[CropArray],
    dim: str = "Exp",
    labels: Sequence[str] | None = None,
    start_index: int = 1,
    join: str = "exact",
    compat: str = "equals",
    coords: str = "minimal",
    combine_attrs: str = "override",
) -> CropArray:
    """Package-level wrapper for :meth:`CropArray.concat`."""
    return CropArray.concat(
        cas,
        dim=dim,
        labels=labels,
        start_index=start_index,
        join=join,
        compat=compat,
        coords=coords,
        combine_attrs=combine_attrs,
    )

# Keep top-level wrapper docs/signature in sync with the canonical classmethod
concat.__doc__ = CropArray.concat.__doc__
concat.__signature__ = inspect.signature(CropArray.concat)


