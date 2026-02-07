# croparray/build.py  

from __future__ import annotations


from typing import Optional, Sequence, Tuple
import numpy as np
import xarray as xr
import pandas as pd

try:
    __all__
except NameError:
    __all__ = []
__all__ += ["standardize_video_axes"]

_CANONICAL = ("fov", "f", "z", "y", "x", "ch")

def _norm_axis(a: str) -> str:
    a = a.strip().lower()
    if a in ("t", "time", "frame", "frames"):
        return "f"
    if a in ("c", "chan", "channel", "channels"):
        return "ch"
    if a in ("pos", "position"):
        return "fov"
    if a in ("slice", "plane"):
        return "z"
    return a

def standardize_video_axes(
    video: np.ndarray,
    axes: Sequence[str],
    *,
    add_missing_singletons: bool = True,
) -> np.ndarray:
    """
    Reorder raw video into croparray canonical order: (fov, f, z, y, x, ch).

    `axes` must describe the CURRENT order of `video`, e.g. res.axes from ca.gui.label_video_axes().
    """
    video = np.asarray(video)
    axes = tuple(_norm_axis(a) for a in axes)

    if video.ndim != len(axes):
        raise ValueError(f"video.ndim={video.ndim} but len(axes)={len(axes)}")

    # y/x required
    for req in ("y", "x"):
        if axes.count(req) != 1:
            raise ValueError(f"Axis '{req}' must be present exactly once. Got axes={axes}")

    # optional axes unique
    for opt in ("fov", "f", "z", "ch"):
        if axes.count(opt) > 1:
            raise ValueError(f"Axis '{opt}' can appear at most once. Got axes={axes}")

    video2 = video
    axes2 = list(axes)

    if add_missing_singletons:
        # add missing optional dims as singleton at end
        for opt in ("fov", "f", "z", "ch"):
            if opt not in axes2:
                video2 = np.expand_dims(video2, axis=-1)
                axes2.append(opt)

    missing = [ax for ax in _CANONICAL if ax not in axes2]
    if missing:
        raise ValueError(f"Missing axes {missing} after normalization. Got axes={axes2}")

    perm = [axes2.index(ax) for ax in _CANONICAL]
    return np.transpose(video2, axes=perm)


def _create_crop_array_dataset(video, df, **kwargs):
    """
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
    - Crops that cannot be extracted due to out-of-bounds coordinates remain zero-filled.
    """

    import numpy as np
    import pandas as pd
    import xarray as xr

    # ----------------
    # kwargs / metadata
    # ----------------
    xy_pad = int(kwargs.get("xy_pad", 5))
    z_pad = int(kwargs.get("z_pad", 0))  # used only if df has zc

    my_dx = kwargs.get("dx", 1)
    my_dy = kwargs.get("dy", 1)
    my_dz = kwargs.get("dz", 1)
    my_dt = kwargs.get("dt", 1)
    units = kwargs.get("units", ["space", "time"])
    name = kwargs.get("name", "video_filename")
    date = kwargs.get("date", "video_date")

    df = df.copy()

    # ----------------
    # validate / schema
    # ----------------
    required = ["fov", "f", "yc", "xc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}. Required: {required}")

    # id
    if "id" not in df.columns:
        df["id"] = np.arange(len(df), dtype=np.int64)
    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(-1).astype(np.int64)

    # track_id
    if "track_id" in df.columns:
        df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce").fillna(-1).astype(np.int16)
    else:
        df["track_id"] = np.int16(-1)

    # frame
    df["f"] = pd.to_numeric(df["f"], errors="coerce").fillna(0).astype(np.int32)

    # global float coords (movie coordinates)
    df["xc"] = pd.to_numeric(df["xc"], errors="coerce")
    df["yc"] = pd.to_numeric(df["yc"], errors="coerce")

    has_zc = "zc" in df.columns
    if has_zc:
        df["zc"] = pd.to_numeric(df["zc"], errors="coerce")
    else:
        # keep a float column for uniformity
        df["zc"] = np.nan

    # slab mode only if z_pad>0 and zc column exists
    use_z_slab = (z_pad > 0) and has_zc

    # ------------
    # video dims
    # ------------
    n_fov, n_frames, z_slices, height_y, width_x, n_channels = list(video.shape)
    print("Original video dimensions:", video.shape)

    # homography per channel (xy only)
    homography = kwargs.get("homography", [np.eye(3) for _ in range(n_channels)])

    # pad video in xy to allow edge crops
    npad = ((0, 0), (0, 0), (0, 0), (xy_pad + 1, xy_pad + 1), (xy_pad + 1, xy_pad + 1), (0, 0))
    video_pad = np.pad(video, pad_width=npad, mode="constant", constant_values=0)
    print("Padded video dimensions:", video_pad.shape)

    # -------------------------
    # compute per-frame crop index n
    # -------------------------
    df["n"] = df.groupby(["fov", "f"]).cumcount()
    n_spots_max = int(df["n"].max()) + 1 if len(df) else 0
    print("Max # of spots per frame:", n_spots_max)

    # -------------------------
    # allocate arrays
    # -------------------------
    z_out = (2 * z_pad + 1) if use_z_slab else z_slices

    my_crops_all = np.zeros(
        (n_fov, n_spots_max, n_frames, z_out, 2 * xy_pad + 1, 2 * xy_pad + 1, n_channels),
        dtype=np.int32,
    )

    # global (movie) float coords per channel
    my_xc_all = np.full((n_fov, n_spots_max, n_frames, n_channels), np.nan, dtype=np.float32)
    my_yc_all = np.full((n_fov, n_spots_max, n_frames, n_channels), np.nan, dtype=np.float32)

    # global (movie) float z coord (broadcast to all channels)
    my_zc_all = np.full((n_fov, n_spots_max, n_frames, n_channels), np.nan, dtype=np.float32)

    # pixel indices in movie coords (rounded)
    my_xc_pix_all = np.full((n_fov, n_spots_max, n_frames, n_channels), -1, dtype=np.int16)
    my_yc_pix_all = np.full((n_fov, n_spots_max, n_frames, n_channels), -1, dtype=np.int16)
    my_zc_pix_all = np.full((n_fov, n_spots_max, n_frames, n_channels), -1, dtype=np.int16)

    # padded xy indices for slicing the padded video
    my_xc_pad_all = np.full((n_fov, n_spots_max, n_frames, n_channels), -1, dtype=np.int16)
    my_yc_pad_all = np.full((n_fov, n_spots_max, n_frames, n_channels), -1, dtype=np.int16)

    # local z selector for best-z selection: index into stored z dimension
    # slab mode: z_pos = z_pad if valid; else -1
    # full mode: can set z_pos = zc_pix if desired downstream; here we store it explicitly.
    my_z_pos_all = np.full((n_fov, n_spots_max, n_frames, n_channels), -1, dtype=np.int16)

    # -------------------------
    # extra numeric df columns -> layers (fov,n,t)
    # -------------------------
    base_coord_cols = {"fov", "f", "yc", "xc", "zc", "n"}
    my_columns = [c for c in df.columns if c not in base_coord_cols]

    # ensure id/track_id exist as layers
    for required_layer in ("id", "track_id"):
        if required_layer not in my_columns:
            my_columns.append(required_layer)

    my_layers = np.zeros((len(my_columns), n_fov, n_spots_max, n_frames), dtype=np.int16)
    print("Shape of extra my_layers numpy array:", my_layers.shape)

    # -------------------------
    # fill arrays
    # -------------------------
    my_fov_ind = 0
    for my_fov in df["fov"].unique():
        df_fov = df[df["fov"] == my_fov]
        for my_f in np.sort(df_fov["f"].unique()):
            my_spots = df_fov[df_fov["f"] == my_f].copy()
            if len(my_spots) == 0:
                continue

            my_ns = my_spots["n"].to_numpy(dtype=int)

            # fill scalar layers (id/track_id and other numeric columns)
            for col_counter, col in enumerate(my_columns):
                vals = pd.to_numeric(my_spots[col], errors="coerce")
                if col not in ("id", "track_id"):
                    vals = vals.round()
                vals = vals.fillna(-1).astype(np.int16).to_numpy()
                my_layers[col_counter, my_fov_ind, : len(vals), int(my_f)] = vals

            # z global float and z pixel (broadcast to all channels)
            zc_float = my_spots["zc"].to_numpy(dtype=np.float32)  # may contain nan
            zc_pix = np.round(zc_float).astype(np.float32)  # still float; we'll handle nan next
            zc_pix = np.where(np.isfinite(zc_pix), zc_pix, -1).astype(np.int16)

            # compute per-channel homography-corrected xy (global float), then pixel and padded indices
            xy_in = my_spots[["xc", "yc"]].to_numpy(dtype=np.float32)  # global float input
            x_global = np.full((n_channels, len(my_spots)), np.nan, dtype=np.float32)
            y_global = np.full((n_channels, len(my_spots)), np.nan, dtype=np.float32)

            for ch_i in range(n_channels):
                H = homography[ch_i]
                # apply homography to each point
                temp = [np.dot(H, np.array([p[0], p[1], 1.0], dtype=np.float32))[0:2] for p in xy_in]
                temp = np.asarray(temp, dtype=np.float32)
                x_global[ch_i] = temp[:, 0]
                y_global[ch_i] = temp[:, 1]

            x_pix = np.round(x_global).astype(np.float32)
            y_pix = np.round(y_global).astype(np.float32)
            x_pix = np.where(np.isfinite(x_pix), x_pix, -1).astype(np.int16)
            y_pix = np.where(np.isfinite(y_pix), y_pix, -1).astype(np.int16)

            x_pad = (x_pix.astype(np.int32) + (xy_pad + 1)).astype(np.int16)
            y_pad = (y_pix.astype(np.int32) + (xy_pad + 1)).astype(np.int16)

            # local z selector z_pos:
            if use_z_slab:
                # valid zc_pix => z_pos=z_pad, else -1 (we also mark out-of-bounds later)
                z_pos = np.where(zc_pix >= 0, z_pad, -1).astype(np.int16)
            else:
                # full-z storage: z_pos can mirror zc_pix (global == local index)
                z_pos = zc_pix.astype(np.int16)

            # write coord layers
            for ch_i in range(n_channels):
                my_xc_all[my_fov_ind, my_ns, int(my_f), ch_i] = x_global[ch_i]
                my_yc_all[my_fov_ind, my_ns, int(my_f), ch_i] = y_global[ch_i]

                my_xc_pix_all[my_fov_ind, my_ns, int(my_f), ch_i] = x_pix[ch_i]
                my_yc_pix_all[my_fov_ind, my_ns, int(my_f), ch_i] = y_pix[ch_i]

                my_xc_pad_all[my_fov_ind, my_ns, int(my_f), ch_i] = x_pad[ch_i]
                my_yc_pad_all[my_fov_ind, my_ns, int(my_f), ch_i] = y_pad[ch_i]

                my_zc_all[my_fov_ind, my_ns, int(my_f), ch_i] = zc_float
                my_zc_pix_all[my_fov_ind, my_ns, int(my_f), ch_i] = zc_pix
                my_z_pos_all[my_fov_ind, my_ns, int(my_f), ch_i] = z_pos

            # extract crops
            for j, n_idx in enumerate(my_ns):
                for ch_i in range(n_channels):
                    # padded indices for slicing padded movie
                    xc_p = int(x_pad[ch_i, j])
                    yc_p = int(y_pad[ch_i, j])

                    # invalid xy -> leave zeros
                    if xc_p < 0 or yc_p < 0:
                        my_z_pos_all[my_fov_ind, n_idx, int(my_f), ch_i] = -1
                        continue

                    y0 = yc_p - xy_pad
                    y1 = yc_p + xy_pad + 1
                    x0 = xc_p - xy_pad
                    x1 = xc_p + xy_pad + 1

                    if use_z_slab:
                        zc_g = int(zc_pix[j])  # global z index into original stack
                        if zc_g < 0:
                            my_z_pos_all[my_fov_ind, n_idx, int(my_f), ch_i] = -1
                            continue

                        z0 = zc_g - z_pad
                        z1 = zc_g + z_pad + 1  # exclusive

                        # out of bounds => keep zeros and mark invalid
                        if (z0 < 0) or (z1 > z_slices) or (z1 <= z0):
                            my_z_pos_all[my_fov_ind, n_idx, int(my_f), ch_i] = -1
                            continue

                        slab = video_pad[my_fov_ind, int(my_f), z0:z1, y0:y1, x0:x1, ch_i]
                        if slab.shape[0] != (2 * z_pad + 1):
                            my_z_pos_all[my_fov_ind, n_idx, int(my_f), ch_i] = -1
                            continue

                        my_crops_all[my_fov_ind, n_idx, int(my_f), :, :, :, ch_i] = slab
                    else:
                        # full z
                        my_crops_all[my_fov_ind, n_idx, int(my_f), :, :, :, ch_i] = video_pad[
                            my_fov_ind, int(my_f), :, y0:y1, x0:x1, ch_i
                        ]

        my_fov_ind += 1

    # -------------------------
    # build coordinates
    # -------------------------
    n = xr.DataArray(np.arange(n_spots_max).astype(np.int16), attrs={"long_name": "crop count"})
    t = xr.DataArray(np.arange(n_frames) * my_dt, attrs={"units": units[1], "long_name": "time"})

    if use_z_slab:
        z_vals = (np.arange(2 * z_pad + 1) - z_pad) * my_dz
        z_long = "axial z (slab-relative, centered at 0)"
    else:
        z_vals = np.arange(z_slices) * my_dz
        z_long = "axial z-distance"

    z = xr.DataArray(z_vals, attrs={"units": units[0], "long_name": z_long})
    y = xr.DataArray(np.arange(-xy_pad, xy_pad + 1) * my_dy, attrs={"units": units[0], "long_name": "radial y-distance"})
    x = xr.DataArray(np.arange(-xy_pad, xy_pad + 1) * my_dx, attrs={"units": units[0], "long_name": "radial x-distance"})
    ch = np.arange(n_channels)
    fov = np.arange(n_fov)

    # -------------------------
    # dataset variables
    # -------------------------
    dx = xr.DataArray(my_dx, coords=[], dims=[], attrs={"units": units[0], "long_name": "x-resolution"})
    dy = xr.DataArray(my_dy, coords=[], dims=[], attrs={"units": units[0], "long_name": "y-resolution"})
    dz = xr.DataArray(my_dz, coords=[], dims=[], attrs={"units": units[0], "long_name": "z-resolution"})
    dt = xr.DataArray(my_dt, coords=[], dims=[], attrs={"units": units[1], "long_name": "temporal resolution"})
    xy_pad_da = xr.DataArray(xy_pad, coords=[], dims=[], attrs={"units": "pixels"})

    intensity = xr.DataArray(
        my_crops_all,
        coords=[fov, n, t, z, y, x, ch],
        dims=["fov", "n", "t", "z", "y", "x", "ch"],
        attrs={"units": "intensity (a.u.)", "long_name": "intensity"},
    )

    xc = xr.DataArray(my_xc_all, coords=[fov, n, t, ch], dims=["fov", "n", "t", "ch"],
                      attrs={"units": "pixels", "long_name": "xc (global movie coords, float)"})
    yc = xr.DataArray(my_yc_all, coords=[fov, n, t, ch], dims=["fov", "n", "t", "ch"],
                      attrs={"units": "pixels", "long_name": "yc (global movie coords, float)"})
    zc = xr.DataArray(my_zc_all, coords=[fov, n, t, ch], dims=["fov", "n", "t", "ch"],
                      attrs={"units": "z-index", "long_name": "zc (global movie coords, float)"})


    xc_pix = xr.DataArray(my_xc_pix_all, coords=[fov, n, t, ch], dims=["fov", "n", "t", "ch"],
                          attrs={"units": "pixels", "long_name": "xc pixel index (global, rounded)"})
    yc_pix = xr.DataArray(my_yc_pix_all, coords=[fov, n, t, ch], dims=["fov", "n", "t", "ch"],
                          attrs={"units": "pixels", "long_name": "yc pixel index (global, rounded)"})
    zc_pix = xr.DataArray(my_zc_pix_all, coords=[fov, n, t, ch], dims=["fov", "n", "t", "ch"],
                          attrs={"units": "index", "long_name": "zc pixel index (global, rounded; -1 unknown)"})


    xc_pad = xr.DataArray(my_xc_pad_all, coords=[fov, n, t, ch], dims=["fov", "n", "t", "ch"],
                          attrs={"units": "pixels", "long_name": "xc index into padded video"})
    yc_pad = xr.DataArray(my_yc_pad_all, coords=[fov, n, t, ch], dims=["fov", "n", "t", "ch"],
                          attrs={"units": "pixels", "long_name": "yc index into padded video"})

    z_pos = xr.DataArray(my_z_pos_all, coords=[fov, n, t, ch], dims=["fov", "n", "t", "ch"],
                         attrs={"units": "index", "long_name": "local z index for best-z selection (-1 invalid)"})

    # optional layers
    optional_layers = [xr.DataArray(my_layers[i], coords=[fov, n, t], dims=["fov", "n", "t"]) for i in range(len(my_columns))]
    dict1 = dict(zip(my_columns, optional_layers))
    if "id" in dict1:
        dict1["id"].attrs.update({"units": "index", "long_name": "spot id"})
    if "track_id" in dict1:
        dict1["track_id"].attrs.update({"units": "index", "long_name": "track assignment (-1 = untracked)"})

    ds = xr.Dataset(
        {
            "int": intensity,
            "xc": xc, "yc": yc, "zc": zc,
            "xc_pix": xc_pix, "yc_pix": yc_pix, "zc_pix": zc_pix,
            "xc_pad": xc_pad, "yc_pad": yc_pad,
            "z_pos": z_pos,
            "dx": dx, "dy": dy, "dz": dz, "dt": dt,
            "xy_pad": xy_pad_da,
            **dict1,
        },
        attrs={"name": name, "date": date},
    )
    return ds


# def _create_crop_array_dataset(video, df, **kwargs): 
#     """
#     Creates a crop x-array from a tif video and a dataframe containing the ids and coordinates of spots of interest. Cropping is only performed in the lateral xy-plane (so each crop has all z-slices in the video). Padding in the xy-plane by zeros is added to create crops for spots that are too close to the edge of the video. 
#     Parameters
#     ----------
#     video: numpy array
#         A 6D numpy array with intensity information from a tif video. The dimensions of the numpy array must be ordered (fov, f, z, y, x, ch), where fov = field of view, f = frame, z = axial z-coordinate, y = lateral y-coordinate, x = lateral x-coordinate, and ch = channels. Note any dimension can have length one (eg. single fov videos would have an fov dimension of length one or a single channel video would have a ch dimension of length one).  
#     df: pandas dataframe
#         A dataframe with the ids and coordinates of selected spots for making crops from video. Minimally, the dataframe must have 5 columns (1) 'fov': the fov number for each spot; can also be a filename for each fov. (2) 'id': the integer id of each spot. (3) 'f': integer frame number of each spot starting from zero. (4) 'yc': the lateral y-coordinate of the spot for centering the crop in y, (5) 'xc': the lateral x-coodinate of the spot for centering the crop in x. Any additional columns must be numeric and will be automatically converted to individual x-arrays in the crop array dataset that have the column header as a name.
#     xy_pad: int, optional
#         The amount of pixels to pad the centered pixel for each crop in the lateral x and y directions. Note the centered pixel is defined as the pixel containing the coordinates (xc, yc) for each crop. As an example, if xy_pad = 5, then each crop in the crop array will have x and y dimensions of 11 = 2*xy_pad + 1.
#     dx: int, optional
#         The size of pixels in the x-direction.
#     dy: int, optional 
#         The size of pixels in the y-direction.   
#     dz: int, optional 
#         The size of pixels in the z-direction.
#     dt: int, optional
#         The time between sequential frames in the video.   
#     video_filename: str, optional
#         The name of the tif video file.
#     video_date: str, optional
#         The date the video was acquired, in the form 'yyyy-mm-dd'.
#     homography: numpy array, optional
#         A list of 3x3 transformation matrices, one for each channel. This is to correct for any misalignments between channels. If a channel is not be adjusted, the unit 3 x 3 matrix can be used.   
#     Returns
#     ---------
#     A crop x-array dataset ca (i.e. crop array) containing 9 default x-arrays (+ optional x-arrays based on optional inputted df columns).
#     Coordinates of x-array dataset: fov, n, t, z, y, x, ch
#         fov = [0, 1, ... n_fov]
#         n = [0, 1, ... n_crops]
#         t = [0, 1, ... n_frames] dt
#         z = [0, 1, ... z_slices] dz
#         y = [-xy_pad, xy_pad + 1, ... xy_pad] dy
#         x = [-xy_pad, xy_pad + 1, ... xy_pad] dx
#         ch = [0, 1, ... n_channels]
#     Attributes of dataset: filename, date
#     X-arrays in dataset:
#     1. ca.int -- dims: (fov, n, t, z, y, x ch); attributes: 'units'; uint16
#         An X-array containing the intensities of all crops in the crop array.
#     2. ca.id -- dims: (fov, n, t); attributes: 'units'; uint16
#         An x-array containing the ids of the crops in the video.
#     3. ca.track_id -- dims: (fov, n, t); int
#         Track assignment for each spot. -1 indicates untracked.
#     4. ca.yc -- dims: (fov, n, t, ch); attributes: 'units'; uint16
#         An x-array containing the yc coordinates of the crops in the video.
#     5. ca.xc -- dims: (fov, n, t, ch); attributes: 'units'; uint16
#         An x-array containing the xc coordinates of the crops in the video.
#     6. ca.xy_pad -- dims: (); attributes: 'units'; uint16
#         A 1D array containing xy-pad. 
#     7. ca.dt -- dims: (); attributes: 'units'; float
#         A 1D arary containing dt.
#     8. ca.dz -- dims: (); attributes: 'units'; float
#         A 1D arary containing dz.
#     9. ca.dy -- dims: (); attributes: 'units'; float
#         A 1D arary containing dy.
#     10. ca.dx -- dims: (); attributes: 'units'; float
#         A 1D arary containing dx.
#     """
#     # Get the optional key word arguments (kwargs):
#     xy_pad = kwargs.get('xy_pad', 5) # default padding of 5 pixels
#     my_dx = kwargs.get('dx', 1)
#     my_dy = kwargs.get('dy', 1)
#     my_dz = kwargs.get('dz', 1)
#     my_dt = kwargs.get('dt', 1)
#     units = kwargs.get('units',['space','time'])
#     name = kwargs.get('name', 'video_filename')
#     date = kwargs.get('date', 'video_date') 

#     # ----------------------------
#     # Enforce id / track_id schema
#     # ----------------------------

#     # Ensure spot id exists
#     if 'id' not in df.columns:
#         # Generate a stable spot id if missing
#         # This is per-row; uniqueness across concatenation is handled later
#         df = df.copy()
#         df['id'] = np.arange(len(df), dtype=np.int64)

#     # Ensure integer id
#     df['id'] = df['id'].astype(np.int64)

#     # Ensure track_id exists
#     if 'track_id' in df.columns:
#         # Normalize: NaN -> -1, allow 0 as valid
#         df['track_id'] = (
#             pd.to_numeric(df['track_id'], errors='coerce')
#             .fillna(-1)
#             .astype(np.int32)
#         )
#     else:
#         df = df.copy()
#         df['track_id'] = -1


#     # Get dimensions of video
#     n_fov, n_frames, z_slices, height_y, width_x, n_channels = list(video.shape)
#     print('Original video dimensions: ', video.shape)

#     # Get homography matrix; default is a 3D identity matrix for transformating x, y, and z
#     homography = kwargs.get('homography', [np.eye(3) for i in np.arange(n_channels)])
    
#     # Pad video in xy-lateral direction by xy_pad so crops can be made for all spots
#     npad = ((0,0),(0,0), (0,0), (xy_pad+1,xy_pad+1), (xy_pad+1,xy_pad+1), (0,0))
#     video = np.pad(video, pad_width=npad, mode='constant', constant_values=0)
#     print('Padded video dimensions: ', video.shape)

#     # Create the 'n' spot/crop counter column for indexing df by 'fov', 'n', and 'f':
#     my_frames = np.arange(n_frames) # A list of the frames
#     my_crops = df.groupby(['fov','f']) # Group spots by frame and fov
#     df['n'] = my_crops.cumcount() # Create a new column 'n' as a cumulative counter of spots per frame per fov
#     n_spots_max = df['n'].max() + 1 # Get max number of spots to create numpy fov x n x f array to hold all crops (note add 1 since start from zero)
#     print('Max # of spots per frame: ', n_spots_max)
    
#     # Create empty array to hold all crop array crops with coordinate dimensions fov, n, f, z, y, x, ch:
#     my_crops_all = np.zeros((n_fov, n_spots_max, n_frames, z_slices, 2*xy_pad+1, 2*xy_pad+1, n_channels))
#     print('Shape of numpy array to hold all crop intensity data: ', my_crops_all.shape)
    
#     # Create arrays for xc and yc with coordinate dimensions (fov, n, f, ch)
#     # Note these arrays do depend on channel ch, and we'll use the homography matrix to correct channel (xc, yc) coordinates 
#     my_xc_all = np.zeros((n_fov, n_spots_max, n_frames, n_channels))
#     my_yc_all = np.zeros((n_fov, n_spots_max, n_frames, n_channels))
#     print('Shape of xc and yc numpy arrays: ', my_xc_all.shape)

#     # Create arrays to hold crop 'id' and any other extra numeric columns in df
#     # Note these will create x-arrays with coordinate dimensions (fov, n, f)
#     my_columns=list(df.columns)   # List of columns for making x-arrays
#     my_columns.remove('fov')        # but need to remove the common layer coordinates 'fov', 'f', 'yc', and 'xc'
#     my_columns.remove('f') 
#     my_columns.remove('yc')
#     my_columns.remove('xc')
#     my_columns.remove('n')  # also need to remove this newly made column
#     my_layers = np.zeros((len(my_columns), n_fov, n_spots_max, n_frames)) 
#     print('Shape of extra my_layers numpy array: ', my_layers.shape)
    
#     # Assign crops to empty arrays defined above
#     my_fov_ind = 0  # index counter for fov (in case fovs are listed as filenames)
#     for my_fov in df['fov'].unique():
#         for my_f in np.sort(df['f'].unique()): # frames MUST be integers counting from 0       
#             # collect all crops at my_fov and my_f 
#             my_spots = df[(df['f'] == my_f) & (df['fov'] == my_fov)]
#             # get list of all crop counter ns:
#             my_ns = my_spots['n'].values.astype(int) # this preserves order in df         
#             # fill my_layers numpy arrays
#             col_counter = 0
#             for col in my_columns:   # these are the other columns besides 'n', 'f, and 'fov' in the 
#                 #my_vals = my_spots[col].round().values.astype(int) # this preserves order in df, so same as my_ns above
#                 my_vals = my_spots[col].values

#                 # Only round coordinates-like quantities; IDs should not be rounded
#                 if col not in ('id', 'track_id'):
#                     my_vals = np.round(my_vals)

#                 my_vals = my_vals.astype(int)

#                 my_layers[col_counter, my_fov_ind, :len(my_vals), my_f] = my_vals 
#                 col_counter = col_counter + 1
#             # create temp arrays to hold (x, y) coords in all channels for all spots at my_fov and my_f
#             my_x = np.zeros((n_channels, len(my_ns)))
#             my_y = np.zeros((n_channels, len(my_ns)))
#             # correct (x,y) coordinates of all crops at my_f and my_fov using homography matrix
#             # use the list of homography matrices to correct channels )        
#             for ch in np.arange(n_channels):
#                 if len(my_spots)>0:   # correct other channels using same homography (since green/blue are image on same camera)
#                     temp = [list(np.dot(homography[ch],np.array([pos[0],pos[1],1]))[0:2]) 
#                             for pos in my_spots[['xc','yc']].values]
#                     my_x[ch], my_y[ch] = np.array(temp).T
#                     my_x[ch] = (my_x[ch] + xy_pad + 1).round(0).astype(int)
#                     my_y[ch] = (my_y[ch] + xy_pad + 1).round(0).astype(int) 
#             for i in my_ns:  # note my_ns is alreayd an index
#                 for ch in np.arange(n_channels):
#                     # create all 3D crops in crop array using corrected x and y values:
#                     my_crops_all[my_fov_ind,i,my_f,:,:,:,ch] = video[my_fov_ind, my_f,:,
#                             my_y[ch,i].astype(int)-xy_pad:my_y[ch,i].astype(int)+xy_pad+1,
#                             my_x[ch,i].astype(int)-xy_pad:my_x[ch,i].astype(int)+xy_pad+1,ch]
#                     # create xc array
#                     my_xc_all[my_fov_ind, i, my_f, ch] = my_x[ch,i]
#                     my_yc_all[my_fov_ind, i, my_f, ch] = my_y[ch,i]  
#         my_fov_ind = my_fov_ind + 1

#     # Create X-arrays from the data arrays to go into the X-array dataset: 
#     #Create coordinates
#     n = xr.DataArray(np.arange(n_spots_max).astype(np.int16), attrs={'long_name':'crop count'})
#     t =  xr.DataArray(np.arange(n_frames)*my_dt, attrs={'units':units[1],'long_name':'time'})
#     z = xr.DataArray(np.arange(z_slices)*my_dz, attrs={'units':units[0],'long_name':'axial z-distance'})
#     y = xr.DataArray(np.arange(-xy_pad,xy_pad+1)*my_dy, attrs={'units':units[0],'long_name':'radial y-distance'})
#     x = xr.DataArray(np.arange(-xy_pad,xy_pad+1)*my_dx, attrs={'units':units[0],'long_name':'radial x-distance'})
#     ch = np.arange(n_channels)
#     fov = np.arange(n_fov)

#     #Create x-array variables/layers
#     dx = xr.DataArray(my_dx, coords=[], dims=[], attrs={'units':units[0],'long_name':'x-resolution'}) 
#     dy = xr.DataArray(my_dy, coords=[], dims=[], attrs={'units':units[0],'long_name':'y-resolution'}) 
#     dz = xr.DataArray(my_dz, coords=[], dims=[], attrs={'units':units[0],'long_name':'z-resolution'}) 
#     dt = xr.DataArray(my_dt, coords=[], dims=[], attrs={'units':units[1], 'long_name':'temporal resolution'}) 
#     xy_pad = xr.DataArray(xy_pad, coords=[], dims=[], attrs={'units':'pixels'}) 
#     intensity = xr.DataArray(my_crops_all.astype(int), coords=[fov, n, t, z, y, x, ch], dims=['fov', 'n', 't', 'z', 'y', 'x', 'ch'], attrs = {'units':'intensity (a.u.)','long_name':'intensity'})
#     xc = xr.DataArray(my_xc_all.astype(int), coords= [fov, n, t, ch], dims=['fov', 'n', 't', 'ch'], attrs = {'units':'pixels','long_name':'crop center x'})
#     yc = xr.DataArray(my_yc_all.astype(int), coords= [fov, n, t, ch], dims=['fov', 'n', 't', 'ch'], attrs = {'units':'pixels','long_name':'crop center y'})
#     optional_layers = [xr.DataArray(my_layers[col], coords = [fov, n, t], dims=['fov', 'n', 't']) for col in np.arange(len(my_columns))] 
    
#     #Set up dictionary of x-arrays for making a dataset
#     dict1 = dict(zip(my_columns, optional_layers))    
#     dict2 = {
#     'int': intensity,
#     'xc': xc,
#     'yc': yc,
#     'dx': dx,
#     'dy': dy,
#     'dz': dz,
#     'dt': dt,
#     'xy_pad': xy_pad
#     }
#     dict2.update(dict1)

#     # Create the X-array dataset
#     ds = xr.Dataset(
#     dict2, 
#     attrs = {'name': name, 'date': date}
#     )
#     return ds




def create_crop_array(
    video,
    df,
    *,
    axes: Optional[Sequence[str]] = None,
    as_object: bool = True,
    **kwargs,
):
    """
    Build a crop-array from raw inputs.

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
    """
    if axes is not None:
        video = standardize_video_axes(np.asarray(video), axes)

    ds = _create_crop_array_dataset(video, df, **kwargs)

    if as_object:
        # Local import avoids circular dependency
        from .crop_array_object import CropArray
        return CropArray(ds)

    return ds

# def create_crop_array(video, df, as_object: bool = True, **kwargs):
#     """
#     Build a crop-array from raw inputs.

#     Parameters
#     ----------
#     video
#         Input video array.
#     df
#         Spot / crop definition table.
#     as_object : bool, default True
#         If True, return a CropArray wrapper (method-style API).
#         If False, return the raw xarray.Dataset (legacy behavior).
#     **kwargs
#         Passed through to the dataset construction logic.

#     Returns
#     -------
#     CropArray or xarray.Dataset
#     """
#     ds = _create_crop_array_dataset(video, df, **kwargs)

#     if as_object:
#         # Local import avoids circular dependency
#         from .crop_array_object import CropArray
#         return CropArray(ds)

#     return ds

# ============================================================
# Append detailed dataset-builder docs to the public wrapper
# ============================================================
if _create_crop_array_dataset.__doc__:
    create_crop_array.__doc__ = (
        (create_crop_array.__doc__ or "")
        + "\n\n---\n\n"
        + _create_crop_array_dataset.__doc__
    )