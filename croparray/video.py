from __future__ import annotations

import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from pathlib import Path
from itertools import permutations
from PIL import Image, ImageDraw, ImageFont


def _infer_axis_order(shape, axis_letters, expected_sizes, default_order):
    """
    Infer axis order for an array from its shape and expected axis sizes.

    Parameters
    ----------
    shape : tuple[int]
        Shape of the array.
    axis_letters : str
        Axes expected to be present, e.g. "TZCYX", "TZYX", "ZCYX", "ZYX".
    expected_sizes : dict
        Mapping like {"T": image_n, "Z": stack_n, "C": ch_n, "Y": y_dim, "X": x_dim}
    default_order : str
        Fallback order if multiple fits remain.

    Returns
    -------
    inferred_order : str
    """
    if len(shape) != len(axis_letters):
        raise ValueError(
            f"Shape {shape} has {len(shape)} dims, but axis_letters='{axis_letters}' has "
            f"{len(axis_letters)} dims."
        )

    possible_orders = []

    for perm in permutations(axis_letters):
        ok = True
        for axis_index, axis_name in enumerate(perm):
            expected_size = expected_sizes.get(axis_name, None)
            if expected_size is not None and shape[axis_index] != expected_size:
                ok = False
                break
        if ok:
            possible_orders.append("".join(perm))

    if len(possible_orders) == 1:
        return possible_orders[0]

    if len(possible_orders) == 0:
        raise ValueError(
            f"Could not infer axis order from shape {shape} using expected sizes "
            f"{expected_sizes} and allowed axes '{axis_letters}'."
        )

    if default_order in possible_orders:
        print(
            f"Axis order ambiguous for shape {shape}. "
            f"Possible orders: {possible_orders}. "
            f"Using default_order='{default_order}'."
        )
        return default_order

    print(
        f"Axis order ambiguous for shape {shape}. "
        f"Possible orders: {possible_orders}. "
        f"default_order='{default_order}' not among them, using '{possible_orders[0]}'."
    )
    return possible_orders[0]

def _choose_axes_for_ndim(ndim, image_n=None, ch_n=None):
    """
    Decide which axes are present based on ndim and provided metadata.

    Supported cases
    ---------------
    3D: ZYX
    4D: TZYX or ZCYX
    5D: TZCYX
    """
    if ndim == 3:
        return "ZYX", "ZYX"

    elif ndim == 4:
        # If user clearly indicates time but not channels
        if image_n is not None and ch_n is None:
            return "TZYX", "TZYX"

        # If user clearly indicates channels but not time
        if ch_n is not None and image_n is None:
            return "ZCYX", "ZCYX"

        # If user clearly says single channel, 4D likely means TZYX
        if ch_n == 1 and image_n is not None:
            return "TZYX", "TZYX"

        # If user clearly says single timepoint, 4D likely means ZCYX
        if image_n == 1 and ch_n is not None and ch_n > 1:
            return "ZCYX", "ZCYX"

        # Ambiguous 4D: by default assume time is present rather than channel
        print("4D input is ambiguous; assuming axes are some permutation of TZYX.")
        return "TZYX", "TZYX"

    elif ndim == 5:
        return "TZCYX", "TZCYX"

    else:
        raise ValueError(
            f"Unsupported ndim={ndim}. Expected 3D, 4D, or 5D input."
        )

def _to_canonical_TZCYX(data, inferred_order):
    """
    Convert array with inferred_order into canonical order (T, Z, C, Y, X).
    Missing T and/or C are inserted as singleton dimensions.
    """
    current_axes = list(inferred_order)
    arr = data

    # Insert missing T at front
    if "T" not in current_axes:
        arr = np.expand_dims(arr, axis=0)
        current_axes = ["T"] + current_axes

    # Insert missing C after Z
    if "C" not in current_axes:
        z_pos = current_axes.index("Z")
        arr = np.expand_dims(arr, axis=z_pos + 1)
        current_axes = current_axes[:z_pos + 1] + ["C"] + current_axes[z_pos + 1:]

    src_axes = {ax: i for i, ax in enumerate(current_axes)}
    arr = np.transpose(
        arr,
        [src_axes["T"], src_axes["Z"], src_axes["C"], src_axes["Y"], src_axes["X"]],
    )
    return arr

def _get_default_channel_labels(C):
    return [f"ch{c}" for c in range(C)]

def _ensure_list_of_movies(tiff_movies):
    if isinstance(tiff_movies, (str, Path, np.ndarray)):
        return [tiff_movies]
    return list(tiff_movies)

# Maps gui.define_video_axes labels (multi-char) to single-char TZCYX axes
_GUI_LABEL_TO_TZCYX = {"f": "t", "fov": "t", "z": "z", "ch": "c", "y": "y", "x": "x"}


def _gui_labels_to_axes_str(labels) -> str:
    """Convert define_video_axes result labels to a single-char axes string.

    e.g. ("f", "ch", "y", "x") -> "tcyx"
         ("f", "z", "ch", "y", "x") -> "tzcyx"
    """
    return "".join(_GUI_LABEL_TO_TZCYX.get(a.lower(), a.lower()) for a in labels)


def _load_as_TZCYX_from_axes(tiff_movie, axes: str):
    """Load TIFF or array and reorder to canonical (T, Z, C, Y, X) using an explicit axes string."""
    if isinstance(tiff_movie, (str, Path)):
        data = tiff.imread(str(tiff_movie))
    else:
        data = np.asarray(tiff_movie)

    axes_upper = axes.upper().replace(" ", "")
    if len(axes_upper) != data.ndim:
        raise ValueError(
            f"axes string '{axes}' has {len(axes_upper)} dims but data has {data.ndim} dims (shape={data.shape})"
        )

    canonical = list("TZCYX")
    # Reorder present axes into canonical order
    present = [a for a in canonical if a in axes_upper]
    order = [axes_upper.index(a) for a in present]
    data = np.transpose(data, order)

    # Insert singleton dims for any missing canonical axes
    result = data
    for i, a in enumerate(canonical):
        if a not in axes_upper:
            result = np.expand_dims(result, axis=i)

    return result  # always (T, Z, C, Y, X)


def _load_as_TZCYX(
    tiff_movie,
    stack_n=None,
    image_n=None,
    ch_n=None,
    x_dim=None,
    y_dim=None,
):
    """
    Load TIFF or array and convert to canonical (T, Z, C, Y, X).
    Requires helper functions already present in video.py:
      - _choose_axes_for_ndim
      - _infer_axis_order
      - _to_canonical_TZCYX
    """
    if isinstance(tiff_movie, (str, Path)):
        data = tiff.imread(str(tiff_movie))
    else:
        data = np.asarray(tiff_movie)

    # Promote 2D (Y,X) to 3D (1,Y,X) so axis inference can proceed
    if data.ndim == 2:
        data = data[np.newaxis]  # treat as single-channel: (C, Y, X)
    if data.ndim not in (3, 4, 5):
        raise ValueError(f"Expected 2D–5D input, got shape {data.shape}")

    axis_letters, default_order = _choose_axes_for_ndim(
        data.ndim,
        image_n=image_n,
        ch_n=ch_n,
    )

    expected_sizes = {
        "T": image_n,
        "Z": stack_n,
        "C": ch_n,
        "Y": y_dim,
        "X": x_dim,
    }

    inferred_order = _infer_axis_order(
        shape=data.shape,
        axis_letters=axis_letters,
        expected_sizes=expected_sizes,
        default_order=default_order,
    )

    return _to_canonical_TZCYX(data, inferred_order)

def _center_crop_2d(img, crop=None):
    """
    Crop a 2D image.

    crop=int               -> centered square crop of that side length
    crop=(h, w)            -> centered rectangular crop
    crop=((cy,cx), w)      -> square of side w centered on (cy, cx)
    crop=((y0,x0),(y1,x1)) -> arbitrary crop from (y0,x0) to (y1,x1) exclusive
    """
    if crop is None:
        return img

    H, W = img.shape

    if isinstance(crop, (list, tuple)) and isinstance(crop[0], (list, tuple)):
        pt, second = crop
        if isinstance(second, (int, float)):
            # ((cy, cx), w) -> square of side w centered on (cy, cx)
            cy, cx = pt
            half = int(second) // 2
            y0 = max(0, int(cy) - half); x0 = max(0, int(cx) - half)
            y1 = min(H, int(cy) + half); x1 = min(W, int(cx) + half)
        else:
            # ((y0, x0), (y1, x1)) -> arbitrary crop
            (y0, x0), (y1, x1) = pt, second
            y0 = max(0, int(y0)); x0 = max(0, int(x0))
            y1 = min(H, int(y1)); x1 = min(W, int(x1))
        return img[y0:y1, x0:x1]

    if isinstance(crop, int):
        crop_y = crop_x = crop
    else:
        crop_y, crop_x = crop

    crop_y = min(int(crop_y), H)
    crop_x = min(int(crop_x), W)

    y0 = (H - crop_y) // 2
    x0 = (W - crop_x) // 2

    return img[y0:y0 + crop_y, x0:x0 + crop_x]

def _normalize_image(
    img,
    int_range=(0.02, 0.98),
):
    """
    Normalize 2D image to [0,1].

    Rules
    -----
    - If both values in int_range are <= 1, interpret as quantiles.
      Example: (0.02, 0.98)
    - Otherwise interpret as explicit intensity values.
      Example: (100, 2500)

    Parameters
    ----------
    img : np.ndarray
        2D image
    int_range : tuple[float, float]
        Either quantile range or explicit intensity range.

    Returns
    -------
    img_norm : np.ndarray
        Normalized image in [0,1]
    vmin : float
    vmax : float
    mode : str
        'quantile' or 'value'
    """
    img = np.asarray(img, dtype=np.float32)

    if not isinstance(int_range, (tuple, list)) or len(int_range) != 2:
        raise ValueError(f"int_range must be a pair, got {int_range}")

    low, high = float(int_range[0]), float(int_range[1])

    if low >= high:
        raise ValueError(f"int_range must satisfy low < high, got {int_range}")

    if low <= 1 and high <= 1:
        mode = "quantile"
        if not (0 <= low <= 1 and 0 <= high <= 1):
            raise ValueError(f"Quantile int_range must be between 0 and 1, got {int_range}")
        vmin = np.quantile(img, low)
        vmax = np.quantile(img, high)
    else:
        mode = "value"
        vmin = low
        vmax = high

    if vmax > vmin:
        img = np.clip(img, vmin, vmax)
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img)

    return img, float(vmin), float(vmax), mode

def _make_rgb_merge(channel_images, palette):
    """
    channel_images: list of 2D arrays normalized to [0,1]
    palette: sequence of matplotlib-compatible color names
    """
    H, W = channel_images[0].shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    for img, color in zip(channel_images, palette):
        color_rgb = np.array(to_rgb(color), dtype=np.float32)
        rgb += img[..., None] * color_rgb[None, None, :]

    return np.clip(rgb, 0, 1)

def _get_default_channel_labels(C):
    return [f"ch{c}" for c in range(C)]

def _ensure_list_of_movies(tiff_movies):
    if isinstance(tiff_movies, (str, Path, np.ndarray)):
        return [tiff_movies]
    return list(tiff_movies)

def _slice_or_project(arr, sel, axis, project_mode="max", axis_name="axis"):
    """
    arr: ndarray
    sel: int or tuple(start, stop) where stop is inclusive
    axis: axis index to operate on
    project_mode: 'max' or 'mean' when sel is a tuple
    """
    if isinstance(sel, int):
        return np.take(arr, indices=sel, axis=axis)

    if (
        isinstance(sel, (tuple, list))
        and len(sel) == 2
    ):
        start, stop = sel
        start = int(start)
        stop = int(stop)

        if stop < start:
            raise ValueError(f"{axis_name} range must satisfy start <= stop")

        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(start, stop + 1)  # inclusive range
        sub = arr[tuple(slicer)]

        if project_mode == "max":
            return np.max(sub, axis=axis)
        elif project_mode == "mean":
            return np.mean(sub, axis=axis)
        else:
            raise ValueError(f"{axis_name}_project must be 'max' or 'mean'")

    raise ValueError(
        f"{axis_name} must be an int or a tuple/list of length 2, got {sel}"
    )

def _extract_plane_or_projection(
    data_tzcyx,
    t=0,
    z=0,
    t_project="max",
    z_project="max",
):
    """
    From canonical (T, Z, C, Y, X), reduce T and Z to return (C, Y, X).

    t:
      - int -> single timepoint
      - tuple(t0, t1) -> projection across inclusive range

    z:
      - int -> single z-plane
      - tuple(z0, z1) -> projection across inclusive range
    """
    T, Z, C, Y, X = data_tzcyx.shape

    # validate scalar indices early if desired
    if isinstance(t, int) and not (0 <= t < T):
        raise ValueError(f"Requested t={t}, but data has T={T}")
    if isinstance(z, int) and not (0 <= z < Z):
        raise ValueError(f"Requested z={z}, but data has Z={Z}")

    # first reduce T: (T,Z,C,Y,X) -> (Z,C,Y,X)
    reduced_t = _slice_or_project(
        data_tzcyx,
        sel=t,
        axis=0,
        project_mode=t_project,
        axis_name="t",
    )

    # now reduce Z: (Z,C,Y,X) -> (C,Y,X)
    reduced_tz = _slice_or_project(
        reduced_t,
        sel=z,
        axis=0,
        project_mode=z_project,
        axis_name="z",
    )

    return reduced_tz

def _resolve_channel_int_ranges(int_range, C):
    """
    Resolve int_range into one pair per channel.

    Supported inputs
    ----------------
    - single pair applied to all channels:
        (0.02, 0.98)
        (100, 2000)

    - one pair per channel:
        ((0.02, 0.98), (0.1, 0.9))
        ((0, 1200), (1000, 5500))

    Returns
    -------
    ranges : list[tuple[float, float]]
        One pair per channel
    """
    if not isinstance(int_range, (tuple, list)):
        raise ValueError("int_range must be a pair or a list/tuple of pairs")

    # Case 1: single pair
    if len(int_range) == 2 and all(np.isscalar(v) for v in int_range):
        return [tuple(int_range)] * C

    # Case 2: per-channel list of pairs
    if len(int_range) != C:
        raise ValueError(
            f"Per-channel int_range must have length {C}, got {len(int_range)}"
        )

    out = []
    for i, pair in enumerate(int_range):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                f"int_range entry {i} must be a pair, got {pair}"
            )
        out.append((pair[0], pair[1]))

    return out

def _format_units_for_display(units):
    """
    Convert simple unit strings to display-ready text.
    """
    if units == "um":
        return "µm"
    return units

def _add_scalebar_to_axis(
    ax,
    image_shape,
    scale_bar,
    pixel_size,
    units="um",
    position="ll",
    bg="white",
    linewidth=3,
    pad_frac=0.06,
    text_pad_frac=0.03,
    fontsize=10,
):
    """
    Draw a scale bar on a matplotlib axis displaying an image.

    Parameters
    ----------
    ax : matplotlib axis
    image_shape : tuple
        (Y, X) of displayed image
    scale_bar : float
        Length of scale bar in display units, e.g. 5 for 5 µm
    pixel_size : float
        Size of one pixel in the same units, e.g. 0.130 µm/pixel
    units : str
        Display units, e.g. 'um', 'mm', 'm'
    position : {'ll','lr','ul','ur'}
        lower-left, lower-right, upper-left, upper-right
    bg : {'white','black'}
        Used to choose bar/text color
    linewidth : float
        Line thickness
    pad_frac : float
        Padding from image edge as fraction of image size
    text_pad_frac : float
        Vertical gap between bar and label as fraction of image height
    fontsize : float
        Font size for scale bar label
    """
    if scale_bar is None:
        return

    if pixel_size is None or pixel_size <= 0:
        raise ValueError("pixel_size must be provided and > 0 when scale_bar is used")

    H, W = image_shape
    bar_px = scale_bar / pixel_size

    if bar_px <= 0:
        raise ValueError("Computed scale bar length must be > 0")

    if bar_px > W * 0.8:
        raise ValueError(
            f"Scale bar is too large for image width: {bar_px:.1f} px for image width {W}"
        )

    color = "white" if bg == "black" else "black"

    xpad = pad_frac * W
    ypad = pad_frac * H
    text_pad = text_pad_frac * H

    if position not in ("ll", "lr", "ul", "ur"):
        raise ValueError("scalebar_position must be one of 'll', 'lr', 'ul', 'ur'")

    if "l" in position:
        x0 = xpad
        x1 = xpad + bar_px
    else:
        x1 = W - xpad
        x0 = x1 - bar_px

    if "l" == position[0]:  # lower
        y_bar = H - ypad
        y_text = y_bar - text_pad
        va = "bottom"
    else:  # upper
        y_bar = ypad
        y_text = y_bar + text_pad
        va = "top"

    ax.plot([x0, x1], [y_bar, y_bar], color=color, linewidth=linewidth, solid_capstyle="butt")

    units_disp = _format_units_for_display(units)
    label = f"{scale_bar:g} {units_disp}"
    ax.text(
        (x0 + x1) / 2,
        y_text,
        label,
        color=color,
        fontsize=fontsize,
        ha="center",
        va=va,
    )

def _panel_matches_scalebar_location(
    m_idx,
    p_idx,
    n_movies,
    n_panels_per_movie,
    arrange,
    scalebar_panel,
):
    """
    Decide whether the current panel is the requested montage-corner panel.

    Parameters
    ----------
    m_idx : int
        Movie index.
    p_idx : int
        Panel index within a movie row/column.
    n_movies : int
        Number of movies.
    n_panels_per_movie : int
        Number of panels per movie (C grayscale + merge).
    arrange : {"row", "column"}
        Montage arrangement.
    scalebar_panel : {"ul", "ur", "ll", "lr"}

    Returns
    -------
    bool
    """
    if scalebar_panel not in ("ul", "ur", "ll", "lr"):
        raise ValueError("scalebar_panel must be one of 'ul', 'ur', 'll', 'lr'")

    if arrange == "row":
        # rows = movies, cols = panels
        row = m_idx
        col = p_idx
        nrows = n_movies
        ncols = n_panels_per_movie
    elif arrange == "column":
        # rows = panels, cols = movies
        row = p_idx
        col = m_idx
        nrows = n_panels_per_movie
        ncols = n_movies
    else:
        raise ValueError("arrange must be 'row' or 'column'")

    if scalebar_panel == "ul":
        return row == 0 and col == 0
    elif scalebar_panel == "ur":
        return row == 0 and col == ncols - 1
    elif scalebar_panel == "ll":
        return row == nrows - 1 and col == 0
    elif scalebar_panel == "lr":
        return row == nrows - 1 and col == ncols - 1

def plot_multichannel_plane_montage(
    tiff_movies,
    t=0,
    z=0,
    t_project="max",
    z_project="max",
    stack_n=None,
    image_n=None,
    ch_n=None,
    x_dim=None,
    y_dim=None,
    axes: str | None = None,
    define_axes: bool = False,
    palette=None,
    arrange="row",
    crop=None,
    labels=None,
    label_fontsize=11,
    movie_labels=None,
    int_range=(0.02, 0.98),
    grayscale_cmap="gray",
    figsize_scale=3.0,
    fig_height=None,
    panel_pad=0.04,
    title=None,
    show=True,
    return_fig=False,
    bg="white",
    scale_bar=None,
    pixel_size=None,
    units="um",
    scalebar_panel="ll",
    scalebar_position="ll",
    scalebar_linewidth=3,
    scalebar_fontsize=10,
    merge_only=False,
    merge_label="Merge",
):
    """
    Plot a single-plane or projected multichannel montage for one or more TIFF movies.

    For each movie, this function displays:
      - one grayscale panel per channel
      - one merged RGB panel using the specified channel colors

    Input movies are loaded and converted to canonical (T, Z, C, Y, X) ordering
    using the same axis-inference logic as the other video utilities.

    Parameters
    ----------
    tiff_movies : str | Path | np.ndarray | list
        One movie or a list of movies. Each movie may be a file path or an
        already-loaded ndarray.

    t : int | tuple[int, int]
        Time index, or inclusive time range for projection.
        - int: use a single timepoint
        - (t0, t1): project over timepoints t0 through t1 inclusive

    z : int | tuple[int, int]
        Z index, or inclusive z range for projection.
        - int: use a single z-plane
        - (z0, z1): project over z-planes z0 through z1 inclusive

    t_project : {"max", "mean"}
        Projection mode to use if `t` is a range.

    z_project : {"max", "mean"}
        Projection mode to use if `z` is a range.

    stack_n, image_n, ch_n, x_dim, y_dim : int | None
        Expected sizes for Z, T, C, X, and Y used for axis inference.
        Ignored if `axes` is provided.

    axes : str | None
        Explicit axis order string, e.g. ``"TZCYX"`` or ``"TCYX"``.
        If provided, skips automatic inference entirely.

    define_axes : bool, default False
        If True and `axes` is None, opens the axis-labeling GUI on the first
        movie (same dialog used by make_croparrays) and uses the result for
        all movies in the batch.

    palette : sequence[str] | None
        One matplotlib-compatible color per channel, e.g. ("green", "magenta").
        These colors are used in the merged RGB panel.
        If None, default colors are assigned.

    arrange : {"row", "column"}
        Layout of the montage.
        - "row": each movie occupies one row, with panels arranged left-to-right
        - "column": each movie occupies one column, with panels arranged top-to-bottom

    crop : int | tuple | None
        Crop to apply before plotting.
        - int: centered square crop of that size
        - (h, w): centered rectangular crop
        - ((cy, cx), w): square of side w centered at (cy, cx)
        - ((y0, x0), (y1, x1)): arbitrary crop from top-left to bottom-right
        - None: no cropping

    labels : sequence[str] | None
        Labels for channel panels, e.g. ("translation", "ZNF598").
        If provided, must have one label per channel.
        If None, channel labels are omitted, except that the merged panel may
        still be labeled.

    label_fontsize: int
        Optional size of label font.

    movie_labels : sequence[str] | None
        Optional labels for each movie, such as condition names.
        If provided, must match the number of movies.

    int_range : tuple[float, float] | sequence[tuple[float, float]]
        Intensity scaling specification.

        Supported forms:
        - single pair applied to all channels:
            (0.02, 0.98)
            (100, 2500)

        - one pair per channel:
            ((0.02, 0.98), (0.1, 0.9))
            ((0, 1200), (1000, 5500))

        Interpretation rule:
        - if both values in a pair are <= 1, they are interpreted as quantiles
        - otherwise they are interpreted as explicit intensity values

        Thus:
        - (0.02, 0.98) means clip to the 2nd and 98th percentiles
        - (0, 1200) means display intensities from 0 to 1200

    grayscale_cmap : str
        Matplotlib colormap used for grayscale channel panels.

    figsize_scale : float
        Base scale factor for figure size, used when `fig_height` is None.

    fig_height : float | None
        Total figure height in inches.
        If provided, overrides the default height derived from `figsize_scale`.
        Figure width is then computed automatically from the montage shape.

    panel_pad : float
        Padding between panels in the matplotlib subplot layout.

    bg : {"white", "black"}
        Figure background style.
        - "white": white figure background with dark text
        - "black": black figure background with white text

    scale_bar : float | None
        Length of the scale bar in the specified `units`.
        For example, `scale_bar=5` with `units="um"` draws a 5 µm scale bar.
        If None, no scale bar is drawn.

    pixel_size : float | None
        Pixel size in the same units as `scale_bar`.
        For example, if `units="um"`, then `pixel_size` should be in µm/pixel.

    units : str
        Units for the scale bar label, e.g. "um", "mm", or "m".
        If "um" is given, it is displayed as "µm".

    scalebar_panel : {"ul", "ur", "ll", "lr"}
        Which panel in the overall montage receives the scale bar:
        - "ul": upper-left panel of the full montage
        - "ur": upper-right panel
        - "ll": lower-left panel
        - "lr": lower-right panel

    scalebar_position : {"ul", "ur", "ll", "lr"}
        Where inside the chosen panel the scale bar is drawn:
        - "ul": upper left inside the selected panel
        - "ur": upper right
        - "ll": lower left
        - "lr": lower right

    scalebar_linewidth : float
        Line width of the scale bar.

    scalebar_fontsize : float
        Font size of the scale bar label.

    title : str | None
        Optional figure title.

    show : bool
        If True, display the figure with `plt.show()`.

    return_fig : bool
        If True, return `(fig, axes, panel_data)`.

    Returns
    -------
    fig, axes, panel_data : tuple
        Returned only if `return_fig=True`.

        - fig : matplotlib.figure.Figure
            The created figure.
        - axes : np.ndarray
            Array of subplot axes.
        - panel_data : list
            Per-movie panel content used for plotting.
    """
    movies = _ensure_list_of_movies(tiff_movies)
    n_movies = len(movies)

    axes_use = axes
    if axes_use is None and define_axes:
        from . import gui as _gui
        first_path = movies[0] if isinstance(movies[0], (str, Path)) else None
        axes_use = _gui_labels_to_axes_str(_gui.define_video_axes(path=first_path).axes)

    loaded = []
    for movie in movies:
        if axes_use is not None:
            data_tzcyx = _load_as_TZCYX_from_axes(movie, axes_use)
        else:
            data_tzcyx = _load_as_TZCYX(
                movie,
                stack_n=stack_n,
                image_n=image_n,
                ch_n=ch_n,
                x_dim=x_dim,
                y_dim=y_dim,
            )
        loaded.append(data_tzcyx)

    # require same channel count across movies
    channel_counts = [arr.shape[2] for arr in loaded]
    if len(set(channel_counts)) != 1:
        raise ValueError(f"All movies must have same number of channels, got {channel_counts}")
    C = channel_counts[0]

    if palette is None:
        default_palette = [
            "green", "magenta", "cyan", "red", "yellow", "blue", "orange", "white"
        ]
        if C <= len(default_palette):
            palette = default_palette[:C]
        else:
            raise ValueError(
                f"No default palette for C={C}. Please pass palette with {C} colors."
            )

    if len(palette) != C:
        raise ValueError(f"palette must have length {C}, got {len(palette)}")

    # None entries in palette hide that channel's panel and exclude it from the merge
    visible_chs = [i for i, p in enumerate(palette) if p is not None]
    if not visible_chs:
        raise ValueError("palette must have at least one non-None color.")
    visible_palette = [palette[i] for i in visible_chs]

    if labels is None:
        channel_labels = _get_default_channel_labels(C)
        add_channel_labels = False
    else:
        if len(labels) != C:
            raise ValueError(f"labels must have length {C}, got {len(labels)}")
        channel_labels = [l if l is not None else "" for l in labels]
        add_channel_labels = any(l is not None for l in labels)

    if movie_labels is not None and len(movie_labels) != n_movies:
        raise ValueError(f"movie_labels must have length {n_movies}, got {len(movie_labels)}")

    n_panels_per_movie = 1 if merge_only else len(visible_chs) + 1

    if arrange not in ("row", "column"):
        raise ValueError("arrange must be 'row' or 'column'")

    if arrange == "row":
        nrows = n_movies
        ncols = n_panels_per_movie
    else:
        nrows = n_panels_per_movie
        ncols = n_movies

    # Compute panel aspect ratio from the actual (cropped) image size so the
    # figure dimensions match and imshow never letterboxes into the background.
    _Y, _X = loaded[0].shape[3], loaded[0].shape[4]
    _dummy = _center_crop_2d(np.zeros((_Y, _X), dtype=np.float32), crop=crop)
    _panel_h_px, _panel_w_px = _dummy.shape
    _panel_aspect = _panel_h_px / max(_panel_w_px, 1)  # height / width

    if fig_height is None:
        fig_w = figsize_scale * ncols
        fig_h = figsize_scale * _panel_aspect * nrows
    else:
        fig_h = float(fig_height)
        _panel_h_in = fig_h / nrows
        fig_w = (_panel_h_in / max(_panel_aspect, 1e-6)) * ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    # Zero outer margins so the axes fill the figure; tight_layout then
    # expands them just enough to avoid clipping any visible titles/labels.
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1,
                        wspace=panel_pad, hspace=panel_pad)
    has_labels = (add_channel_labels or title is not None
                  or movie_labels is not None or merge_label is not None)
    if has_labels:
        try:
            fig.tight_layout(pad=max(panel_pad, 0.02),
                             w_pad=panel_pad, h_pad=panel_pad)
        except Exception:
            pass

    if bg not in ("white", "black"):
        raise ValueError("bg must be 'white' or 'black'")

    if bg == "black":
        fig.patch.set_facecolor("black")
        text_color = "white"
        axes_bg = "black"
    else:
        fig.patch.set_facecolor("white")
        text_color = "black"
        axes_bg = "white"

    panel_data = []

    for m_idx, data_tzcyx in enumerate(loaded):
        reduced = _extract_plane_or_projection(
            data_tzcyx,
            t=t,
            z=z,
            t_project=t_project,
            z_project=z_project,
        )  # (C,Y,X)

        C_this, Y, X = reduced.shape

        gray_imgs_norm = []
        channel_ranges = []

        channel_int_ranges = _resolve_channel_int_ranges(int_range, C_this)

        for c in range(C_this):
            img = reduced[c]
            img = _center_crop_2d(img, crop=crop)

            img_norm, vmin, vmax, mode = _normalize_image(
                img,
                int_range=channel_int_ranges[c],
            )

            gray_imgs_norm.append(img_norm)
            channel_ranges.append(
                {
                    "channel": c,
                    "input_range": tuple(channel_int_ranges[c]),
                    "resolved_vmin": vmin,
                    "resolved_vmax": vmax,
                    "mode": mode,
                }
            )

        visible_imgs = [gray_imgs_norm[c] for c in visible_chs]
        merge_rgb = _make_rgb_merge(visible_imgs, visible_palette)

        if merge_only:
            row_panels = [("Merge", None, merge_rgb)]
        else:
            row_panels = [("gray", c, gray_imgs_norm[c]) for c in visible_chs]
            row_panels.append(("Merge", None, merge_rgb))
        panel_data.append(row_panels)

        for p_idx, (ptype, c_idx, panel_img) in enumerate(row_panels):
            if arrange == "row":
                ax = axes[m_idx, p_idx]
            else:
                ax = axes[p_idx, m_idx]

            ax.set_facecolor(axes_bg)
            
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            if ptype == "Merge":
                ax.imshow(panel_img, aspect='auto')
            else:
                ax.imshow(panel_img, cmap=grayscale_cmap, vmin=0, vmax=1, aspect='auto')

            ax.set_xticks([])
            ax.set_yticks([])

            if ptype == "Merge":
                panel_label = merge_label  # None suppresses the label
            else:
                panel_label = channel_labels[c_idx]

            label_color = palette[c_idx] if ptype != "Merge" else text_color
            show_this_label = panel_label is not None and (
                add_channel_labels or ptype == "Merge"
            )

            if arrange == "row":
                if show_this_label:
                    ax.set_title(panel_label, fontsize=label_fontsize, color=label_color)
                if movie_labels is not None and p_idx == 0:
                    ax.set_ylabel(movie_labels[m_idx], fontsize=label_fontsize, color=text_color)
            else:
                if movie_labels is not None and p_idx == 0:
                    ax.set_title(movie_labels[m_idx], fontsize=label_fontsize, color=text_color)
                if show_this_label:
                    if m_idx == 0:
                        ax.set_ylabel(
                            panel_label,
                            fontsize=label_fontsize,
                            rotation=0,
                            labelpad=35,
                            va="center",
                            color=label_color,
                        )

            # Decide whether to draw the scale bar on this panel
            draw_scalebar_here = False

            if scale_bar is not None:
                draw_scalebar_here = _panel_matches_scalebar_location(
                    m_idx=m_idx,
                    p_idx=p_idx,
                    n_movies=n_movies,
                    n_panels_per_movie=n_panels_per_movie,
                    arrange=arrange,
                    scalebar_panel=scalebar_panel,
                )

            if draw_scalebar_here:
                if ptype == "Merge":
                    img_shape_for_bar = panel_img.shape[:2]
                else:
                    img_shape_for_bar = panel_img.shape

                _add_scalebar_to_axis(
                    ax,
                    image_shape=img_shape_for_bar,
                    scale_bar=scale_bar,
                    pixel_size=pixel_size,
                    units=units,
                    position=scalebar_position,
                    bg=bg,
                    linewidth=scalebar_linewidth,
                    fontsize=scalebar_fontsize,
                )

    if title is not None:
        fig.suptitle(title, fontsize=label_fontsize+2, color=text_color)

    if show:
        plt.show()

    if return_fig:
        return fig, axes, panel_data

def Color_Z_Depth(
    tiff_movie,
    stack_n=None,
    image_n=None,
    ch_n=None,
    x_dim=None,
    y_dim=None,
    ch=0,
    stack_range=(4, 23),
    palette="rainbow",
    int_range=(0.02, 0.95),
    normalize_each_frame=True,
    blend_mode="max",
    return_metadata=True,
):
    """
    Load a TIFF stack/movie, infer dimensional ordering, and color z-depth.

    Supported input dimensionalities
    --------------------------------
    3D: Z,Y,X
    4D: T,Z,Y,X   or   Z,C,Y,X
    5D: T,Z,C,Y,X
    Any permutation of those axes is allowed if sizes let us infer them.

    Parameters
    ----------
    tiff_movie : str | Path | np.ndarray
        TIFF path or already-loaded array.
    stack_n : int | None
        Expected number of z slices.
    image_n : int | None
        Expected number of timepoints.
    ch_n : int | None
        Expected number of channels.
    x_dim, y_dim : int | None
        Expected X and Y image sizes.
    ch : int
        Channel to colorize after canonicalization to (T,Z,C,Y,X).
        If there is only one channel, ch may be 0 or 1.
    stack_range : tuple[int, int]
        z range mapped across the colormap.
    palette : str
        Matplotlib colormap name.
    int_range : tuple[float, float]
        Intensity quantile range used for clipping and normalization.
        Example (0.02, 0.95) means clip to 2nd and 95th percentiles.
    normalize_each_frame : bool
        Normalize selected channel per timepoint if True, otherwise globally.
    blend_mode : str
        'max' or 'sum' for combining z-colored slices.
    return_metadata : bool
        If True, return metadata dictionary too.

    Returns
    -------
    rgb_movie : np.ndarray
        Shape (T, Y, X, 3), dtype uint8
    data_tzcyx : np.ndarray
        Canonicalized raw data in shape (T, Z, C, Y, X)
    metadata : dict
        Returned only if return_metadata=True
    """
    # -----------------------------
    # 1. Load
    # -----------------------------
    if isinstance(tiff_movie, (str, Path)):
        data = tiff.imread(str(tiff_movie))
    else:
        data = np.asarray(tiff_movie)

    if data.ndim not in (3, 4, 5):
        raise ValueError(
            f"Expected 3D, 4D, or 5D input, but got shape {data.shape}"
        )

    # -----------------------------
    # 2. Determine which axes exist
    # -----------------------------
    axis_letters, default_order = _choose_axes_for_ndim(
        data.ndim,
        image_n=image_n,
        ch_n=ch_n,
    )

    expected_sizes = {
        "T": image_n,
        "Z": stack_n,
        "C": ch_n,
        "Y": y_dim,
        "X": x_dim,
    }

    inferred_order = _infer_axis_order(
        shape=data.shape,
        axis_letters=axis_letters,
        expected_sizes=expected_sizes,
        default_order=default_order,
    )

    # -----------------------------
    # 3. Canonicalize to (T,Z,C,Y,X)
    # -----------------------------
    data_tzcyx = _to_canonical_TZCYX(data, inferred_order)
    T, Z, C, Y, X = data_tzcyx.shape

    # -----------------------------
    # 4. Handle channel selection
    # -----------------------------
    if C == 1 and ch in (0, 1):
        ch_use = 0
    else:
        ch_use = ch

    if ch_use < 0 or ch_use >= C:
        raise ValueError(
            f"Requested ch={ch}, interpreted as ch_use={ch_use}, but data has C={C}."
        )

    # -----------------------------
    # 5. Validate intensity range
    # -----------------------------
    q_low, q_high = int_range
    if not (0 <= q_low <= 1 and 0 <= q_high <= 1):
        raise ValueError("int_range values must be between 0 and 1")
    if q_low >= q_high:
        raise ValueError("int_range must satisfy low < high")

    # -----------------------------
    # 6. z -> color mapping
    # -----------------------------
    z_low, z_high = stack_range
    if z_low > z_high:
        raise ValueError("stack_range must be (low, high)")

    z_positions = np.arange(Z)

    if z_high == z_low:
        z_norm = np.where(z_positions <= z_low, 0.0, 1.0)
    else:
        z_norm = (z_positions - z_low) / (z_high - z_low)
        z_norm = np.clip(z_norm, 0, 1)

    cmap = colormaps[palette]
    z_colors = cmap(z_norm)[:, :3]  # (Z, 3)

    # -----------------------------
    # 7. Extract selected channel
    # -----------------------------
    selected = data_tzcyx[:, :, ch_use, :, :].astype(np.float32)  # (T,Z,Y,X)
    rgb_movie_float = np.zeros((T, Y, X, 3), dtype=np.float32)

    if normalize_each_frame:
        for t in range(T):
            frame_stack = selected[t]

            low_val = np.quantile(frame_stack, q_low)
            high_val = np.quantile(frame_stack, q_high)

            if high_val > low_val:
                frame_stack = np.clip(frame_stack, low_val, high_val)
                frame_stack = (frame_stack - low_val) / (high_val - low_val)
            else:
                frame_stack = np.zeros_like(frame_stack)

            colored_stack = frame_stack[..., None] * z_colors[:, None, None, :]  # (Z,Y,X,3)

            if blend_mode == "max":
                rgb = np.max(colored_stack, axis=0)
            elif blend_mode == "sum":
                rgb = np.sum(colored_stack, axis=0)
                if rgb.max() > 0:
                    rgb = rgb / rgb.max()
            else:
                raise ValueError("blend_mode must be 'max' or 'sum'")

            rgb_movie_float[t] = rgb

    else:
        low_val = np.quantile(selected, q_low)
        high_val = np.quantile(selected, q_high)

        if high_val > low_val:
            selected = np.clip(selected, low_val, high_val)
            selected = (selected - low_val) / (high_val - low_val)
        else:
            selected = np.zeros_like(selected)

        for t in range(T):
            frame_stack = selected[t]
            colored_stack = frame_stack[..., None] * z_colors[:, None, None, :]

            if blend_mode == "max":
                rgb = np.max(colored_stack, axis=0)
            elif blend_mode == "sum":
                rgb = np.sum(colored_stack, axis=0)
                if rgb.max() > 0:
                    rgb = rgb / rgb.max()
            else:
                raise ValueError("blend_mode must be 'max' or 'sum'")

            rgb_movie_float[t] = rgb

    rgb_movie = np.clip(rgb_movie_float * 255, 0, 255).astype(np.uint8)

    metadata = {
        "original_shape": tuple(data.shape),
        "input_ndim": data.ndim,
        "axes_present_before_canonicalization": axis_letters,
        "inferred_order_before_canonicalization": inferred_order,
        "canonical_order_after_canonicalization": "TZCYX",
        "canonical_shape": tuple(data_tzcyx.shape),
        "T": T,
        "Z": Z,
        "C": C,
        "Y": Y,
        "X": X,
        "channel_requested": ch,
        "channel_used": ch_use,
        "stack_range": stack_range,
        "palette": palette,
        "int_range": int_range,
        "normalize_each_frame": normalize_each_frame,
        "blend_mode": blend_mode,
    }

    if return_metadata:
        return rgb_movie, data_tzcyx, metadata
    else:
        return rgb_movie, data_tzcyx

def add_z_legend_bar_labeled(
    rgb_movie,
    z_total,
    stack_range=(4, 23),
    palette="turbo",
    bar_width=24,
    pad=12,
    top_pad=20,
    bottom_pad=20,
    background_value=0,
    label_top=None,
    label_bottom=None,
    font_size=18,
    text_color=(255, 255, 255),
    text_pad=8,
):
    """
    Append a vertical z-color legend bar to the right side of an RGB movie,
    with top and bottom z labels burned directly into each frame.

    Parameters
    ----------
    rgb_movie : np.ndarray
        RGB movie of shape (T, Y, X, 3), dtype uint8 preferred.
    z_total : int
        Total number of z slices in the original stack.
    stack_range : tuple[int, int]
        (z_low, z_high) used in Color_Z_Depth.
    palette : str
        Matplotlib colormap name.
    bar_width : int
        Width of the legend bar in pixels.
    pad : int
        Horizontal padding between movie and legend area, and right margin.
    top_pad : int
        Top margin above the legend bar.
    bottom_pad : int
        Bottom margin below the legend bar.
    background_value : int
        Background fill value, usually 0 for black or 255 for white.
    label_top : str | None
        Label to draw near the top of the bar. If None, uses f"z={stack_range[1]}".
    label_bottom : str | None
        Label to draw near the bottom of the bar. If None, uses f"z={stack_range[0]}".
    font_size : int
        Font size for burned-in labels.
    text_color : tuple[int, int, int]
        RGB color for text.
    text_pad : int
        Padding between the legend bar and text.

    Returns
    -------
    rgb_out : np.ndarray
        RGB movie with labeled legend bar, shape (T, Y, X_new, 3), dtype uint8.
    """
    rgb_movie = np.asarray(rgb_movie)
    if rgb_movie.ndim != 4 or rgb_movie.shape[-1] != 3:
        raise ValueError("rgb_movie must have shape (T, Y, X, 3)")

    if rgb_movie.dtype != np.uint8:
        rgb_movie = np.clip(rgb_movie, 0, 255).astype(np.uint8)

    T, Y, X, _ = rgb_movie.shape
    z_low, z_high = stack_range

    if z_total < 1:
        raise ValueError("z_total must be >= 1")
    if z_low > z_high:
        raise ValueError("stack_range must be (low, high)")

    if label_top is None:
        label_top = f"z={z_high}"
    if label_bottom is None:
        label_bottom = f"z={z_low}"

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Measure text boxes
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)

    def text_size(txt):
        bbox = draw.textbbox((0, 0), txt, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    top_text_w, top_text_h = text_size(label_top)
    bot_text_w, bot_text_h = text_size(label_bottom)
    text_block_w = max(top_text_w, bot_text_w)

    # Height available for the bar
    bar_height = Y - top_pad - bottom_pad
    if bar_height <= 1:
        raise ValueError("top_pad + bottom_pad too large for image height")

    # Colormap
    cmap = colormaps[palette]
    z_positions = np.arange(z_total)

    if z_high == z_low:
        z_norm = np.where(z_positions <= z_low, 0.0, 1.0)
    else:
        z_norm = (z_positions - z_low) / (z_high - z_low)
        z_norm = np.clip(z_norm, 0, 1)

    z_colors = (cmap(z_norm)[:, :3] * 255).astype(np.uint8)  # (Z,3)

    # Build vertical legend image
    legend = np.full((Y, bar_width, 3), background_value, dtype=np.uint8)

    y0 = top_pad
    y1 = Y - bottom_pad
    y_coords = np.arange(y0, y1)

    # top = highest z, bottom = lowest z
    if len(y_coords) > 1:
        frac = 1.0 - (y_coords - y0) / (len(y_coords) - 1)
    else:
        frac = np.array([0.5])

    z_float = frac * (z_total - 1)
    z_idx = np.clip(np.round(z_float).astype(int), 0, z_total - 1)
    legend[y0:y1, :, :] = z_colors[z_idx][:, None, :]

    # Total added width: gap + bar + gap-to-text + text + right margin
    extra_width = pad + bar_width + text_pad + text_block_w + pad
    out = np.full((T, Y, X + extra_width, 3), background_value, dtype=np.uint8)
    out[:, :, :X, :] = rgb_movie
    out[:, :, X + pad:X + pad + bar_width, :] = legend[None, :, :, :]

    # Burn text into every frame
    text_x = X + pad + bar_width + text_pad
    top_text_y = max(0, top_pad - top_text_h // 2)
    bottom_text_y = min(Y - bot_text_h, Y - bottom_pad - bot_text_h // 2)

    for t in range(T):
        img = Image.fromarray(out[t])
        d = ImageDraw.Draw(img)
        d.text((text_x, top_text_y), label_top, font=font, fill=text_color)
        d.text((text_x, bottom_text_y), label_bottom, font=font, fill=text_color)
        out[t] = np.array(img, dtype=np.uint8)

    return out

def save_rgb_tiff(rgb_movie, path, compression="zlib"):
    tiff.imwrite(
        path,
        rgb_movie,
        photometric="rgb",
        compression=compression,
        imagej=True,
        metadata={"axes": "TYXS"},
    )

def show_rgb_frame(rgb_movie, t=0, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.imshow(rgb_movie[t])
    plt.axis("off")
    plt.title(f"RGB z-depth frame t={t}")
    plt.show()

def save_rgb_gif(rgb_movie, out_path, fps=5):
    import imageio.v2 as imageio
    imageio.mimsave(out_path, rgb_movie, duration=1 / fps)


def save_montage_movie(
    tiff_movies,
    out_path,
    *,
    n_t: int | tuple | None = None,
    z=0,
    fps: float = 5,
    dt: float | None = None,
    timestamp_units: str = "s",
    show_timestamp: bool = True,
    timestamp_panel: str = "ll",
    timestamp_position: str = "ll",
    timestamp_fontsize: int | None = None,
    timestamp_bg_alpha: float = 0.6,
    merge_only: bool = False,
    preview_t: int = 0,
    skip_preview: bool = False,
    ffmpeg_preset: str = "medium",
    **montage_kwargs,
):
    """
    Save a movie of plot_multichannel_plane_montage frames across all timepoints.

    Shows a preview of one frame first and asks for confirmation before rendering
    the full movie (which can be slow). Pass skip_preview=True to bypass.

    Parameters
    ----------
    tiff_movies
        One movie path/array or a list of them (same as plot_multichannel_plane_montage).
    out_path
        Output path. Extension sets format: .mp4, .avi, .gif, .tif/.tiff.
    n_t
        Number of timepoints to render. If None (default), uses the full movie length.
    z
        Z index or projection range (passed through to montage).
    fps
        Frames per second for video/GIF output.
    dt
        Seconds per frame. If provided, timestamp reads "t = X.X s"; otherwise "t = N".
    timestamp_units
        Units label shown in timestamp when dt is set (default 's').
    merge_only
        If True, only the merged RGB panel is shown (individual channels hidden).
    preview_t
        Timepoint index for the preview frame (default 0).
    skip_preview
        If True, skip the interactive preview + confirmation and render immediately.
    **montage_kwargs
        All other arguments forwarded to plot_multichannel_plane_montage.
        Supported extras: define_axes (bool), axes (str), and all montage params.
        Do not pass t, z, show, or return_fig.
    """
    # ── 1. Resolve axes string once ───────────────────────────────────────────
    if montage_kwargs.pop("define_axes", False):
        first_movie = tiff_movies if not isinstance(tiff_movies, (list, tuple)) else tiff_movies[0]
        from . import gui as _gui
        first_path = first_movie if isinstance(first_movie, (str, Path)) else None
        axes_str = _gui_labels_to_axes_str(_gui.define_video_axes(path=first_path).axes)
    else:
        axes_str = montage_kwargs.pop("axes", None)

    # ── 2. Pre-load all movies once (avoids re-reading TIFF every frame) ──────
    movies_list = tiff_movies if isinstance(tiff_movies, (list, tuple)) else [tiff_movies]
    preloaded = []
    for m in movies_list:
        if axes_str is not None:
            arr = _load_as_TZCYX_from_axes(m, axes_str)
        else:
            arr = _load_as_TZCYX(
                m,
                stack_n=montage_kwargs.get("stack_n"),
                image_n=montage_kwargs.get("image_n"),
                ch_n=montage_kwargs.get("ch_n"),
                x_dim=montage_kwargs.get("x_dim"),
                y_dim=montage_kwargs.get("y_dim"),
            )
        preloaded.append(arr)

    # Arrays are now canonical TZCYX — strip loading-related kwargs but keep axes="TZCYX"
    # so plot_multichannel_plane_montage uses _load_as_TZCYX_from_axes (silent no-op)
    # instead of _infer_axis_order which prints "Axis order ambiguous" once per frame.
    for _k in ("stack_n", "image_n", "ch_n", "x_dim", "y_dim"):
        montage_kwargs.pop(_k, None)
    montage_kwargs["axes"] = "TZCYX"

    tiff_movies_loaded = preloaded if len(preloaded) > 1 else preloaded[0]

    # Resolve n_t into a sequence of frame indices
    T = preloaded[0].shape[0]
    if n_t is None:
        t_indices = range(T)
    elif isinstance(n_t, int):
        t_indices = range(min(n_t, T))
    elif isinstance(n_t, (tuple, list)) and len(n_t) == 3:
        start, stop, step = n_t
        if stop is None:
            stop = T
        t_indices = range(start, min(stop, T), step)
    else:
        raise TypeError(
            "n_t must be None, an int, or a (start, stop, step) tuple "
            "(use stop=None to go to end of movie)"
        )
    n_t = len(t_indices)

    # ── 3. Rendering parameters ───────────────────────────────────────────────
    text_color = "white" if montage_kwargs.get("bg", "white") == "black" else "black"
    label_fontsize = montage_kwargs.get("label_fontsize", 11)
    _ts_fs = timestamp_fontsize if timestamp_fontsize is not None else label_fontsize
    _ts_inner = {
        "ll": (0.03, 0.03, "bottom", "left"),
        "lr": (0.97, 0.03, "bottom", "right"),
        "ul": (0.03, 0.97, "top",    "left"),
        "ur": (0.97, 0.97, "top",    "right"),
    }
    _ts_x, _ts_y, _ts_va, _ts_ha = _ts_inner.get(timestamp_position, (0.03, 0.03, "bottom", "left"))
    _n_movies = len(preloaded)
    _arrange  = montage_kwargs.get("arrange", "row")

    # Build a timestamp formatter that auto-selects SS / MM:SS / HH:MM:SS
    _UNITS_TO_S = {"s": 1.0, "sec": 1.0, "ms": 0.001, "min": 60.0, "h": 3600.0, "hr": 3600.0}
    _unit_factor = _UNITS_TO_S.get(timestamp_units.lower(), 1.0) if dt is not None else 1.0
    _max_t = (max(t_indices) if len(t_indices) > 0 else 0) if dt is not None else 0
    _total_s = _max_t * dt * _unit_factor if dt is not None else 0.0

    def _fmt_ts(t_i):
        if dt is None:
            return f"t = {t_i}"
        t_s = t_i * dt * _unit_factor
        if _total_s < 60:
            return f"{int(round(t_s)):02d}"
        elif _total_s < 3600:
            m, s = divmod(int(round(t_s)), 60)
            return f"{m:02d}:{s:02d}"
        else:
            h, rem = divmod(int(round(t_s)), 3600)
            m, s = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

    # Derive visible channels so we can recompute panel images cheaply per frame
    _raw_palette = montage_kwargs.get("palette", None)
    _C = preloaded[0].shape[2]
    if _raw_palette is None:
        _default_pal = ["green", "magenta", "cyan", "red", "yellow", "blue", "orange", "white"]
        _raw_palette = _default_pal[:_C]
    _visible_chs     = [i for i, p in enumerate(_raw_palette) if p is not None]
    _visible_palette = [_raw_palette[i] for i in _visible_chs]
    _n_panels        = 1 if merge_only else len(_visible_chs) + 1
    _int_range = montage_kwargs.get("int_range", (0.02, 0.98))
    _crop      = montage_kwargs.get("crop", None)
    _t_proj    = montage_kwargs.get("t_project", "max")
    _z_proj    = montage_kwargs.get("z_project", "max")

    # Pre-compute per-channel intensity ranges from the full (uncropped) movie.
    # Quantile-based ranges are resolved once here so they're stable across frames
    # and independent of the crop setting.
    _ch_ranges_raw = _resolve_channel_int_ranges(_int_range, _C)
    _global_int_ranges = []  # list of (vmin, vmax) per channel
    t_list = list(t_indices)
    print("  Computing global intensity ranges...", end="\r")
    for c in range(_C):
        low, high = _ch_ranges_raw[c]
        if low <= 1.0 and high <= 1.0:
            # Collect uncropped pixel values across all frames and all movies
            all_px = []
            for movie_data in preloaded:
                if isinstance(z, int):
                    all_px.append(movie_data[t_list, z, c, :, :].ravel())
                else:
                    z0, z1 = z
                    all_px.append(movie_data[t_list, z0:z1+1, c, :, :].ravel())
            px = np.concatenate(all_px).astype(np.float32)
            _global_int_ranges.append((float(np.quantile(px, low)),
                                       float(np.quantile(px, high))))
        else:
            _global_int_ranges.append((float(low), float(high)))
    print(f"  Global intensity ranges: { {c: _global_int_ranges[c] for c in range(_C)} }")

    def _apply_norm(img, vmin, vmax):
        img_f = np.asarray(img, dtype=np.float32)
        if vmax > vmin:
            return np.clip((img_f - vmin) / (vmax - vmin), 0.0, 1.0)
        return np.zeros_like(img_f)

    def _panels_for_frame(movie_data, t_i):
        reduced = _extract_plane_or_projection(
            movie_data, t=t_i, z=z, t_project=_t_proj, z_project=_z_proj,
        )
        gray_norm = []
        for c in range(reduced.shape[0]):
            img = _center_crop_2d(reduced[c], crop=_crop)
            vmin, vmax = _global_int_ranges[c]
            gray_norm.append(_apply_norm(img, vmin, vmax))
        vis_imgs = [gray_norm[c] for c in _visible_chs]
        merge_rgb = _make_rgb_merge(vis_imgs, _visible_palette)
        if merge_only:
            return [merge_rgb]
        return [gray_norm[c] for c in _visible_chs] + [merge_rgb]

    # ── 4. Build figure once; collect per-panel AxesImage references ──────────
    # Use the pre-computed global ranges for the initial figure too
    montage_kwargs["int_range"] = _global_int_ranges
    montage_kwargs["merge_only"] = merge_only
    fig, axes_arr, _ = plot_multichannel_plane_montage(
        tiff_movies=tiff_movies_loaded,
        t=0, z=z, show=False, return_fig=True,
        **montage_kwargs,
    )

    # (m_idx, p_idx) -> AxesImage for cheap per-frame set_data() updates
    _panel_ims = {}
    for mi in range(_n_movies):
        for pi in range(_n_panels):
            ax = axes_arr[mi, pi] if _arrange == "row" else axes_arr[pi, mi]
            if ax.get_visible() and ax.images:
                _panel_ims[(mi, pi)] = ax.images[0]

    # Add timestamp text once; update its string per frame
    _ts_text = None
    if show_timestamp:
        all_ax_flat = list(axes_arr.flat)
        target_ax = None
        for flat_idx, ax in enumerate(all_ax_flat):
            if not ax.get_visible():
                continue
            if _arrange == "row":
                mi, pi = divmod(flat_idx, _n_panels)
            else:
                pi, mi = divmod(flat_idx, _n_movies)
            if _panel_matches_scalebar_location(
                m_idx=mi, p_idx=pi,
                n_movies=_n_movies, n_panels_per_movie=_n_panels,
                arrange=_arrange, scalebar_panel=timestamp_panel,
            ):
                target_ax = ax
                break
        if target_ax is None:
            target_ax = next((ax for ax in all_ax_flat if ax.get_visible()), None)
        if target_ax is not None:
            ts0 = _fmt_ts(0)
            _ts_bbox = (
                dict(boxstyle="round,pad=0.2",
                     fc="black" if text_color == "white" else "white",
                     alpha=timestamp_bg_alpha, ec="none")
                if timestamp_bg_alpha > 0 else None
            )
            _ts_text = target_ax.text(
                _ts_x, _ts_y, ts0, transform=target_ax.transAxes,
                color=text_color, fontsize=_ts_fs, va=_ts_va, ha=_ts_ha,
                bbox=_ts_bbox,
            )

    def _update_figure(t_i) -> np.ndarray:
        """Update panel images + timestamp for time t_i; return captured RGB frame."""
        for mi, movie_data in enumerate(preloaded):
            for pi, panel_img in enumerate(_panels_for_frame(movie_data, t_i)):
                if (mi, pi) in _panel_ims:
                    _panel_ims[(mi, pi)].set_data(panel_img)
        if _ts_text is not None:
            ts = _fmt_ts(t_i)
            _ts_text.set_text(ts)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    # ── 5. Preview + confirmation ─────────────────────────────────────────────
    if not skip_preview:
        print(f"Rendering preview (t={preview_t})...")
        _update_figure(preview_t)
        # IPython.display.display is more reliable than plt.show() in Jupyter
        # because it flushes the figure to cell output before input() renders.
        try:
            from IPython.display import display as _ipy_display
            _ipy_display(fig)
        except Exception:
            plt.show()
        answer = input(f"Save full movie ({n_t} frames) to {out_path}? [y/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            plt.close(fig)
            print("Cancelled.")
            return

    # ── 6. Render + write (streaming for mp4/avi to avoid RAM accumulation) ───
    out_path = Path(out_path)
    suffix = out_path.suffix.lower()

    if suffix in (".mp4", ".avi"):
        import subprocess, shutil as _shutil, sys, tempfile as _tempfile
        ffmpeg_exe = _shutil.which("ffmpeg")
        if ffmpeg_exe is None:
            _conda_ff = Path(sys.executable).parent.parent / "Library" / "bin" / "ffmpeg.exe"
            if _conda_ff.exists():
                ffmpeg_exe = str(_conda_ff)
        if ffmpeg_exe is None:
            raise RuntimeError(
                "ffmpeg not found. Install with: conda install -c conda-forge ffmpeg"
            )
        # Render first frame to learn pixel dimensions
        t_iter = iter(t_indices)
        t0 = next(t_iter)
        print(f"  frame 1/{n_t} (t={t0})", end="\r")
        frame0 = _update_figure(t0)
        h, w = frame0.shape[:2]
        codec = "ffv1" if suffix == ".avi" else "libx264"
        # Write to a local temp file first — avoids slow NAS writes during encoding
        tmp_fd, tmp_path = _tempfile.mkstemp(suffix=suffix)
        os.close(tmp_fd)
        try:
            cmd = [
                ffmpeg_exe, "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "rgb24",
                "-r", str(fps),
                "-i", "pipe:0",
                "-vcodec", codec,
            ]
            if suffix == ".mp4":
                cmd += ["-crf", "18", "-preset", ffmpeg_preset, "-pix_fmt", "yuv420p"]
            cmd.append(tmp_path)
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            proc.stdin.write(frame0.tobytes())
            for i, t_i in enumerate(t_iter, start=2):
                if i % 10 == 0:
                    print(f"  frame {i}/{n_t} (t={t_i})", end="\r")
                proc.stdin.write(_update_figure(t_i).tobytes())
            proc.stdin.close()
            plt.close(fig)
            print(f"  frame {n_t}/{n_t} — done.")

            # Stream-parse ffmpeg stderr for progress while encoding
            import threading, time as _time
            _enc_frame = [0]
            _stderr_buf = [b""]
            _stderr_lines = []

            def _read_stderr(pipe):
                buf = b""
                while True:
                    ch = pipe.read(1)
                    if not ch:
                        break
                    if ch in (b"\r", b"\n"):
                        line = buf.decode("utf-8", errors="replace")
                        _stderr_lines.append(line)
                        if "frame=" in line:
                            try:
                                _enc_frame[0] = int(line.split("frame=")[1].split()[0])
                            except Exception:
                                pass
                        buf = b""
                    else:
                        buf += ch

            _t = threading.Thread(target=_read_stderr, args=(proc.stderr,), daemon=True)
            _t.start()

            print()
            while proc.poll() is None:
                f = _enc_frame[0]
                pct = int(100 * f / max(n_t, 1))
                filled = pct // 5
                bar = "█" * filled + "░" * (20 - filled)
                print(f"  Encoding [{bar}] {pct:3d}%  {f}/{n_t} frames", end="\r")
                _time.sleep(0.2)

            _t.join()
            bar = "█" * 20
            print(f"  Encoding [{bar}] 100%  {n_t}/{n_t} frames")

            if proc.returncode != 0:
                raise RuntimeError("ffmpeg failed:\n" + "\n".join(_stderr_lines))
            print(f"  Copying to final destination...")
            _shutil.move(tmp_path, str(out_path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    else:
        # For tif/gif: accumulate frames then write
        frames = []
        for i, t_i in enumerate(t_indices, start=1):
            if i % 10 == 0:
                print(f"  frame {i}/{n_t} (t={t_i})", end="\r")
            frames.append(_update_figure(t_i))
        plt.close(fig)
        print(f"  frame {n_t}/{n_t} — done.  Writing {suffix}...")
        if suffix in (".tif", ".tiff"):
            import tifffile as _tiff
            _tiff.imwrite(str(out_path), np.stack(frames), photometric="rgb",
                          imagej=True, metadata={"axes": "TYXS"})
        else:
            import imageio.v2 as imageio
            imageio.mimsave(str(out_path), frames, fps=fps)

    print(f"Saved → {out_path}")