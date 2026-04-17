# croparray/build.py  

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, Literal, Any, Dict, overload, TYPE_CHECKING
if TYPE_CHECKING:
    from .crop_array_object import CropArray
    from .trackarray.object import TrackArray
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from datetime import datetime

__all__ = ["standardize_video_axes","standardize_spots", "create_crop_array", "make_croparrays", "open_measure_concat", "auto_minmass", "preview_detection", "review_thresholds", "make_csvs"]

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

def _detect_tracker_from_columns(cols) -> str:
    cols = set(cols)
    if {"POSITION_X", "POSITION_Y", "POSITION_Z", "FRAME"}.issubset(cols):
        return "trackmate"
    cols_lc = {c.lower() for c in cols}
    if {"x", "y", "frame"}.issubset(cols_lc):
        return "trackpy"
    if {"xc", "yc", "f"}.issubset(cols_lc) or {"xc", "yc", "frame"}.issubset(cols_lc):
        return "croparray"
    return "unknown"

def _detect_tracker_from_path(path: Path) -> str:
    # quick sniff: TrackMate has those POSITION_* headers on row 0
    try:
        df0 = pd.read_csv(path, nrows=1)
        t = _detect_tracker_from_columns(df0.columns)
        if t != "unknown":
            return t
    except Exception:
        pass
    return "unknown"

def _trackmate_units_from_csv(path: Path) -> dict:
    """
    Return a dict mapping column name -> unit string (lowercased),
    e.g. {"POSITION_X": "micron", "POSITION_T": "sec"}.
    """
    import csv

    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= 6:
                break

    if len(rows) < 4:
        return {}

    header = rows[0]
    units_row = rows[3]  # TrackMate: units usually live here

    units = {}
    for j, col in enumerate(header):
        if j < len(units_row):
            u = (units_row[j] or "").strip().lower()
            u = u.strip("()")  # "(micron)" -> "micron"
            if u:
                units[col] = u
    return units

def _read_trackmate_csv(path: Path) -> "pd.DataFrame":
    # robust TrackMate preamble handling (skips abbrev/units rows)
    import csv
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if i >= 25:
                break

    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    header = rows[0]
    try:
        id_idx = header.index("ID")
    except ValueError:
        # common TrackMate export: 3-line preamble
        skiprows = [1, 2, 3]
        return pd.read_csv(path, header=0, skiprows=skiprows)

    start_row = 1
    for i in range(1, len(rows)):
        v = rows[i][id_idx] if id_idx < len(rows[i]) else ""
        if isinstance(v, str) and v.strip().isdigit():
            start_row = i
            break

    skiprows = list(range(1, start_row))
    return pd.read_csv(path, header=0, skiprows=skiprows)

def _standardize_spots_df(
    df: "pd.DataFrame",
    *,
    tracker: str,
    fov: int,
    xy_um_per_px: Optional[float],
    z_um_per_plane: Optional[float],
    keep_cols: Optional[Sequence[str]],
    units: Optional[dict] = None,
    require_calibration_if_needed: bool = True,
) -> "pd.DataFrame":
    """
    Internal: normalize a spots dataframe into croparray canonical columns.

    Canonical required output columns:
      fov, id, f, yc, xc, zc
    Optional:
      track_id
    """
    tracker = tracker.lower()
    df = df.copy()

    # -----------------------------
    # Helper: normalize unit strings
    # -----------------------------
    def _norm_unit(u: Optional[str]) -> str:
        if u is None:
            return ""
        u = str(u).strip().lower()
        u = u.strip("()")
        # common variants
        if u in ("pixel", "pixels", "px"):
            return "px"
        if u in ("micron", "microns", "um", "µm", "micrometer", "micrometers"):
            return "um"
        if u in ("nm", "nanometer", "nanometers"):
            return "nm"
        return u  # unknown; pass through

    def _unit_for(colname: str) -> str:
        if not units:
            return ""
        return _norm_unit(units.get(colname, ""))

    # -----------------------------
    # TRACKMATE
    # -----------------------------
    if tracker == "trackmate":
        # rename TrackMate -> internal names (X_um/Y_um/Z_um are "position in exported spatial units")
        rename = {
            "POSITION_X": "X_um",
            "POSITION_Y": "Y_um",
            "POSITION_Z": "Z_um",
            "FRAME": "f",
            "TRACK_ID": "track_id",
            "ID": "id",
        }
        df = df.rename(columns=rename, errors="ignore")

        # Positions could also already be in xc/yc/zc if user preprocessed
        has_xyz = {"X_um", "Y_um"}.issubset(df.columns)
        has_z = "Z_um" in df.columns

        if not has_xyz and not {"xc", "yc"}.issubset(df.columns):
            raise ValueError(
                "TrackMate spots missing position columns. "
                "Expected POSITION_X/POSITION_Y (or already-standardized xc/yc)."
            )

        # Determine exported spatial units from preamble (if available)
        x_unit = _unit_for("POSITION_X")
        y_unit = _unit_for("POSITION_Y")
        z_unit = _unit_for("POSITION_Z")

        # Decide conversion needs
        # If unit is blank/unknown, we assume already in pixel units (no conversion)
        needs_xy_convert = (x_unit in ("um", "nm")) or (y_unit in ("um", "nm"))
        needs_z_convert = (z_unit in ("um", "nm"))

        # Build xc/yc from X_um/Y_um if present, else assume already xc/yc
        if has_xyz:
            if needs_xy_convert:
                if xy_um_per_px is None:
                    if require_calibration_if_needed:
                        raise ValueError(
                            f"TrackMate POSITION_X/Y appear to be in '{x_unit or y_unit}', "
                            "but xy_um_per_px was not provided."
                        )
                    # fallback: treat as pixels
                    df["xc"] = df["X_um"]
                    df["yc"] = df["Y_um"]
                else:
                    # If exported in nm, convert using nm_per_px = xy_um_per_px*1000
                    denom = float(xy_um_per_px) * (1000.0 if (x_unit == "nm" or y_unit == "nm") else 1.0)
                    df["xc"] = df["X_um"] / denom
                    df["yc"] = df["Y_um"] / denom
            else:
                # px/unknown -> treat as already pixels
                df["xc"] = df["X_um"]
                df["yc"] = df["Y_um"]

        # zc
        if "zc" not in df.columns:
            if has_z:
                if needs_z_convert:
                    if z_um_per_plane is None:
                        if require_calibration_if_needed:
                            raise ValueError(
                                f"TrackMate POSITION_Z appears to be in '{z_unit}', "
                                "but z_um_per_plane was not provided."
                            )
                        df["zc"] = df["Z_um"]
                    else:
                        denom = float(z_um_per_plane) * (1000.0 if z_unit == "nm" else 1.0)
                        df["zc"] = df["Z_um"] / denom
                else:
                    df["zc"] = df["Z_um"]
            else:
                # no z in TrackMate export -> treat as singleton z=0
                df["zc"] = 0.0
        # ------------------------------------------------
        # Detect TrackMate "fake Z" from 2D tracking
        # (TrackMate exports POSITION_Z even for 2D movies)
        # If all z values are identical, drop zc entirely
        # so downstream code treats the data as 2D.
        # ------------------------------------------------
        if "zc" in df.columns:
            zvals = pd.to_numeric(df["zc"], errors="coerce")
            zvals = zvals[np.isfinite(zvals)]

            if len(zvals) > 0 and zvals.max() - zvals.min() < 1e-6:
                df = df.drop(columns=["zc"])

    # -----------------------------
    # TRACKPY
    # -----------------------------
    elif tracker == "trackpy":
        cols_lc = {c.lower(): c for c in df.columns}

        def _col(name: str) -> str:
            return cols_lc.get(name, name)

        if _col("frame") in df.columns and "f" not in df.columns:
            df = df.rename(columns={_col("frame"): "f"})
        if _col("x") in df.columns and "xc" not in df.columns:
            df = df.rename(columns={_col("x"): "xc"})
        if _col("y") in df.columns and "yc" not in df.columns:
            df = df.rename(columns={_col("y"): "yc"})
        if "zc" not in df.columns:
            df["zc"] = 0.0
        if _col("particle") in df.columns and "track_id" not in df.columns:
            df = df.rename(columns={_col("particle"): "track_id"})
        if "id" not in df.columns:
            df["id"] = np.arange(len(df), dtype=int)

    # -----------------------------
    # CROPARAY / UNKNOWN (best-effort)
    # -----------------------------
    else:
        cols_lc = {c.lower(): c for c in df.columns}
        if "f" not in df.columns and "frame" in cols_lc:
            df = df.rename(columns={cols_lc["frame"]: "f"})
        if "xc" not in df.columns and "x" in cols_lc:
            df = df.rename(columns={cols_lc["x"]: "xc"})
        if "yc" not in df.columns and "y" in cols_lc:
            df = df.rename(columns={cols_lc["y"]: "yc"})
        if "zc" not in df.columns:
            df["zc"] = 0.0
        if "id" not in df.columns:
            df["id"] = np.arange(len(df), dtype=int)

    # -----------------------------
    # Final required columns check
    # -----------------------------
    required = {"id", "f", "xc", "yc"}   # zc is optional
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Spots table missing required columns {sorted(missing)}. Got {list(df.columns)}"
        )

    # If zc is absent, leave it absent (signals 2D / unknown-z to downstream)
    # (Alternatively: set df["zc"]=0.0 if you prefer always having it.)

    # Enforce numeric types where it matters
    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(-1).astype(np.int64)
    df["f"] = pd.to_numeric(df["f"], errors="coerce").fillna(0).astype(np.int32)

    # Keep xc/yc/zc as float (subpixel ok)
    df["xc"] = pd.to_numeric(df["xc"], errors="coerce")
    df["yc"] = pd.to_numeric(df["yc"], errors="coerce")
    if "zc" in df.columns:
        df["zc"] = pd.to_numeric(df["zc"], errors="coerce").fillna(0.0)

    # fov assignment
    df["fov"] = int(fov)

    # -----------------------------
    # Choose output columns
    # -----------------------------
    base_cols = ["fov"]
    if "track_id" in df.columns:
        base_cols += ["track_id"]

    base_cols += ["id", "f", "yc", "xc"]
    if "zc" in df.columns:
        base_cols += ["zc"]


    import re

    if keep_cols is None:
        extra = [c for c in df.columns if "INTENSITY" in c.upper()]
    else:
        # If user requested something like MAX_INTENSITY_CH1, automatically keep all channels
        # that exist for that base name: MAX_INTENSITY_CH1/CH2/CH3...
        ch_pat = re.compile(r"^(?P<base>.+)_CH(?P<ch>\d+)$", re.IGNORECASE)

        extra = []
        seen = set()

        for req in keep_cols:
            m = ch_pat.match(req)
            if m:
                base_name = m.group("base")
                # collect all sibling channel columns that exist
                siblings = []
                for c in df.columns:
                    m2 = ch_pat.match(c)
                    if m2 and m2.group("base").lower() == base_name.lower():
                        siblings.append((int(m2.group("ch")), c))
                siblings.sort(key=lambda x: x[0])  # CH1, CH2, ...

                for _, c in siblings:
                    if c not in seen:
                        extra.append(c)
                        seen.add(c)
            else:
                if req in df.columns and req not in seen:
                    extra.append(req)
                    seen.add(req)


    return df[base_cols + extra].copy()

def standardize_spots(
    source: Union[str, Path, "pd.DataFrame"],
    *,
    tracker: Literal["auto", "trackmate", "trackpy", "croparray"] = "auto",
    fov: int = 0,
    xy_um_per_px: Optional[float] = None,
    z_um_per_plane: Optional[float] = None,
    keep_cols: Optional[Sequence[str]] = None,
    require_calibration_if_needed: bool = True,
) -> "pd.DataFrame":
    """
    Convert tracker output into a croparray-ready spots dataframe.

    Output (always includes):
      - fov, id, f, yc, xc, zc
    Output (optional):
      - track_id (if present in the input)
    Plus:
      - extra columns requested by `keep_cols` (or default intensity columns)

    Unit handling (TrackMate)
    -------------------------
    TrackMate exports POSITION_X/Y/Z in the *image spatial units* at export time
    (often micron/um; sometimes pixel). We read the units row from the CSV and:

      - if units indicate pixel/px: treat values as already in pixels/planes
      - if units indicate micron/um: convert to pixels/planes using xy_um_per_px / z_um_per_plane
      - if units indicate nm: convert to pixels/planes using xy_um_per_px / z_um_per_plane (with nm scaling)

    If conversion is needed but calibration is missing:
      - if require_calibration_if_needed=True (default), raise ValueError
      - else, fall back to treating coordinates as already pixel/plane units

    Parameters
    ----------
    source
        Path to a CSV file (TrackMate/TrackPy/etc.) or a pandas DataFrame.
    tracker
        "auto" | "trackmate" | "trackpy" | "croparray"
    fov
        Field-of-view index to assign to all rows in this spots table.
    xy_um_per_px
        Microns per pixel (used when TrackMate units are micron/um/nm).
    z_um_per_plane
        Microns per z-plane (used when TrackMate units are micron/um/nm).
    keep_cols
        Extra columns to retain in the output. If None, keeps intensity-like columns by default.
    require_calibration_if_needed
        If True, raise when TrackMate units require conversion but calibration is missing.

    Returns
    -------
    pd.DataFrame
    """
    # ---- DataFrame input ----
    if isinstance(source, pd.DataFrame):
        df = source.copy()
        if tracker == "auto":
            tracker = _detect_tracker_from_columns(df.columns)

        units = None  # no CSV preamble available
        return _standardize_spots_df(
            df,
            tracker=tracker,
            fov=fov,
            xy_um_per_px=xy_um_per_px,
            z_um_per_plane=z_um_per_plane,
            keep_cols=keep_cols,
            units=units,
            require_calibration_if_needed=require_calibration_if_needed,
        )

    # ---- Path input ----
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(path)

    if tracker == "auto":
        tracker = _detect_tracker_from_path(path)

    tracker_lc = tracker.lower()

    # Read input into a dataframe + any sidecar metadata needed for standardization
    units = None
    if tracker_lc == "trackmate":
        # Get units (micron/pixel/etc.) from TrackMate CSV preamble
        units = _trackmate_units_from_csv(path)
        df = _read_trackmate_csv(path)
    else:
        # For trackpy/croparray/unknown: straight read for now
        df = pd.read_csv(path)

        # If auto-detect said unknown, try column sniff now that we have the header
        if tracker_lc == "unknown":
            tracker_lc = _detect_tracker_from_columns(df.columns)

    return _standardize_spots_df(
        df,
        tracker=tracker_lc,
        fov=fov,
        xy_um_per_px=xy_um_per_px,
        z_um_per_plane=z_um_per_plane,
        keep_cols=keep_cols,
        units=units,
        require_calibration_if_needed=require_calibration_if_needed,
    )

def auto_minmass(
    frames,
    diameter: int,
    *,
    n_thresholds: int = 50,
    percentile: int = 64,
    separation: Optional[int] = None,
    n_sample_frames: int = 10,
    max_spots: int = 200,
) -> float:
    """
    Automatically select a TrackPy ``minmass`` threshold using the knee/elbow
    of the spots-detected-vs-threshold curve (same concept as BigFISH auto-threshold).

    The approach:
      1. Detect all spots with ``minmass=0`` on a sample of frames.
      2. Scan log-spaced thresholds from the 1st to the 99th percentile of spot masses.
      3. Find the knee (max curvature / Kneedle method).
      4. Also find the threshold that gives ``max_spots`` detections per frame.
      5. Return whichever is more conservative (higher threshold), so the result
         never exceeds ``max_spots`` spots per frame.

    Parameters
    ----------
    frames
        A 2D numpy array (single frame) or a 3D array (t, y, x).
    diameter
        Spot diameter in pixels (odd integer).
    n_thresholds
        Number of threshold values to try (log-spaced).
    percentile
        TrackPy ``percentile`` argument passed to ``tp.locate``.
    separation
        TrackPy ``separation`` argument. ``None`` uses TrackPy default (= diameter).
    n_sample_frames
        Max frames to sample when ``frames`` is 3D.
    max_spots
        Safety cap: if the knee threshold would detect more than this many spots
        per frame on average, raise the threshold until the count is at or below
        this value.  Default 200.

    Returns
    -------
    float
        Recommended ``minmass`` threshold.
    """
    try:
        import trackpy as tp
    except ImportError as e:
        raise ImportError("trackpy is required for auto_minmass. Install with: conda install -c conda-forge trackpy") from e

    frames = np.asarray(frames)
    if frames.ndim == 2:
        sample = [frames]
        n_frames_sampled = 1
    else:
        n = frames.shape[0]
        idx = np.linspace(0, n - 1, min(n, n_sample_frames), dtype=int)
        sample = [frames[i] for i in idx]
        n_frames_sampled = len(idx)

    pieces = []
    for f in sample:
        spots = tp.locate(f, diameter=diameter, minmass=0,
                          separation=separation, percentile=percentile)
        pieces.append(spots)

    all_spots = pd.concat(pieces, ignore_index=True) if len(pieces) > 1 else pieces[0]

    if len(all_spots) == 0:
        return 0.0

    masses = all_spots["mass"].values
    lo = float(np.percentile(masses, 1))
    hi = float(np.percentile(masses, 99))

    if hi <= lo:
        return lo

    thresholds = np.logspace(np.log10(max(lo, 1.0)), np.log10(hi), n_thresholds)
    # counts per frame (average across sampled frames)
    counts = np.array([(masses >= t).sum() for t in thresholds], dtype=float) / n_frames_sampled

    # Knee: max distance from diagonal
    x = (thresholds - thresholds[0]) / (thresholds[-1] - thresholds[0])
    denom = counts[0] - counts[-1]
    y = (counts - counts[-1]) / (denom if denom > 0 else 1.0)
    distances = np.abs(x + y - 1.0) / np.sqrt(2.0)
    knee_thresh = float(thresholds[int(np.argmax(distances))])

    # Safety cap: find the lowest threshold where avg spots/frame <= max_spots
    below = np.where(counts <= max_spots)[0]
    cap_thresh = float(thresholds[below[0]]) if len(below) > 0 else float(thresholds[-1])

    return max(knee_thresh, cap_thresh)


def _load_detect_channel(path, axes: str, detect_ch: int) -> "np.ndarray":
    """Load a .tif and return (n_frames, y, x) for a single channel, max-projecting z if present."""
    from tifffile import imread
    vid = imread(path)
    axes = axes.lower()
    if 'c' in axes:
        vid = np.take(vid, detect_ch, axis=axes.index('c'))
        axes = axes.replace('c', '')
    if 'z' in axes:
        vid = vid.max(axis=axes.index('z'))
        axes = axes.replace('z', '')
    return vid  # (t, y, x)


def preview_detection(
    video_path,
    *,
    detect_ch: int = 0,
    axes: Optional[str] = None,
    define_axes: bool = True,
    diameter: int = 7,
    minmass: "Union[float, str]" = "auto",
    separation: Optional[int] = None,
    percentile: int = 64,
    frame_index: Optional[int] = None,
    max_spots: int = 200,
) -> None:
    """
    Show the minmass elbow curve and an annotated frame side-by-side for one video.
    Use this to tune DIAMETER and MINMASS before running make_csvs.
    """
    try:
        import trackpy as tp
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("trackpy and matplotlib are required for preview_detection.") from e

    axes_use = axes
    if axes_use is None and define_axes:
        from . import gui as _gui
        axes_use = "".join(_gui.define_video_axes(path=video_path).axes)

    vid = _load_detect_channel(video_path, axes_use, detect_ch)
    auto_thresh = auto_minmass(vid, diameter, percentile=percentile, separation=separation,
                               max_spots=max_spots)
    thresh = auto_thresh if minmass == "auto" else float(minmass)

    # elbow curve data
    n = len(vid)
    idx = np.linspace(0, n - 1, min(n, 10), dtype=int)
    all_spots = pd.concat(
        [tp.locate(vid[i], diameter=diameter, minmass=0,
                   separation=separation, percentile=percentile) for i in idx],
        ignore_index=True,
    )
    masses = all_spots["mass"].values
    thresholds = np.logspace(
        np.log10(max(float(np.percentile(masses, 1)), 1.0)),
        np.log10(float(np.percentile(masses, 99))),
        50,
    )
    counts = [(masses >= t).sum() for t in thresholds]

    fi = frame_index if frame_index is not None else n // 2
    frame_spots = tp.locate(vid[fi], diameter=diameter, minmass=thresh,
                             separation=separation, percentile=percentile)

    fig, axes_ax = plt.subplots(1, 2, figsize=(12, 5))

    axes_ax[0].plot(thresholds, counts, "k-")
    axes_ax[0].axvline(auto_thresh, color="r", linestyle="--", label=f"auto = {auto_thresh:.0f}")
    if minmass != "auto":
        axes_ax[0].axvline(thresh, color="b", linestyle=":", label=f"manual = {thresh:.0f}")
    axes_ax[0].set_xscale("log")
    axes_ax[0].set_xlabel("minmass threshold")
    axes_ax[0].set_ylabel("spots detected")
    axes_ax[0].set_title("Elbow curve")
    axes_ax[0].legend()

    tp.annotate(frame_spots, vid[fi], ax=axes_ax[1], plot_style={"markersize": 8})
    axes_ax[1].set_title(f"Frame {fi}  —  {len(frame_spots)} spots at minmass={thresh:.0f}")

    fig.suptitle(Path(video_path).name, fontsize=10)
    plt.tight_layout()
    print(f"Auto-detected minmass: {auto_thresh:.1f}  |  Using: {thresh:.1f}")


def review_thresholds(
    videos,
    *,
    detect_ch: int = 0,
    axes: Optional[str] = None,
    define_axes: bool = True,
    axes_source_index: int = 0,
    diameter: int = 7,
    percentile: int = 64,
    separation: Optional[int] = None,
    frame_index: Optional[int] = None,
    max_spots: int = 200,
    n_thresholds: int = 50,
) -> dict:
    """
    Interactive GUI to review and adjust per-video minmass thresholds.

    Shows one video at a time: annotated frame (left) + elbow curve (right),
    with a dropdown to switch between videos and a slider to adjust minmass.
    Updating the slider re-filters pre-detected spots instantly.

    Returns a dict {filename: threshold} for use with make_csvs(minmass=...).
    """
    try:
        import trackpy as tp
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except ImportError as e:
        raise ImportError("trackpy, ipywidgets, and matplotlib are required.") from e

    def _as_list(x):
        if isinstance(x, (str, Path)):
            return [x]
        return list(x)

    video_list = _as_list(videos)

    axes_use = axes
    if axes_use is None and define_axes:
        from . import gui as _gui
        axes_use = "".join(_gui.define_video_axes(path=video_list[axes_source_index]).axes)

    result = {}

    print("Pre-detecting spots (minmass=0) on sample frame for each video...")
    precomp = []
    for vid_path in video_list:
        vid_path = Path(vid_path)
        vid = _load_detect_channel(vid_path, axes_use, detect_ch)
        fi = frame_index if frame_index is not None else len(vid) // 2
        frame = vid[fi]

        all_spots = tp.locate(frame, diameter=diameter, minmass=0,
                              separation=separation, percentile=percentile)

        auto_thresh = auto_minmass(vid, diameter, percentile=percentile,
                                   separation=separation, max_spots=max_spots)

        masses = all_spots["mass"].values if len(all_spots) > 0 else np.array([1.0])
        lo = max(float(np.percentile(masses, 1)), 1.0)
        hi = float(np.percentile(masses, 99))
        thresholds = np.logspace(np.log10(lo), np.log10(max(hi, lo * 2)), n_thresholds)
        counts = np.array([(masses >= t).sum() for t in thresholds])

        result[vid_path.name] = float(auto_thresh)
        precomp.append(dict(path=vid_path, frame=frame, all_spots=all_spots,
                            auto_thresh=float(auto_thresh), masses=masses,
                            thresholds=thresholds, counts=counts))

    print("Done. Adjust sliders; plots update on release.")

    # --- widgets ---
    names = [vd["path"].name for vd in precomp]
    dropdown = widgets.Dropdown(options=names, description="Video:",
                                layout=widgets.Layout(width="600px"))

    vd0 = precomp[0]
    lo0 = float(np.log10(max(vd0["masses"].min(), 1.0)))
    hi0 = float(np.log10(vd0["masses"].max()))
    slider = widgets.FloatLogSlider(
        value=vd0["auto_thresh"], base=10, min=lo0, max=hi0, step=0.01,
        description="minmass:", continuous_update=False,
        layout=widgets.Layout(width="600px"),
    )
    img_widget = widgets.Image(format="png", layout=widgets.Layout(width="800px"))

    def draw(vd, thresh):
        import io
        import matplotlib.pyplot as plt
        filtered = vd["all_spots"][vd["all_spots"]["mass"] >= thresh] \
            if len(vd["all_spots"]) > 0 else vd["all_spots"]
        with plt.ioff():
            fig, (ax_frame, ax_elbow) = plt.subplots(
                1, 2, figsize=(10, 4),
                gridspec_kw={"width_ratios": [2, 1]},
            )
            tp.annotate(filtered, vd["frame"], ax=ax_frame,
                        plot_style={"markersize": 6, "markeredgewidth": 1})
            ax_frame.set_title(f"{len(filtered)} spots  |  minmass={thresh:.0f}", fontsize=9)
            ax_frame.set_xticks([]); ax_frame.set_yticks([])
            ax_elbow.plot(vd["thresholds"], vd["counts"], "k-", lw=1.5)
            ax_elbow.axvline(thresh, color="r", linestyle="--", lw=1.5,
                             label=f"{thresh:.0f}")
            ax_elbow.axvline(vd["auto_thresh"], color="gray", linestyle=":",
                             lw=1, alpha=0.7, label=f"auto={vd['auto_thresh']:.0f}")
            ax_elbow.set_xscale("log")
            ax_elbow.set_xlabel("minmass", fontsize=9)
            ax_elbow.set_ylabel("spots", fontsize=9)
            ax_elbow.legend(fontsize=8)
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
        buf.seek(0)
        img_widget.value = buf.read()

    def on_dropdown(change):
        vd = precomp[names.index(change["new"])]
        lo = float(np.log10(max(vd["masses"].min(), 1.0)))
        hi = float(np.log10(vd["masses"].max()))
        slider.unobserve(on_slider, names="value")
        slider.min, slider.max = lo, hi
        slider.value = vd["auto_thresh"]
        slider.observe(on_slider, names="value")
        draw(vd, vd["auto_thresh"])

    def on_slider(change):
        vd = precomp[names.index(dropdown.value)]
        result[vd["path"].name] = float(change["new"])
        draw(vd, change["new"])

    dropdown.observe(on_dropdown, names="value")
    slider.observe(on_slider, names="value")

    draw(vd0, vd0["auto_thresh"])
    display(widgets.VBox([dropdown, img_widget, slider]))

    return result


def make_csvs(
    videos,
    *,
    detect_ch: int = 0,
    axes: Optional[str] = None,
    define_axes: bool = True,
    axes_source_index: int = 0,
    diameter: int = 7,
    minmass: "Union[float, str]" = "auto",
    separation: Optional[int] = None,
    percentile: int = 64,
    track: bool = True,
    search_range: float = 5,
    memory: int = 2,
    min_track_len: int = 3,
    out_dir: Optional[Union[str, Path]] = None,
    out_suffix: str = "_allspots",
    skip_existing: bool = True,
    progress: bool = True,
) -> list:
    """
    Detect spots with TrackPy and save one CSV per video, ready for make_croparrays.

    Parameters
    ----------
    videos
        A single path or a list of paths to .tif files.
    detect_ch
        Channel index (0-based) to use for spot detection.
    axes
        Axis order string, e.g. 'tcyx'. If None and define_axes=True, opens the
        axis-labeling GUI on the video at axes_source_index.
    define_axes
        If True and axes is None, open the GUI to label axes once for the whole batch.
    axes_source_index
        Which video to use when opening the GUI (default 0).
    diameter
        Spot diameter in pixels (odd integer).
    minmass
        Minimum integrated brightness threshold. 'auto' uses auto_minmass per video.
    track
        If True, link spots into tracks with tp.link and filter short ones.
    search_range
        Max displacement (pixels) between frames for tp.link.
    memory
        Frames a spot may disappear and still be linked.
    min_track_len
        Discard tracks shorter than this (frames).
    out_dir
        Directory for CSV output. Defaults to each video's parent directory.
    out_suffix
        String appended to the video stem before '.csv'.
    skip_existing
        If True, skip videos whose output CSV already exists.

    Returns
    -------
    list of Path
        Paths of written CSV files.
    """
    try:
        import trackpy as tp
    except ImportError as e:
        raise ImportError("trackpy is required for make_csvs. Install with: conda install -c conda-forge trackpy") from e

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    def _as_list(x):
        if isinstance(x, (str, Path)):
            return [x]
        return list(x)

    video_list = _as_list(videos)

    # Resolve axes — GUI fires once for the whole batch, same as make_croparrays
    axes_use = axes
    if axes_use is None and define_axes:
        from . import gui as _gui
        axes_result = _gui.define_video_axes(path=video_list[axes_source_index])
        axes_use = "".join(axes_result.axes)

    written = []
    it = video_list
    if progress and tqdm is not None:
        it = tqdm(it, desc="make_csvs", unit="file")

    for vid_path in it:
        vid_path = Path(vid_path)
        csv_dir = Path(out_dir) if out_dir is not None else vid_path.parent
        csv_dir.mkdir(parents=True, exist_ok=True)
        out_csv = csv_dir / f"{vid_path.stem}{out_suffix}.csv"

        if skip_existing and out_csv.exists():
            print(f"[skip] {out_csv.name}")
            continue

        vid = _load_detect_channel(vid_path, axes_use, detect_ch)

        if isinstance(minmass, dict):
            val = minmass.get(vid_path.name, minmass.get(vid_path.stem, "auto"))
            thresh = auto_minmass(vid, diameter, percentile=percentile, separation=separation) \
                if val == "auto" else float(val)
        elif minmass == "auto":
            thresh = auto_minmass(vid, diameter, percentile=percentile, separation=separation)
        else:
            thresh = float(minmass)

        spots = tp.batch(vid, diameter=diameter, minmass=thresh,
                         separation=separation, percentile=percentile)

        if track:
            spots = tp.link(spots, search_range=search_range, memory=memory)
            if min_track_len > 1:
                spots = tp.filter_stubs(spots, threshold=min_track_len)

        spots.to_csv(out_csv, index=False)
        n_tracks = spots["particle"].nunique() if "particle" in spots.columns else "N/A"
        print(f"[done] {out_csv.name}  ({len(spots)} detections, {n_tracks} tracks, minmass={thresh:.0f})")
        written.append(out_csv)

    return written


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
    import re
    from collections import defaultdict
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
    # extra numeric df columns -> layers
    #   scalar: (fov, n, t)
    #   channelized (suffix _CHk): (fov, n, t, ch)
    # -------------------------
    base_coord_cols = {"fov", "f", "yc", "xc", "zc", "n"}

    # candidate columns (everything except coordinates)
    extra_cols = [c for c in df.columns if c not in base_coord_cols]

    # ensure id/track_id exist as layers
    for required_layer in ("id", "track_id"):
        if required_layer not in extra_cols:
            extra_cols.append(required_layer)

    # detect *_CHk groups
    ch_pat = re.compile(r"^(?P<base>.+)_CH(?P<ch>\d+)$", re.IGNORECASE)
    ch_groups = defaultdict(list)  # base -> list[(colname, ch_index)]
    scalar_cols = []

    for c in extra_cols:
        m = ch_pat.match(c)
        if m:
            base = m.group("base")
            ch_num = int(m.group("ch"))
            ch_index = ch_num - 1  # TrackMate CH1->0
            ch_groups[base].append((c, ch_index))
        else:
            scalar_cols.append(c)

    # validate channel indices against video channels
    for base, items in ch_groups.items():
        max_idx = max(idx for _, idx in items)
        if max_idx >= n_channels:
            raise ValueError(
                f"Found TrackMate columns for {base} up to CH{max_idx+1}, "
                f"but video has only n_channels={n_channels}. "
                "Either export fewer channels or expand the video channel dimension."
            )

    # allocate scalar layers (use float32 to avoid clipping TrackMate intensities)
    my_scalar_layers = np.full(
        (len(scalar_cols), n_fov, n_spots_max, n_frames),
        np.nan,
        dtype=np.float32,
    )

    # allocate channelized layers: one array per base name
    my_ch_layers = {
        base: np.full((n_fov, n_spots_max, n_frames, n_channels), np.nan, dtype=np.float32)
        for base in ch_groups
    }

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
            for col_counter, col in enumerate(scalar_cols):
                vals = pd.to_numeric(my_spots[col], errors="coerce").astype(np.float32).to_numpy()
                my_scalar_layers[col_counter, my_fov_ind, : len(vals), int(my_f)] = vals

            # fill channelized layers
            for base, items in ch_groups.items():
                for colname, ch_index in items:
                    vals = pd.to_numeric(my_spots[colname], errors="coerce").astype(np.float32).to_numpy()
                    my_ch_layers[base][my_fov_ind, : len(vals), int(my_f), ch_index] = vals


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
    signal_units = str(kwargs.get("signal_units", "a.u."))
    intensity = xr.DataArray(
        my_crops_all,
        coords=[fov, n, t, z, y, x, ch],
        dims=["fov", "n", "t", "z", "y", "x", "ch"],
        attrs={"units": signal_units, "long_name": "intensity (a.u.)"},
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

    # -------------------------
    # scalar optional layers (fov,n,t)
    # -------------------------
    scalar_optional_layers = [
        xr.DataArray(my_scalar_layers[i], coords=[fov, n, t], dims=["fov", "n", "t"])
        for i in range(len(scalar_cols))
    ]
    scalar_dict = dict(zip(scalar_cols, scalar_optional_layers))

    # attrs for known ones
    if "id" in scalar_dict:
        scalar_dict["id"].attrs.update({"units": "index", "long_name": "spot id"})
    if "track_id" in scalar_dict:
        scalar_dict["track_id"].attrs.update({"units": "index", "long_name": "track assignment (-1 = untracked)"})

    # -------------------------
    # channelized optional layers (fov,n,t,ch) using base names
    # -------------------------
    ch_dict = {}
    for base, arr in my_ch_layers.items():
        ch_dict[base] = xr.DataArray(arr, coords=[fov, n, t, ch], dims=["fov", "n", "t", "ch"])

    ds = xr.Dataset(
        {
            "int": intensity,
            "xc": xc, "yc": yc, "zc": zc,
            "xc_pix": xc_pix, "yc_pix": yc_pix, "zc_pix": zc_pix,
            "xc_pad": xc_pad, "yc_pad": yc_pad,
            "z_pos": z_pos,
            **scalar_dict,
            **ch_dict,
        },
        attrs={"name": str(name), "date": str(date)},
    )
    # define unit strings once (plain python str)
    xyz_units = str(units[0]) if units and len(units) > 0 else "space"
    t_units   = str(units[1]) if units and len(units) > 1 else "time"
    signal_units = str(kwargs.get("signal_units", "a.u."))

    ds = xr.Dataset(
        {
            "int": intensity,
            "xc": xc, "yc": yc, "zc": zc,
            "xc_pix": xc_pix, "yc_pix": yc_pix, "zc_pix": zc_pix,
            "xc_pad": xc_pad, "yc_pad": yc_pad,
            "z_pos": z_pos,
            **scalar_dict,
            **ch_dict,
        },
        attrs={"name": str(name), "date": str(date)},
    )

    ds.attrs.update(
        {
            "xy_pad": int(xy_pad),
            "z_pad": int(z_pad),
            "dx": float(my_dx),
            "dy": float(my_dy),
            "dz": float(my_dz),
            "dt": float(my_dt),
            "xyz_units": xyz_units,
            "t_units": t_units,
            "signal_units": signal_units,
            "croparray_schema_version": "2.0",
        }
    )

    return ds

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

def make_croparrays(
    videos,
    spots,
    *,
    out_dir: Optional[Union[str, Path]] = None,
    out_ext: str = ".nc",
    axes: Optional[Sequence[str]] = None,
    define_axes: bool = True,
    axes_source_index: int = 0,
    tracker: Literal["auto", "trackmate", "trackpy", "croparray"] = "auto",
    xy_um_per_px: Optional[float] = None,
    z_um_per_plane: Optional[float] = None,
    keep_cols: Optional[Sequence[str]] = None,
    xy_pad: int = 10,
    z_pad: int = 1,
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    dz: Optional[float] = None,
    dt: Optional[float] = None,
    units: Optional[Sequence[str]] = None,
    date: Optional[str] = None,
    as_object: bool = False,
    skip_existing: bool = True,
    progress: bool = True,
    notes: str = "A croparray built from a tracking file and a video.",
):
    """
    High-level convenience builder that accepts either:
      - single (video, spots) OR
      - lists of (videos, spots)

    It standardizes video axes, standardizes spots tables, builds croparrays,
    and optionally writes .nc files.

    Behavior:
      - Never overwrites existing outputs.
      - If out_dir is provided and skip_existing=True, existing .nc files are skipped.

    Returns:
      - If out_dir is None: returns a single dataset/object (single input) or list of them (batch)
      - If out_dir is provided: returns list of output paths written (skipped files omitted)
    """
    from pathlib import Path
    from tifffile import imread
    import pandas as pd
    import numpy as np
    import xarray as xr

    # tqdm is optional
    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None

    def _as_list(x):
        """Treat strings/Paths/DataFrames/ndarrays as singletons; otherwise assume iterable."""
        if isinstance(x, (str, Path, pd.DataFrame, np.ndarray)) or hasattr(x, "shape"):
            return [x]
        return list(x)

    def _normalize_attrs_for_netcdf(ds: xr.Dataset) -> xr.Dataset:
        """
        Make ds.attrs safe for NetCDF writers (especially SciPy/NetCDF3):
        - convert numpy scalars/arrays to python scalars/lists
        - ensure strings are plain Python str
        """
        def norm(v):
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if isinstance(v, (list, tuple)):
                return [norm(x) for x in v]
            if isinstance(v, (bytes, bytearray)):
                return v.decode("utf-8", errors="replace")
            if isinstance(v, np.str_):
                return str(v)
            if isinstance(v, Path):
                return str(v)
            return v

        ds = ds.copy()
        ds.attrs = {str(k): norm(v) for k, v in ds.attrs.items()}
        return ds

    def _atomic_to_netcdf(ds: xr.Dataset, out_file: Path) -> None:
        """
        Atomic write: write to temp then replace.
        Prevents 0-byte / truncated files when something fails mid-write.
        """
        tmp = out_file.with_suffix(out_file.suffix + ".tmp")

        # If a previous tmp exists, remove it
        if tmp.exists():
            tmp.unlink()

        try:
            ds.to_netcdf(tmp, mode="w")  # let xarray pick best available engine
            # Replace atomically
            tmp.replace(out_file)
        except Exception:
            # Clean up partial tmp
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass
            raise

    video_list = _as_list(videos)
    spots_list = _as_list(spots)

    if len(video_list) != len(spots_list):
        raise ValueError(
            f"videos and spots must have the same length; got {len(video_list)} vs {len(spots_list)}"
        )

    out_path_dir = Path(out_dir) if out_dir is not None else None
    if out_path_dir is not None:
        out_path_dir.mkdir(parents=True, exist_ok=True)

    # Determine axes once if needed
    axes_use = tuple(axes) if axes is not None else None
    if axes_use is None and define_axes:
        from . import gui as _gui  # local import avoids GUI import at module import-time
        axes_result = _gui.define_video_axes(path=video_list[axes_source_index])
        axes_use = axes_result.axes

    it = enumerate(zip(video_list, spots_list))
    if progress and tqdm is not None:
        it = tqdm(it, total=len(video_list), desc="croparray", unit="file")

    results = []
    written_paths = []

    for fov_index, (vf, sf) in it:
        out_file = None
        if out_path_dir is not None:
            vf_path = Path(vf) if isinstance(vf, (str, Path)) else None
            stem = vf_path.stem if vf_path is not None else f"video_{fov_index}"
            out_file = out_path_dir / f"{stem}{out_ext}"

            if out_file.exists():
                if skip_existing:
                    continue
                raise FileExistsError(f"{out_file} exists. Croparray will not overwrite existing files.")

        # Load video
        video_raw = imread(vf) if isinstance(vf, (str, Path)) else vf

        # Standardize axes
        video_std = standardize_video_axes(video_raw, axes_use) if axes_use is not None else video_raw

        # Standardize spots
        spots_std = standardize_spots(
            sf,
            tracker=tracker,
            fov=fov_index,
            xy_um_per_px=xy_um_per_px,
            z_um_per_plane=z_um_per_plane,
            keep_cols=keep_cols,
        )

        # Build crop array
        ca_data = create_crop_array(
            video_std,
            spots_std,
            xy_pad=xy_pad,
            z_pad=z_pad,
            dx=dx,
            dy=dy,
            dz=dz,
            dt=dt,
            units=units,
            name=str(vf),
            date=date,
            as_object=as_object,
        )
        results.append(ca_data)

        ds = ca_data.ds if hasattr(ca_data, "ds") else ca_data
        ds.attrs["notes"] = str(notes)

        ds = _normalize_attrs_for_netcdf(ds)

        if out_file is not None:
            _atomic_to_netcdf(ds, out_file)
            written_paths.append(out_file)

    if out_path_dir is not None:
        return written_paths

    return results[0] if len(results) == 1 else results

@overload
def open_measure_concat(*, groups: Any, dims: Sequence[str], labels: Optional[Sequence[Optional[Sequence[str]]]] = None, measure_kwargs: Optional[Dict[str, Any]] = None, drop_vars: Any = None, open_as: Literal["trackarray"], join: str = "outer", attach_provenance: bool = True) -> "TrackArray": ...
@overload
def open_measure_concat(*, groups: Any, dims: Sequence[str], labels: Optional[Sequence[Optional[Sequence[str]]]] = None, measure_kwargs: Optional[Dict[str, Any]] = None, drop_vars: Any = None, open_as: Literal["croparray"] = "croparray", join: str = "outer", attach_provenance: bool = True) -> "CropArray": ...
def open_measure_concat(
    *,
    groups: Any,
    dims: Sequence[str],
    labels: Optional[Sequence[Optional[Sequence[str]]]] = None,
    measure_kwargs: Optional[Dict[str, Any]] = None,
    drop_vars=None,
    open_as: str = "croparray",
    join: str = "outer",
    attach_provenance: bool = True,
) -> "CropArray | TrackArray":
    """
    Open CropArray/TrackArray files, measure translation-site signal, optionally
    drop variables, and concatenate results into a single object.

    This function is a convenience wrapper around file discovery, object opening,
    signal measurement, and concatenation. Each file is opened as either a
    ``TrackArray`` or ``CropArray`` object, the signal is measured using
    ``.measure_signal()``, optional variables are removed, and the resulting
    objects are concatenated.

    Processing order for each file
    ------------------------------
    1. Open the file as a wrapper object (``TrackArray`` or ``CropArray``).
    2. Run ``.measure_signal(**measure_kwargs)``.
    3. Optionally remove variables listed in ``drop_vars``.
    4. Concatenate all objects along the requested dimensions.

    Importantly, variables are dropped **after** signal measurement so that
    measurement can access any required raw data (e.g., intensity stacks).

    Parameters
    ----------
    root : str or Path
        Root directory containing CropArray/TrackArray files.

    pattern : str, default="*.nc"
        File pattern used to locate input files.

    recursive : bool, default=True
        If True, search for files recursively within ``root``.

    open_as : {"trackarray", "croparray"}, default="trackarray"
        Determines how files are opened and which wrapper class is returned.

        - ``"trackarray"`` → files opened with ``open_as_trackarray()``
        - ``"croparray"`` → files opened with ``open_croparray()``

    measure_kwargs : dict or None, default=None
        Keyword arguments passed to ``.measure_signal()``.

    drop_vars : sequence of str or None, default=None
        Dataset variables to remove *after* signal measurement. This is useful
        for removing large arrays (e.g., raw image stacks) to reduce memory
        usage before concatenation.

    concat_dims : sequence of str or None, default=None
        Dimensions along which concatenation should occur.

    labels : sequence or None, default=None
        Labels applied to concatenated dimensions (passed to the underlying
        concat routine).

    Returns
    -------
    TrackArray or CropArray
        A concatenated wrapper object containing all processed datasets.
        The returned type matches the value of ``open_as``.

    Notes
    -----
    • Each file is measured independently before concatenation.

    • Dropping variables after measurement can significantly reduce memory
      footprint when concatenating many files.

    • The returned object preserves the wrapper API (TrackArray/CropArray),
      allowing downstream methods such as ``.plot()``, ``.measure_signal()``,
      and other accessor functions to remain available.
    """
    import croparray as ca  # local import to avoid cycles on package import

    if not dims:
        raise ValueError("dims must be non-empty")

    measure_kwargs = {} if measure_kwargs is None else dict(measure_kwargs)

    # normalize labels
    if labels is None:
        labels = [None] * len(dims)
    else:
        labels = list(labels)
        if len(labels) != len(dims):
            raise ValueError(f"labels must have same length as dims ({len(labels)} vs {len(dims)})")

    def _is_leaf(node: Any) -> bool:
        return isinstance(node, (list, tuple)) and len(node) > 0 and all(isinstance(v, (str, Path)) for v in node)

    def _default_leaf_labels(files: Sequence[Union[str, Path]]) -> list[str]:
        return [Path(f).stem for f in files]

    prov = []
    leaf_names: list[str] = []
    leaf_notes: list[str] = []

    def _auto_use_zc(ds) -> bool:
        """
        Decide whether to use zc/z_pos as a selector.

        Scenarios handled automatically:
        - Scenario 1 (2D tracked, full z-stack): zc all same value (e.g. 0.0) → False
        - Scenario 2 (3D tracked, full z-stack): zc genuinely varies → True
        - Scenario 3 (2D tracked, single z plane): z size <= 1 → False
        """
        if ds is None or "zc" not in ds:
            return False

        # Single z plane — use_zc irrelevant, best_z just picks that plane
        if "z" in ds.dims and ds.sizes["z"] <= 1:
            return False

        vals = np.asarray(ds["zc"].values).ravel()
        finite = vals[np.isfinite(vals)]

        if len(finite) == 0:
            return False

        # 2D tracking sets zc=0.0 everywhere — constant value means no real z info
        if np.all(finite == finite[0]):
            return False

        # zc genuinely varies → 3D tracked, use it
        return True

    def _recurse(node: Any, level: int):
        dim = dims[level]
        lvl_labels = labels[level]

        # ---- leaf: node is list of files ----
        if _is_leaf(node):
            
            files = [str(f) for f in node]

            tas = []
            for fn in files:

                if open_as == "trackarray":
                    obj = ca.tools.open_as_trackarray(
                        fn,
                        as_object=True,
                        drop_vars=None,
                    )

                elif open_as == "croparray":
                    obj = ca.tools.open_croparray(fn, as_object=True)

                else:
                    raise ValueError(f"open_as must be 'croparray' or 'trackarray', got {open_as}")

                # ---- auto use_zc unless overridden ----
                mk = dict(measure_kwargs)
                if ("use_zc" not in mk) or (mk.get("use_zc", None) == "auto"):
                    mk["use_zc"] = _auto_use_zc(obj.ds)

                obj = obj.measure_signal(**mk)

                if drop_vars:
                    obj.ds = obj.ds.drop_vars(list(drop_vars), errors="ignore")

                tas.append(obj)

            for _ta in tas:
                ds0 = _ta.ds
                n0 = str(ds0.attrs.get("name", "")).strip()
                if n0:
                    leaf_names.append(n0)

                note0 = str(ds0.attrs.get("notes", "")).strip()
                if note0:
                    leaf_notes.append(note0)

            leaf_labels = lvl_labels if lvl_labels is not None else _default_leaf_labels(files)

            # ----------------------------------------------------------
            # Special handling for file-level concatenation along fov:
            # use integer fov indices for alignment, but preserve the
            # human-readable file stems in a separate coordinate fov_name.
            # ----------------------------------------------------------
            if dim == "fov":
                fov_names = list(leaf_labels)
                fov_index = list(range(len(tas) ))  # 0, 1, 2, 3, ...

                out = ca.concat(cas=tas, dim=dim, labels=fov_index, join=join)

                # Save original labels as a regular data variable on fov
                # (not a coordinate), so outer concatenations promote it to
                # (exp, fov), (rep, exp, fov), etc.
                if "fov_name" in out.ds.coords:
                    out.ds = out.ds.reset_coords("fov_name", drop=False)

                out.ds["fov_name"] = xr.DataArray(
                    np.asarray(fov_names, dtype=str),
                    dims=("fov",),
                )

                prov.append(
                    dict(
                        dim=dim,
                        level=level,
                        labels=list(fov_index),
                        fov_names=list(fov_names),
                        directories=sorted({str(Path(f).parent) for f in files}),
                        files=[Path(f).name for f in files],
                    )
                )
            else:
                out = ca.concat(cas=tas, dim=dim, labels=leaf_labels, join=join)

                prov.append(
                    dict(
                        dim=dim,
                        level=level,
                        labels=list(leaf_labels),
                        directories=sorted({str(Path(f).parent) for f in files}),
                        files=[Path(f).name for f in files],
                    )
                )

            return out

        # ---- non-leaf: list of subgroups ----
        if not isinstance(node, (list, tuple)):
            raise TypeError(f"Expected list/tuple at dim={dim} (level {level}), got {type(node)}")

        children = [_recurse(child, level + 1) for child in node]

        this_labels = lvl_labels
        if this_labels is None:
            this_labels = [f"{dim}{i}" for i in range(len(children))]
        else:
            if len(this_labels) != len(children):
                raise ValueError(f"labels for dim={dim} must match group length ({len(this_labels)} vs {len(children)})")

        out = ca.concat(cas=children, dim=dim, labels=this_labels, join=join)

        prov.append(dict(dim=dim, level=level, labels=list(this_labels), n_children=len(children)))
        return out

    ta = _recurse(groups, level=0)

    if attach_provenance:
        import json 
        ta.ds.attrs["provenance_json"] = json.dumps(
            dict(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                dims=list(dims),
                labels=[list(x) if x is not None else None for x in labels],
                measure_kwargs=measure_kwargs,
                drop_vars=drop_vars,
                join=join,
                hierarchy=prov,
            ),
            indent=2,
        )

    def _uniq_keep_order(xs: list[str]) -> list[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    # ---- name ----
    dim_str = "×".join(dims)  # e.g. rep×exp×fov
    n_files = 0
    # prov includes the leaf file lists; easiest is to count them
    for rec in prov:
        if rec.get("dim") == dims[-1] and "files" in rec:
            n_files += len(rec["files"])

    ln = _uniq_keep_order([x for x in leaf_names if x])
    if len(ln) == 0:
        ta.ds.attrs["name"] = f"concat[{n_files}] ({dim_str})"
    elif len(ln) == 1:
        ta.ds.attrs["name"] = f"{ln[0]} (concat[{n_files}] {dim_str})"
    else:
        ta.ds.attrs["name"] = f"{Path(ln[0]).stem} +{len(ln)-1} (concat[{n_files}] {dim_str})"

    # ---- notes ----
    notes_u = _uniq_keep_order([x for x in leaf_notes if x])
    if len(notes_u) == 0:
        # leave existing notes alone (or set to empty if you prefer)
        pass
    elif len(notes_u) == 1:
        ta.ds.attrs["notes"] = notes_u[0]
    else:
        ta.ds.attrs["notes"] = "\n\n---\n\n".join(notes_u)

    # ==========================================================
    # ULTRAMINIMAL CONCAT METADATA (for deterministic filenames)
    # ==========================================================
    try:
        # Collect leaf names from provenance
        leaf_files = []
        for rec in prov:
            if isinstance(rec, dict) and "files" in rec:
                leaf_files.extend(rec["files"])

        n_files = len(leaf_files)

        # Determine base name from first leaf dataset
        base_name = None
        if leaf_files:
            first_path = Path(leaf_files[0])
            base_name = first_path.stem

        # Fallback to existing name attr if needed
        if not base_name:
            base_name = str(ta.ds.attrs.get("name", "")).strip()

        # Clean base_name (filesystem friendly)
        import re
        base_name = (base_name or "croparray").strip()
        base_name = re.sub(r"\s+", "_", base_name)
        base_name = re.sub(r"[^A-Za-z0-9._-]+", "", base_name)
        base_name = base_name.strip("._-") or "croparray"

        # Store structured metadata
        import json
        ta.ds.attrs["concat_meta_json"] = json.dumps(
            {"base_name": base_name, "n_files": int(n_files), "dims": list(dims)},
            ensure_ascii=True,
            sort_keys=True,
        )


    except Exception:
        # Never let filename metadata break concat
        pass



    return ta


# ============================================================
# Append detailed dataset-builder docs to the public wrapper
# ============================================================
if _create_crop_array_dataset.__doc__:
    create_crop_array.__doc__ = (
        (create_crop_array.__doc__ or "")
        + "\n\n---\n\n"
        + _create_crop_array_dataset.__doc__
    )