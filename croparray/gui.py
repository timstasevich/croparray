from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import json
import time

import numpy as np

from magicgui.widgets import (
    Container,
    ComboBox,
    FileEdit,
    Label,
    PushButton,
    CheckBox,
)
from qtpy.QtWidgets import QApplication

__all__ = ["define_video_axes"]



# If you already have these in gui.py, keep yours and delete these duplicates.
AXIS_CHOICES = ("(none)", "fov", "f", "z", "y", "x", "ch")


@dataclass(frozen=True)
class AxisLabelResult:
    axes: Tuple[str, ...]   # e.g. ("f", "z", "y", "x", "ch") in CURRENT order, excluding "(none)"
    shape: Tuple[int, ...]
    notes: str = ""


def _guess_axes_from_shape(shape: Tuple[int, ...]) -> List[str]:
    """Conservative guess: largest 2 -> y/x, small -> ch, then try z/fov/f."""
    ndim = len(shape)
    dim_sizes = list(enumerate(shape))
    sorted_by_size = sorted(dim_sizes, key=lambda t: t[1], reverse=True)

    guess = ["(none)"] * ndim
    if ndim >= 2:
        guess[sorted_by_size[0][0]] = "y"
        guess[sorted_by_size[1][0]] = "x"

    for i, s in dim_sizes:
        if s in (1, 2, 3, 4) and guess[i] == "(none)":
            guess[i] = "ch"
            break

    remaining = [i for i in range(ndim) if guess[i] == "(none)"]
    for i in remaining:
        s = shape[i]
        if 5 <= s <= 80:
            guess[i] = "z"
            break

    remaining = [i for i in range(ndim) if guess[i] == "(none)"]
    for i in remaining:
        s = shape[i]
        if 2 <= s <= 50:
            guess[i] = "fov"
            break

    remaining = [i for i in range(ndim) if guess[i] == "(none)"]
    for i in remaining:
        guess[i] = "f"
        break

    return guess


def _validate_axes_labels(labels: List[str]) -> AxisLabelResult:
    """
    Validate dropdown labels and construct AxisLabelResult.axes (excluding "(none)").
    Requires y and x exactly once; others at most once.
    """
    labels_norm = [None if lab == "(none)" else lab for lab in labels]

    for req in ("y", "x"):
        if labels_norm.count(req) != 1:
            raise ValueError(f"Axis '{req}' must be selected exactly once.")

    for opt in ("ch", "z", "f", "fov"):
        if labels_norm.count(opt) > 1:
            raise ValueError(f"Axis '{opt}' can be selected at most once.")

    axes = tuple(lab for lab in labels_norm if lab is not None)
    notes = []
    if labels_norm.count("ch") == 0:
        notes.append("No channel axis selected; ch will be treated as singleton.")
    if labels_norm.count("z") == 0:
        notes.append("No z axis selected; z will be treated as singleton.")
    if labels_norm.count("f") == 0:
        notes.append("No frame/time axis selected; f will be treated as singleton.")
    if labels_norm.count("fov") == 0:
        notes.append("No fov axis selected; fov will be treated as singleton.")

    # shape is filled by caller; placeholder here
    return AxisLabelResult(axes=axes, shape=tuple(), notes=" ".join(notes).strip())


def _default_axes_json_path(video_path: Path) -> Path:
    # Example: movie.tif -> movie.croparray_axes.json
    return video_path.with_suffix(video_path.suffix + ".croparray_axes.json")


def _read_video_shape_fast(path: Path) -> Tuple[int, ...]:
    """
    Attempt to read shape without loading the full array.
    Uses tifffile if available; falls back to loading.
    """
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            # Most stacks: first series is correct
            shape = tf.series[0].shape
            return tuple(int(s) for s in shape)
    except Exception:
        # Fallback: load array (may be heavy)
        from tifffile import imread
        arr = imread(str(path))
        return tuple(int(s) for s in arr.shape)


def _load_axes_json(json_path: Path) -> Optional[dict]:
    if not json_path.exists():
        return None
    try:
        with json_path.open("r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_axes_json(json_path: Path, payload: dict) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)


def define_video_axes(
    *,
    path: Union[str, Path],
    axes_json: Optional[Union[str, Path]] = None,
    block: bool = True,
) -> AxisLabelResult:
    """
    Lightweight axis labeler (no napari):

    - shows the video path + detected shape
    - provides one dropdown per dimension
    - loads a saved json (if present) to pre-populate
    - Save button writes json next to the video by default
    - Accept returns AxisLabelResult usable by build.standardize_video_axes

    Returns AxisLabelResult where `.axes` describes the CURRENT order of the raw video.
    """
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    shape = _read_video_shape_fast(video_path)
    ndim = len(shape)

    json_path = Path(axes_json) if axes_json is not None else _default_axes_json_path(video_path)

    # defaults
    guess = _guess_axes_from_shape(shape)

    # load previous if exists and shape matches (or user allows reuse)
    saved = _load_axes_json(json_path)
    saved_labels = None
    if saved and isinstance(saved.get("labels"), list):
        if tuple(saved.get("shape", ())) == tuple(shape):
            saved_labels = saved["labels"]

    # state holders
    accepted = {"done": False}
    canceled = {"done": False}
    result_holder = {"result": None}

    # ---- widgets ----
    w_path = FileEdit(value=str(video_path), mode="r")
    w_shape = Label(value=f"shape = {shape}  (ndim={ndim})")
    w_json = FileEdit(value=str(json_path), mode="w")
    w_reuse_even_if_shape_diff = CheckBox(
        value=False,
        text="Allow loading saved labels even if shape differs (not recommended)",
    )

    dropdowns: List[ComboBox] = []
    for d in range(ndim):
        init = guess[d]
        if saved_labels and d < len(saved_labels):
            init = saved_labels[d]
        cb = ComboBox(choices=AXIS_CHOICES, value=init, label=f"dim {d} (size {shape[d]})")
        dropdowns.append(cb)

    msg = Label(value="")
    msg.native.setWordWrap(True)
    msg.native.setMaximumWidth(500)  # adjust as you like


    btn_load = PushButton(text="Load saved labels")
    btn_guess = PushButton(text="Reset to guess")
    btn_save = PushButton(text="Save labels")
    btn_accept = PushButton(text="Accept")
    btn_cancel = PushButton(text="Cancel")

    def _set_message(text: str, error: bool = False) -> None:
        msg.value = ("❌ " if error else "✅ ") + text

    def _current_labels() -> List[str]:
        return [cb.value for cb in dropdowns]

    def _apply_labels(labels: List[str]) -> None:
        for cb, lab in zip(dropdowns, labels):
            if lab in AXIS_CHOICES:
                cb.value = lab

    def _load_clicked():
        jp = Path(w_json.value)
        payload = _load_axes_json(jp)
        if not payload:
            _set_message(f"No readable labels file: {jp}", error=True)
            return
        labels = payload.get("labels", None)
        shp = tuple(payload.get("shape", ()))
        if not isinstance(labels, list):
            _set_message(f"Malformed labels in {jp}", error=True)
            return
        if shp != shape and not w_reuse_even_if_shape_diff.value:
            _set_message(
                f"Saved shape {shp} != current shape {shape}. "
                "Enable reuse checkbox to force load.",
                error=True,
            )
            return
        _apply_labels(labels)
        _set_message(f"Loaded labels from {jp}")

    def _guess_clicked():
        _apply_labels(_guess_axes_from_shape(shape))
        _set_message("Reset to heuristic guess")

    def _save_clicked():
        jp = Path(w_json.value)
        labels = _current_labels()
        payload = {
            "version": 1,
            "video": str(video_path),
            "shape": list(shape),
            "labels": labels,                 # includes "(none)" entries, length == ndim
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_axes_json(jp, payload)
        _set_message(f"Saved labels to {jp}")

    def _accept_clicked():
        labels = _current_labels()
        try:
            tmp = _validate_axes_labels(labels)
        except Exception as e:
            _set_message(str(e), error=True)
            return

        # fill in shape
        result_holder["result"] = AxisLabelResult(axes=tmp.axes, shape=shape, notes=tmp.notes)
        accepted["done"] = True
        # close container window (handled below)
        container.close()

    def _cancel_clicked():
        canceled["done"] = True
        container.close()

    btn_load.changed.connect(lambda _: _load_clicked())
    btn_guess.changed.connect(lambda _: _guess_clicked())
    btn_save.changed.connect(lambda _: _save_clicked())
    btn_accept.changed.connect(lambda _: _accept_clicked())
    btn_cancel.changed.connect(lambda _: _cancel_clicked())

    container = Container(
        widgets=[
            Label(value="croparray: label video axes"),
            Label(value="Video file:"),
            w_path,
            w_shape,
            Label(value="Axes labels file (json):"),
            w_json,
            w_reuse_even_if_shape_diff,
            Label(value="Assign each dimension:"),
            *dropdowns,
            Container(widgets=[btn_load, btn_guess, btn_save], layout="horizontal"),
            Container(widgets=[btn_accept, btn_cancel], layout="horizontal"),
            msg,
        ],
        layout="vertical",
    )

    # show window
    container.show()

    if block:
        app = QApplication.instance()
        if app is None:
            raise RuntimeError("No Qt application instance found.")
        while not accepted["done"] and not canceled["done"]:
            app.processEvents()
            time.sleep(0.01)

    if accepted["done"] and result_holder["result"] is not None:
        return result_holder["result"]

    raise RuntimeError("Axis labeling canceled.")
