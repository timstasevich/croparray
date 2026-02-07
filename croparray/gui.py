from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, List

import numpy as np

# Optional dependencies for GUI
import napari
from magicgui import magicgui

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication
import time

try:
    from tifffile import imread as tif_imread
except Exception:
    tif_imread = None

__all__ = [
    "label_video_axes",
    "AxisLabelResult",
]


AXIS_CHOICES = ("(none)", "fov", "f", "z", "y", "x", "ch")


@dataclass
class AxisLabelResult:
    axes: Tuple[str, ...]          # e.g. ("f", "z", "y", "x", "ch")
    shape: Tuple[int, ...]         # original shape
    notes: str                     # any warnings / normalization notes


def _load_video_any(path: Union[str, Path]) -> np.ndarray:
    path = Path(path)
    if tif_imread is None:
        raise ImportError("tifffile is required to load TIFF stacks. Install tifffile.")
    return tif_imread(str(path))


def label_video_axes(
    video: Optional[np.ndarray] = None,
    *,
    path: Optional[Union[str, Path]] = None,
    layer_name: str = "video",
    block: bool = True,
) -> AxisLabelResult:
    """
    Open a napari GUI to label each dimension of a video array.

    You label each axis with dropdowns: fov, f, z, y, x, ch, or (none).
    Returns an AxisLabelResult with an `axes` tuple describing the CURRENT
    order of `video`.

    Parameters
    ----------
    video
        Numpy array already loaded. If None, `path` must be provided.
    path
        File to load (TIFF stack via tifffile).
    layer_name
        Napari layer name.
    block
        If True, blocks until you click "Accept" or close the viewer.

    Returns
    -------
    AxisLabelResult
    """
    if video is None:
        if path is None:
            raise ValueError("Provide either `video` or `path`.")
        video = _load_video_any(path)

    shape = tuple(int(s) for s in video.shape)
    ndim = video.ndim

    # Pre-fill heuristic: last two largest -> y/x, small -> ch, etc.
    # This is intentionally conservative (you’ll override via GUI).
    dim_sizes = list(enumerate(shape))
    sorted_by_size = sorted(dim_sizes, key=lambda t: t[1], reverse=True)

    guess = ["(none)"] * ndim
    if ndim >= 2:
        # assign y/x as two largest dims (order among them is ambiguous)
        guess[sorted_by_size[0][0]] = "y"
        guess[sorted_by_size[1][0]] = "x"

    # guess channel as a small dim (<= 4) if exists and not already used
    for i, s in dim_sizes:
        if s in (1, 2, 3, 4) and guess[i] == "(none)":
            guess[i] = "ch"
            break

    # guess z/time/fov among remaining dims (pure guess)
    remaining = [i for i in range(ndim) if guess[i] == "(none)"]
    # common patterns: (t,z,...) or (z,t,...) or include fov
    # If there is a dim around 10-80 -> z; around 20-5000 -> f; around 2-50 -> fov
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

    # --- napari viewer ---
    viewer = napari.Viewer()
    viewer.add_image(video, name=layer_name)

    accepted = {"done": False}
    canceled = {"done": False}
    result_holder = {"result": None}

    def _on_close(event=None):
        # Viewer was closed without Accept
        if not accepted["done"]:
            canceled["done"] = True

    # This works across napari versions:
    viewer.window._qt_window.destroyed.connect(lambda *args: _on_close())

    # Build a magicgui widget with one dropdown per dim
    # We create fields dynamically, but magicgui wants static signatures.
    # So we generate a function with *args-like parameters via closure:

    def _validate_and_make_axes(labels: List[str]) -> AxisLabelResult:
        # Normalize "(none)" -> None
        labels_norm = [None if lab == "(none)" else lab for lab in labels]

        # Required: y and x must be present exactly once
        for req in ("y", "x"):
            if labels_norm.count(req) != 1:
                raise ValueError(f"Axis '{req}' must be selected exactly once.")

        # Optional: ch/z/f/fov can be absent or present once
        for opt in ("ch", "z", "f", "fov"):
            if labels_norm.count(opt) > 1:
                raise ValueError(f"Axis '{opt}' can be selected at most once.")

        # If ch absent, we assume ch=1 later in standardization step
        # Same for z/f/fov depending on your standardizer’s behavior.
        axes = tuple(lab for lab in labels_norm if lab is not None)

        notes = []
        if labels_norm.count("ch") == 0:
            notes.append("No channel axis selected; you’ll probably want to treat ch as singleton.")
        if labels_norm.count("z") == 0:
            notes.append("No z axis selected; z will be treated as singleton.")
        if labels_norm.count("f") == 0:
            notes.append("No time/frame axis selected; f will be treated as singleton.")
        if labels_norm.count("fov") == 0:
            notes.append("No fov axis selected; fov will be treated as singleton.")

        return AxisLabelResult(axes=axes, shape=shape, notes=" ".join(notes).strip())

    # We’ll implement dropdowns as separate widgets and a single accept button.
    dropdowns = []
    for d in range(ndim):
        w = magicgui(
            lambda axis="(none)": axis,
            axis={"choices": AXIS_CHOICES, "value": guess[d]},
            call_button=False,
        )
        w.name = f"dim{d}_size{shape[d]}"
        dropdowns.append(w)

    @magicgui(call_button="Accept")
    def accept():
        labels = [w.axis.value for w in dropdowns]
        try:
            res = _validate_and_make_axes(labels)
        except Exception as e:
            # show error in napari notifications
            viewer.status = f"Axis label error: {e}"
            napari.utils.notifications.show_error(str(e))
            return

        result_holder["result"] = res
        accepted["done"] = True

        # IMPORTANT: close viewer AFTER magicgui finishes its internal button bookkeeping
        QTimer.singleShot(0, viewer.close)

    @magicgui(call_button="Print current labels")
    def print_labels():
        labels = [w.axis.value for w in dropdowns]
        msg = " | ".join([f"dim{d}={shape[d]}→{lab}" for d, lab in enumerate(labels)])
        napari.utils.notifications.show_info(msg)

    # Add widgets to viewer
    for d, w in enumerate(dropdowns):
        w.native.setToolTip(f"Label dimension {d} (size {shape[d]})")
        viewer.window.add_dock_widget(w, area="right")
    viewer.window.add_dock_widget(print_labels, area="right")
    viewer.window.add_dock_widget(accept, area="right")

    if block:
        viewer.show()

        app = QApplication.instance()
        if app is None:
            raise RuntimeError("No Qt application instance found.")

        while not accepted["done"] and not canceled["done"]:
            app.processEvents()
            time.sleep(0.01)  # prevent CPU spin

    if accepted["done"] and result_holder["result"] is not None:
        return result_holder["result"]

    raise RuntimeError("Axis labeling canceled (viewer closed without Accept).")

