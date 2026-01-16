# scripts/generate_accessors.py
from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

OUT_PATH = Path("croparray/_accessors_generated.py")

# Map accessor class -> module that defines dataset-aware functions to expose via __all__
ACCESSOR_MODULES = {
    "CropArrayPlot": "croparray.plot",
    "CropArrayMeasure": "croparray.measure",
    "CropArrayDF": "croparray.dataframe",
    "CropArrayView": "croparray.napari_view",
    "CropArrayTrack": "croparray.tracking",
    "TrackArrayPlot": "croparray.trackarray.plot",
    "TrackArrayMeasure": "croparray.trackarray.measure",
    "TrackArrayView": "croparray.trackarray.napari_view",
    "TrackArrayDF": "croparray.trackarray.dataframe",
}

def _format_default(val: Any) -> str:
    # Avoid generating "nan" without an import
    if isinstance(val, float) and val != val:  # NaN check
        return 'float("nan")'
    return repr(val)

def _format_param(p: inspect.Parameter) -> str:
    s = p.name
    if p.kind == inspect.Parameter.VAR_POSITIONAL:
        s = "*" + s
    elif p.kind == inspect.Parameter.VAR_KEYWORD:
        s = "**" + s

    if p.default is not inspect._empty:
        s += f"={_format_default(p.default)}"
    return s

def _sig_without_first_arg(func) -> tuple[str, list[inspect.Parameter]]:
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    # Drop first arg (ds/ta_dataset/etc.) because accessor injects it
    if params:
        params = params[1:]
    rendered = ", ".join(_format_param(p) for p in params)
    return rendered, params

def _build_forward_args(params: list[inspect.Parameter]) -> str:
    """
    Build a safe forwarding argument string that preserves behavior:
      - positional-only and positional-or-keyword -> passed positionally
      - keyword-only -> passed as keyword
      - *args/**kwargs forwarded properly
    """
    parts: list[str] = []
    for p in params:
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            parts.append(f"*{p.name}")
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            parts.append(f"**{p.name}")
        elif p.kind == inspect.Parameter.KEYWORD_ONLY:
            parts.append(f"{p.name}={p.name}")
        else:
            # POSITIONAL_ONLY or POSITIONAL_OR_KEYWORD
            parts.append(p.name)
    return ", ".join(parts)

def generate() -> None:
    lines: list[str] = []
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from dataclasses import dataclass")
    lines.append("from typing import Any")
    lines.append("")
    lines.append("# This file is AUTO-GENERATED. Do not edit by hand.")
    lines.append("# Rebuild with: python scripts/generate_accessors.py")
    lines.append("")

    # Base accessor
    lines.append("@dataclass")
    lines.append("class _BaseAccessor:")
    lines.append("    parent: Any")
    lines.append("    @property")
    lines.append("    def ds(self):")
    lines.append("        return self.parent.ds")
    lines.append("")

    for cls_name, mod in ACCESSOR_MODULES.items():
        module = importlib.import_module(mod)
        func_names = list(getattr(module, "__all__", []))

        # Allow empty __all__ (still generate class, but with no methods)
        # Import impl functions as private names
        for fn in func_names:
            if not hasattr(module, fn):
                raise AttributeError(f"{mod} has no function {fn!r} (listed in __all__)")
            lines.append(f"from {mod} import {fn} as _impl_{cls_name}_{fn}")
        if func_names:
            lines.append("")

        lines.append("@dataclass")
        lines.append(f"class {cls_name}(_BaseAccessor):")
        lines.append('    """Generated accessor methods."""')

        # Emit methods (all indented inside class)
        doc_assignments: list[str] = []
        for fn in func_names:
            impl_name = f"_impl_{cls_name}_{fn}"
            func = getattr(module, fn)

            sig_str, params = _sig_without_first_arg(func)
            fwd = _build_forward_args(params)

            # Build call
            if fwd:
                call = f"return {impl_name}(self.ds, {fwd})"
            else:
                call = f"return {impl_name}(self.ds)"

            # Method signature: keep explicit (helps notebook tooltips)
            if sig_str:
                lines.append(f"    def {fn}(self, {sig_str}):")
            else:
                lines.append(f"    def {fn}(self):")
            lines.append(f"        {call}")
            lines.append("")

            # Defer docstring assignment until after the class ends (prevents indentation bugs)
            doc_assignments.append(f"{cls_name}.{fn}.__doc__ = {impl_name}.__doc__")

        lines.append("")  # end class block

        # Emit docstring assignments at top-level
        for stmt in doc_assignments:
            lines.append(stmt)
        if doc_assignments:
            lines.append("")
        lines.append("")

    # ---- Emit an installer that binds accessors as class properties (for autocomplete/tooltips) ----
    lines.append("")
    lines.append("")
    lines.append("def install_generated_accessors(CropArray, TrackArray):")
    lines.append('    """Attach generated accessors as @property on wrapper classes."""')
    lines.append("    # CropArray accessors")
    lines.append("    CropArray.plot    = property(lambda self, _A=CropArrayPlot: _A(self))")
    lines.append("    CropArray.measure = property(lambda self, _A=CropArrayMeasure: _A(self))")
    lines.append("    CropArray.view    = property(lambda self, _A=CropArrayView: _A(self))")
    lines.append("    CropArray.df      = property(lambda self, _A=CropArrayDF: _A(self))")
    lines.append("    CropArray.track   = property(lambda self, _A=CropArrayTrack: _A(self))")
    lines.append("")
    lines.append("    # TrackArray accessors")
    lines.append("    TrackArray.tplot    = property(lambda self, _A=TrackArrayPlot: _A(self))")
    lines.append("    TrackArray.tmeasure = property(lambda self, _A=TrackArrayMeasure: _A(self))")
    lines.append("    TrackArray.tview    = property(lambda self, _A=TrackArrayView: _A(self))")
    lines.append("    TrackArray.tdf      = property(lambda self, _A=TrackArrayDF: _A(self))")


    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")

if __name__ == "__main__":
    generate()
