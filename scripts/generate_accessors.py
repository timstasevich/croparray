# scripts/generate_accessors.py
from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, NamedTuple

OUT_PATH = Path("croparray/_accessors_generated.py")


# --- Optional dependency stubs (generation should not require heavy GUI deps) ---
def _ensure_stub(name: str) -> None:
    try:
        importlib.import_module(name)
    except Exception:
        import sys
        import types

        sys.modules[name] = types.ModuleType(name)


_ensure_stub("napari")


def _ensure_magicgui_qtpy_stubs() -> None:
    import sys
    import types

    # magicgui.widgets
    if "magicgui" not in sys.modules:
        sys.modules["magicgui"] = types.ModuleType("magicgui")
    if "magicgui.widgets" not in sys.modules:
        widgets = types.ModuleType("magicgui.widgets")
        # Define dummy widget classes used at import-time
        for _name in ["Container", "ComboBox", "FileEdit", "Label", "PushButton", "CheckBox"]:
            setattr(widgets, _name, type(_name, (), {}))
        sys.modules["magicgui.widgets"] = widgets
        setattr(sys.modules["magicgui"], "widgets", widgets)

    # qtpy.QtWidgets
    if "qtpy" not in sys.modules:
        sys.modules["qtpy"] = types.ModuleType("qtpy")
    if "qtpy.QtWidgets" not in sys.modules:
        qtwidgets = types.ModuleType("qtpy.QtWidgets")
        setattr(qtwidgets, "QApplication", type("QApplication", (), {}))
        sys.modules["qtpy.QtWidgets"] = qtwidgets
        setattr(sys.modules["qtpy"], "QtWidgets", qtwidgets)


_ensure_magicgui_qtpy_stubs()


class AccessorSpec(NamedTuple):
    """Spec for generating an accessor class.

    modules: ordered list/tuple of module names.
      - earlier modules act as "base"
      - later modules override functions of the same name
    """

    # Name of the generated accessor class
    cls_name: str
    # Modules containing functions to expose (via __all__ if present, else public callables)
    modules: tuple[str, ...]
    # Whether accessor injects self.ds as the first argument
    bind_ds: bool
    # Which wrapper class to install on: "CropArray" or "TrackArray"
    target: str
    # Property name on the wrapper (e.g., "plot", "ops", "napari")
    prop: str


# ---- Configure accessor coverage here ----
# Within each module, add functions to __all__ (recommended). If __all__ is absent,
# we fall back to exporting all public callables defined in that module.
ACCESSOR_SPECS: list[AccessorSpec] = [
    # CropArray: ds-first (bind_ds=True)
    AccessorSpec("CropArrayPlot", ("croparray.plot",), True, "CropArray", "plot"),
    AccessorSpec("CropArrayMeasure", ("croparray.measure",), True, "CropArray", "measure"),
    AccessorSpec("CropArrayDF", ("croparray.dataframe",), True, "CropArray", "df"),
    AccessorSpec("CropArrayView", ("croparray.napari_view",), True, "CropArray", "view"),
    AccessorSpec("CropArrayTrack", ("croparray.tracking",), True, "CropArray", "track"),
    AccessorSpec("CropArrayOps", ("croparray.crop_ops.apply",), True, "CropArray", "ops"),
    AccessorSpec("CropArrayNapari", ("croparray.napari_view",), True, "CropArray", "napari"),

    # CropArray: non-ds-first (bind_ds=False)
    AccessorSpec("CropArrayBuild", ("croparray.build",), False, "CropArray", "build"),
    AccessorSpec("CropArrayIO", ("croparray.io",), False, "CropArray", "io"),

    # TrackArray: ds-first (bind_ds=True)
    # TrackArray accessors inherit CropArray APIs by default, and allow trackarray modules
    # to add/override behavior.
    AccessorSpec(
        "TrackArrayPlot",
        ("croparray.plot", "croparray.trackarray.plot"),
        True,
        "TrackArray",
        "tplot",
    ),
    AccessorSpec(
        "TrackArrayMeasure",
        ("croparray.measure", "croparray.trackarray.measure"),
        True,
        "TrackArray",
        "tmeasure",
    ),
    AccessorSpec(
        "TrackArrayView",
        ("croparray.napari_view", "croparray.trackarray.napari_view"),
        True,
        "TrackArray",
        "tview",
    ),
    AccessorSpec(
        "TrackArrayDF",
        ("croparray.dataframe", "croparray.trackarray.dataframe"),
        True,
        "TrackArray",
        "tdf",
    ),
    AccessorSpec("TrackArrayOps", ("croparray.crop_ops.apply",), True, "TrackArray", "ops"),
    AccessorSpec(
        "TrackArrayNapari",
        ("croparray.napari_view", "croparray.trackarray.napari_view"),
        True,
        "TrackArray",
        "napari",
    ),

    # TrackArray: non-ds-first (bind_ds=False) — add when you have a stable API
    # AccessorSpec("TrackArrayBuild", ("croparray.build", "croparray.trackarray.build"), False, "TrackArray", "build"),
]


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


def _sig_params(func) -> tuple[str, list[inspect.Parameter]]:
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    rendered = ", ".join(_format_param(p) for p in params)
    return rendered, params


def _sig_without_first_arg(func) -> tuple[str, list[inspect.Parameter]]:
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if params:
        params = params[1:]
    rendered = ", ".join(_format_param(p) for p in params)
    return rendered, params


def _build_forward_args(params: list[inspect.Parameter]) -> str:
    """Build a safe forwarding argument string."""
    parts: list[str] = []
    for p in params:
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            parts.append(f"*{p.name}")
        elif p.kind == inspect.Parameter.VAR_KEYWORD:
            parts.append(f"**{p.name}")
        elif p.kind == inspect.Parameter.KEYWORD_ONLY:
            parts.append(f"{p.name}={p.name}")
        else:
            parts.append(p.name)
    return ", ".join(parts)


def _exported_function_names(module) -> list[str]:
    """Return exported names for a module.

    Prefer __all__ if present. If __all__ is absent/empty, export public callables
    defined in that module.
    """
    names = list(getattr(module, "__all__", []))
    if names:
        return names

    out: list[str] = []
    for name, obj in module.__dict__.items():
        if name.startswith("_"):
            continue
        if callable(obj) and getattr(obj, "__module__", None) == module.__name__:
            out.append(name)
    return sorted(out)


def _merged_functions(modnames: tuple[str, ...]) -> tuple[list[str], dict[str, tuple[str, Any]]]:
    """Merge exported callables across modules.

    Returns:
      ordered_names: unique function names in first-seen order across modules
      winners: mapping fn_name -> (winning_module_name, function_object)

    Later modules override earlier ones for the same function name.
    """
    if not modnames:
        return [], {}

    ordered: list[str] = []
    seen: set[str] = set()
    winners: dict[str, tuple[str, Any]] = {}

    for modname in modnames:
        module = importlib.import_module(modname)
        names = _exported_function_names(module)

        # preserve first-seen order
        for fn in names:
            if fn not in seen:
                seen.add(fn)
                ordered.append(fn)

        # update winners (later modules override)
        for fn in names:
            if not hasattr(module, fn):
                raise AttributeError(f"{modname} has no attribute {fn!r} (exported)")
            winners[fn] = (modname, getattr(module, fn))

    return ordered, winners


def generate() -> None:
    lines: list[str] = []
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from dataclasses import dataclass")
    lines.append("from typing import Any")
    lines.append("import inspect")
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

    # Generate each accessor class
    for spec in ACCESSOR_SPECS:
        func_names, winners = _merged_functions(spec.modules)

        # Import winning impl functions as private names
        for fn in func_names:
            modname, _func = winners[fn]
            lines.append(f"from {modname} import {fn} as _impl_{spec.cls_name}_{fn}")
        if func_names:
            lines.append("")

        lines.append("@dataclass")
        lines.append(f"class {spec.cls_name}(_BaseAccessor):")
        lines.append('    """Generated accessor methods."""')

        doc_assignments: list[str] = []
        for fn in func_names:
            impl_name = f"_impl_{spec.cls_name}_{fn}"
            _modname, func = winners[fn]

            if spec.bind_ds:
                sig_str, params = _sig_without_first_arg(func)
                fwd = _build_forward_args(params)
                call = f"return {impl_name}(self.ds{', ' + fwd if fwd else ''})"
            else:
                sig_str, params = _sig_params(func)
                fwd = _build_forward_args(params)
                call = f"return {impl_name}({fwd})" if fwd else f"return {impl_name}()"

            # Method signature: explicit (helps notebook tooltips)
            if sig_str:
                lines.append(f"    def {fn}(self, {sig_str}):")
            else:
                lines.append(f"    def {fn}(self):")

            # Put the docstring inside the method so VS Code/Pylance can see it
            doc = inspect.getdoc(func) or ""
            doc = doc.replace('"""', r"\"\"\"")
            if doc:
                lines.append(f'        """{doc}"""')

            lines.append(f"        {call}")
            lines.append("")

            # Ensure runtime metadata is perfect for IPython/Jupyter
            doc_assignments.append(f"{spec.cls_name}.{fn}.__doc__ = {impl_name}.__doc__")
            doc_assignments.append(f"{spec.cls_name}.{fn}.__wrapped__ = {impl_name}")

            if spec.bind_ds:
                doc_assignments.append(
                    f"{spec.cls_name}.{fn}.__signature__ = inspect.Signature("  # noqa: E501
                    f"parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + "
                    f"list(inspect.signature({impl_name}).parameters.values())[1:]"  # drop ds
                    f")"
                )
            else:
                doc_assignments.append(
                    f"{spec.cls_name}.{fn}.__signature__ = inspect.Signature("  # noqa: E501
                    f"parameters=[inspect.Parameter('self', inspect.Parameter.POSITIONAL_OR_KEYWORD)] + "
                    f"list(inspect.signature({impl_name}).parameters.values())"
                    f")"
                )

        lines.append("")  # end class block

        for stmt in doc_assignments:
            lines.append(stmt)
        if doc_assignments:
            lines.append("")
        lines.append("")

    # Installer
    lines.append("")
    lines.append("def install_generated_accessors(CropArray, TrackArray):")
    lines.append('    """Attach generated accessors as @property on wrapper classes."""')
    lines.append("    # CropArray accessors")
    for spec in ACCESSOR_SPECS:
        if spec.target == "CropArray":
            lines.append(f"    CropArray.{spec.prop} = property(lambda self, _A={spec.cls_name}: _A(self))")
    lines.append("")
    lines.append("    # TrackArray accessors")
    for spec in ACCESSOR_SPECS:
        if spec.target == "TrackArray":
            lines.append(f"    TrackArray.{spec.prop} = property(lambda self, _A={spec.cls_name}: _A(self))")
    lines.append("")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    generate()
