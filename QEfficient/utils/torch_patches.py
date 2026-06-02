# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Runtime monkey patches for ONNX export compatibility.

Patches kept here:
  - TorchScript ONNX exporter (_setup_trace_module_map, _get_module_attributes,
    _jit_pass_onnx_track_scope_attributes): fix attribute-type mismatches in the
    legacy trace-based exporter (use_dynamo=False path).

Patches removed (upstreamed to PyTorch):
  - FunctionalTensorMode.__torch_dispatch__ tracker-entry KeyError
  - _verify_exported_program_signature repeated_subgraph buffer handling
  - ExportedProgram.named_buffers constants fallback
  - invoke_subgraph_placeholder kwargs forwarding
  - materialize_as_graph FunctionalTensor mode handling
  - InvokeSubgraphHOP.gen_schema GraphModule reuse
  - _translate_fx_graph / _convert_fx_arg_to_onnx_arg nested tensor constants
"""

import inspect
from contextlib import contextmanager

from contextlib import contextmanager

import torch
import torch.onnx.utils as onnx_utils
from torch import _C
from torch.onnx._internal.torchscript_exporter import utils as ts_utils

# Store original references before patching
_original_setup_trace_module_map = onnx_utils._setup_trace_module_map
_original_get_module_attributes = getattr(onnx_utils, "_get_module_attributes", None)
_original_track_scope_attrs = getattr(_C, "_jit_pass_onnx_track_scope_attributes", None)
_original_ts_setup_trace_module_map = ts_utils._setup_trace_module_map
_original_ts_get_module_attributes = getattr(ts_utils, "_get_module_attributes", None)

_PATCHES_ACTIVE = False
_MISSING_INSTANCE_ATTR = object()


def _setup_trace_module_map_patched(
    model,
    export_modules_as_functions,
):
    """Patched version of _setup_trace_module_map that fixes onnx_attrs type mismatch."""

    def __register_attribute_hook():
        attr_name = "_onnx_attrs"

        def _track_module_attributes_forward_pre_hook(module, input):
            setattr(module, attr_name, _get_module_attributes(module))

        def _track_module_attributes_forward_hook(module, input, output):
            tracing_state = _C._get_tracing_state()
            if not tracing_state:
                return
            graph = tracing_state.graph()
            onnx_attrs = {}
            if hasattr(module, attr_name):
                onnx_attrs = getattr(module, attr_name)
                delattr(module, attr_name)
            try:
                onnx_attrs = {}  # HACK: to reduce export time # TODO: study behaviour across models
                _C._jit_pass_onnx_track_scope_attributes(graph, onnx_attrs)
            except Exception:
                # Silently skip: scope-attribute tracking is best-effort and not required for export.
                pass

        for m in model.modules():
            m.register_forward_hook(_track_module_attributes_forward_hook)
            m.register_forward_pre_hook(_track_module_attributes_forward_pre_hook)

    def _unqualified_variable_name(qualified_name: str) -> str:
        name_atoms = qualified_name.split(".")
        for i, atom in reversed(list(enumerate(name_atoms))):
            if not atom.isnumeric():
                return ".".join(name_atoms[i:])
        return qualified_name

    trace_module_map = {
        _m: torch._C._jit_onnx_create_full_scope_name(torch.typename(type(_m)), _unqualified_variable_name(_n))
        for _n, _m in model.named_modules()
    }
    torch.jit._trace._trace_module_map = trace_module_map

    if isinstance(export_modules_as_functions, bool) and export_modules_as_functions:
        module_typenames = {torch.typename(type(module)) for module in trace_module_map}
    elif isinstance(export_modules_as_functions, set) and export_modules_as_functions:

        def _find_typename(v):
            if isinstance(v, type):
                return torch.typename(v)
            else:
                raise RuntimeError(
                    "Only type of the `nn.Module` should be passed in the set for argument `export_modules_as_functions`. "
                    f"Got `{type(v).__name__}`."
                )

        module_typenames = {_find_typename(v) for v in export_modules_as_functions}
    else:
        module_typenames = set()

    if module_typenames:
        __register_attribute_hook()

    return module_typenames


def _get_module_attributes(module):
    """Helper function to get module attributes safely."""
    import typing

    import torch.nn

    def _is_safe_value(value):
        if isinstance(value, (int, float, bool, str, torch.Tensor)) or value is None:
            return True
        if isinstance(value, (list, tuple)):
            return all(_is_safe_value(item) for item in value)
        return False

    annotations = typing.get_type_hints(type(module))
    base_m_annotations = typing.get_type_hints(torch.nn.Module)
    [annotations.pop(k, None) for k in base_m_annotations]

    attrs = {}
    for k in annotations:
        try:
            value = getattr(module, k)
            if _is_safe_value(value):
                attrs[k] = value
        except AttributeError:
            _C._jit_onnx_log(f"Skipping module attribute '{k}'")
            continue
    return attrs


def _track_scope_attributes_patched(graph, attrs):
    """Ensure scope attributes passed to ONNX are IValue-compatible."""
    safe_attrs = {}
    for key, value in attrs.items():
        if isinstance(value, (int, float, bool, str, torch.Tensor)) or value is None:
            safe_attrs[key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(item, (int, float, bool, str, torch.Tensor)) or item is None for item in value
        ):
            safe_attrs[key] = value
    return _original_track_scope_attrs(graph, safe_attrs)


def apply_torch_patches():
    """Apply monkey patches for ONNX export (TorchScript path)."""
    global _PATCHES_ACTIVE
    if _PATCHES_ACTIVE:
        return

    onnx_utils._setup_trace_module_map = _setup_trace_module_map_patched
    if hasattr(onnx_utils, "_get_module_attributes"):
        onnx_utils._get_module_attributes = _get_module_attributes
    ts_utils._setup_trace_module_map = _setup_trace_module_map_patched
    if hasattr(ts_utils, "_get_module_attributes"):
        ts_utils._get_module_attributes = _get_module_attributes
    if _original_track_scope_attrs is not None:
        _C._jit_pass_onnx_track_scope_attributes = _track_scope_attributes_patched

    _PATCHES_ACTIVE = True


def undo_torch_patches():
    """Undo monkey patches and restore original functions."""
    global _PATCHES_ACTIVE
    if not _PATCHES_ACTIVE:
        return

    onnx_utils._setup_trace_module_map = _original_setup_trace_module_map
    if _original_get_module_attributes:
        onnx_utils._get_module_attributes = _original_get_module_attributes
    ts_utils._setup_trace_module_map = _original_ts_setup_trace_module_map
    if _original_ts_get_module_attributes:
        ts_utils._get_module_attributes = _original_ts_get_module_attributes
    if _original_track_scope_attrs is not None:
        _C._jit_pass_onnx_track_scope_attributes = _original_track_scope_attrs

    _PATCHES_ACTIVE = False


@contextmanager
def temporarily_enable_nested_compile_regions(model, target_classes=None):
    """
    Wrap selected module ``forward`` methods with ``nested_compile_region``
    during export so repeated block functions are materialized by dynamo.
    """

    target_classes = tuple(target_classes) if target_classes else None
    patched_modules = []

    try:
        for module in model.modules():
            if target_classes and not isinstance(module, target_classes):
                continue

            bound_forward = getattr(module, "forward", None)
            if bound_forward is None:
                continue

            wrapped_forward = getattr(bound_forward, "__func__", bound_forward)
            if getattr(wrapped_forward, "__qualname__", "") == "mark_compile_region.<locals>.wrap.<locals>.inner":
                continue

            previous_forward = module.__dict__.get("forward", _MISSING_INSTANCE_ATTR)
            nested_forward = torch.compiler.nested_compile_region(wrapped_forward)
            setattr(module, "forward", nested_forward.__get__(module, type(module)))
            patched_modules.append((module, previous_forward))

        yield
    finally:
        for module, previous_forward in reversed(patched_modules):
            if previous_forward is _MISSING_INSTANCE_ATTR:
                delattr(module, "forward")
            else:
                setattr(module, "forward", previous_forward)


@contextmanager
def temporarily_disable_nested_compile_regions(model, target_classes=None):
    """
    Replace nested_compile_region-wrapped ``forward`` methods with their original
    underlying functions for the duration of plain dynamo export.
    """

    target_classes = tuple(target_classes) if target_classes else None
    patched_modules = []

    try:
        for module in model.modules():
            if target_classes and not isinstance(module, target_classes):
                continue

            bound_forward = getattr(module, "forward", None)
            if bound_forward is None:
                continue

            wrapped_forward = getattr(bound_forward, "__func__", bound_forward)
            if getattr(wrapped_forward, "__qualname__", "") != "mark_compile_region.<locals>.wrap.<locals>.inner":
                continue

            closure = getattr(wrapped_forward, "__closure__", None) or ()
            original_forward = next(
                (cell.cell_contents for cell in closure if inspect.isfunction(cell.cell_contents)),
                None,
            )
            if original_forward is None:
                continue

            previous_forward = module.__dict__.get("forward", _MISSING_INSTANCE_ATTR)
            setattr(module, "forward", original_forward.__get__(module, type(module)))
            patched_modules.append((module, previous_forward))

        yield
    finally:
        for module, previous_forward in reversed(patched_modules):
            if previous_forward is _MISSING_INSTANCE_ATTR:
                delattr(module, "forward")
            else:
                setattr(module, "forward", previous_forward)
