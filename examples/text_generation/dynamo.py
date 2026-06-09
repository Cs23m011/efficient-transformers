#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _add_repo_root_to_path() -> None:
    # This file is at examples/text_generation/dynamo.py → repo root is parent(2)
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_add_repo_root_to_path()

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner
from scripts.memory_profiling import QEffMemoryProfiler


def _str_to_bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}. Use true/false.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run export/ORT/compile flow with timing and export RAM profiling.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="tiny-random/glm-5.1",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--use-dynamo",
        type=_str_to_bool,
        default=True,
        metavar="{true,false}",
        help="Whether to enable dynamo during export (true/false).",
    )
    parser.add_argument(
        "--use-onnx-subfunctions",
        type=_str_to_bool,
        default=True,
        metavar="{true,false}",
        help="Whether to enable ONNX subfunctions during export/compile (true/false).",
    )
    parser.add_argument(
        "--use-weight-free",
        type=_str_to_bool,
        default=True,
        metavar="{true,false}",
        help="Also run weight-free export (meta model, weights loaded from checkpoint at ORT time).",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=2,
        help="Override config.num_hidden_layers for quick experiments.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Dtype to set in config.torch_dtype.",
    )
    parser.add_argument("--prompt", type=str, default="My name is")
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--ctx-len", type=int, default=32)
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=Path(__file__).resolve().parent / "export_memory_profile.png",
        help="Output path for export RAM profile graph.",
    )
    return parser.parse_args()


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    return getattr(torch, dtype_name)


def _to_1d_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, (list, tuple)):
        if len(x) == 1 and isinstance(x[0], (np.ndarray, torch.Tensor, list, tuple)):
            return _to_1d_tensor(x[0])
        t = torch.tensor(x)
    else:
        t = torch.as_tensor(x)
    return t.reshape(-1)


def _convert_checkpoint_to_fp32(onnx_path: Path, weight_spec_path: Path) -> None:
    """Copy checkpoint tensors as local FP32 files and sync ONNX embedded metadata.
    Inlined from compare.py — required before QAIC compile for weight-free models."""
    import json
    import onnx as _onnx
    from safetensors.torch import load_file as _load_file, save_file as _save_file
    from QEfficient.exporter.weight_free import _default_weights_roots
    from QEfficient.exporter.weight_spec import (
        ExternalDataFile, load_weight_spec, save_weight_spec,
    )

    spec = load_weight_spec(weight_spec_path)
    export_dir = onnx_path.parent
    candidate_roots = _default_weights_roots(weight_spec_path, spec)

    if spec.files and all(
        not Path(f.path).is_absolute() and (export_dir / f.path).is_file()
        for f in spec.files
    ):
        print("[WF] Reusing existing local FP32 safetensors.")
        return

    needed: dict = {}
    for inp in spec.inputs:
        needed.setdefault(int(inp.location.file), set()).add(inp.location.key)

    old_to_new = {old: new for new, old in enumerate(sorted(needed.keys()))}
    new_files = []
    for old_idx in sorted(needed.keys()):
        ext_file = spec.files[old_idx]
        rel_path = Path(ext_file.path)
        abs_path = rel_path if rel_path.is_absolute() else None
        if abs_path is None:
            for root in candidate_roots:
                candidate = root / rel_path
                if candidate.exists():
                    abs_path = candidate
                    break
        if abs_path is None or not abs_path.exists():
            raise FileNotFoundError(f"Cannot resolve: {ext_file.path}")

        keys_needed = needed[old_idx]
        tensors = _load_file(str(abs_path))
        fp32 = {k: v.to(torch.float32) for k, v in tensors.items() if k in keys_needed}
        out_name = f"model_{old_to_new[old_idx]:04d}.safetensors"
        _save_file(fp32, str(export_dir / out_name))
        new_files.append(ExternalDataFile(path=out_name, format="safetensors"))
        print(f"  {abs_path.name} ({len(keys_needed)}/{len(tensors)} tensors) → {out_name}")

    for inp in spec.inputs:
        inp.location.file = old_to_new[int(inp.location.file)]
    spec.files = new_files
    save_weight_spec(weight_spec_path, spec)

    # Sync the updated paths into the com.qti.aisw.extdata metadata embedded
    # inside the ONNX — the QAIC compiler reads weights from this, not weight_spec.json.
    updated_json = json.dumps(
        json.loads(weight_spec_path.read_text()), separators=(",", ":"), sort_keys=True
    )
    onnx_model = _onnx.load(str(onnx_path), load_external_data=False)
    for entry in onnx_model.metadata_props:
        if entry.key == "com.qti.aisw.extdata":
            entry.value = updated_json
            break
    tmp = onnx_path.with_suffix(onnx_path.suffix + ".tmp")
    _onnx.save(onnx_model, str(tmp))
    tmp.replace(onnx_path)
    print("[WF] Synced embedded ONNX metadata → local FP32 paths.")


def _run_weight_free_flow(args, runner, qeff_model_regular, hf_tokens, pt_tokens, ort_tokens):
    """Run weight-free export + ORT + compile and compare results."""
    from accelerate import init_empty_weights
    from QEfficient.exporter.weight_free import load_weight_free_ort_inputs
    from QEfficient.exporter.weight_spec import resolve_weight_spec_path
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print("WEIGHT-FREE EXPORT FLOW")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    config.torch_dtype = torch.float32

    # Build meta (weight-free) model — no weights loaded into RAM
    print("\nBuilding meta model (init_empty_weights) ...")
    with init_empty_weights():
        meta_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")

    qeff_model_wf = QEFFAutoModelForCausalLM(
        meta_model,
        pretrained_model_name_or_path=args.model_name,
    )

    # Export
    wf_profile_output = args.profile_output.parent / "export_memory_profile_weightfree.png"
    profiler_wf = QEffMemoryProfiler(output_file=str(wf_profile_output), verbose=True)
    profiler_wf.start_monitoring()
    profiler_wf.mark_operation("Export")
    export_start = time.perf_counter()
    try:
        onnx_path_wf = qeff_model_wf.export(
            use_dynamo=args.use_dynamo,
            use_onnx_subfunctions=args.use_onnx_subfunctions,
            use_weight_free_export=True,
            offload_pt_weights=False,
        )
    finally:
        profiler_wf.stop_monitoring()

    export_elapsed = time.perf_counter() - export_start
    print(f"[WF-TIMING] export: {export_elapsed:.3f} seconds")
    print(f"[WF-MEMORY] export peak RSS: {profiler_wf.peak_rss:.2f} MB")
    print(f"[WF-ARTIFACT] onnx_path={onnx_path_wf}")
    print(profiler_wf.get_memory_report())
    profiler_wf.generate_memory_graph(str(wf_profile_output))
    print(f"[WF-MEMORY] export profile graph saved to: {wf_profile_output}")

    weight_spec_path = resolve_weight_spec_path(Path(onnx_path_wf))

    # Copy FP32 checkpoint tensors locally so the QAIC compiler can find them
    print("\n[WF] Converting checkpoint to local FP32 safetensors ...")
    fp32_start = time.perf_counter()
    _convert_checkpoint_to_fp32(Path(onnx_path_wf), weight_spec_path)
    print(f"[WF-TIMING] fp32 convert: {time.perf_counter() - fp32_start:.3f} seconds")

    # ORT inference — decode loop to generate tokens (not just one step of logits)
    print("\n--- Weight-Free ORT inference ---")
    session_wf = ort.InferenceSession(str(onnx_path_wf))
    ort_inputs = load_weight_free_ort_inputs(weight_spec_path, runner.input_handler.prepare_ort_inputs())
    ort_outputs = runner.run_ort_session(ort_inputs, session_wf)
    ort_outputs = runner.input_handler.update_ort_outputs(ort_outputs)

    wf_ort_ids = []
    for _ in range(1, runner.gen_len):
        wf_ort_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
        ort_inputs = runner.input_handler.update_ort_inputs(ort_inputs, ort_outputs)
        ort_inputs = load_weight_free_ort_inputs(weight_spec_path, ort_inputs)
        ort_outputs = runner.run_ort_session(ort_inputs, session_wf)
        ort_outputs = runner.input_handler.update_ort_outputs(ort_outputs)
    wf_ort_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))

    import numpy as _np
    wf_ort_tokens = _np.concatenate(wf_ort_ids, axis=1)
    wf_ort_text = tokenizer.batch_decode(wf_ort_tokens, skip_special_tokens=True)
    print(f"[WF-ORT] Completion: {wf_ort_text}")
    print(f"[WF-ORT] token ids: {wf_ort_tokens}")

    # Compile
    print("\n--- Weight-Free Compile ---")
    compile_start = time.perf_counter()
    try:
        qpc_path_wf = qeff_model_wf.compile(
            onnx_path=str(onnx_path_wf),
            prefill_seq_len=args.prompt_len,
            ctx_len=args.ctx_len,
            use_onnx_subfunctions=args.use_onnx_subfunctions,
            use_dynamo=args.use_dynamo,
            use_weight_free_export=True,
        )
        compile_elapsed = time.perf_counter() - compile_start
        print(f"[WF-TIMING] compile: {compile_elapsed:.3f} seconds")
        print(f"[WF-ARTIFACT] qpc_path={qpc_path_wf}")

        # AIC inference
        print("\n--- Weight-Free AIC inference ---")
        try:
            output_wf = qeff_model_wf.generate(
                prompts=[args.prompt], tokenizer=tokenizer, automation=True
            )
            wf_aic_tokens = _to_1d_tensor(output_wf.generated_ids)
            print(f"[WF-AIC] tokens: {output_wf.generated_ids}")
        except RuntimeError as e:
            print(f"[WF-AIC] Skipped: {e}")
            wf_aic_tokens = None
    except RuntimeError as e:
        print(f"[WF-COMPILE] Failed: {e}")
        wf_aic_tokens = None

    # Compare weight-free ORT vs regular ORT
    print("\n" + "=" * 60)
    print("WEIGHT-FREE vs REGULAR COMPARISON")
    print("=" * 60)
    wf_ort_t = _to_1d_tensor(wf_ort_tokens)
    reg_ort_t = _to_1d_tensor(ort_tokens)
    min_len = min(wf_ort_t.numel(), reg_ort_t.numel())
    match = torch.allclose(wf_ort_t[:min_len].float(), reg_ort_t[:min_len].float(), rtol=0, atol=0)
    print(f"WF-ORT == Regular-ORT (first {min_len} tokens): {'✓ MATCH' if match else '✗ MISMATCH'}")
    print(f"  Regular-ORT : {reg_ort_t[:min_len].tolist()}")
    print(f"  WF-ORT      : {wf_ort_t[:min_len].tolist()}")


def main() -> None:
    args = _parse_args()
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    config.torch_dtype = torch.float32
    print(config)

    runner = ApiRunner(
        batch_size=1,
        tokenizer=tokenizer,
        config=config,
        prompt=[args.prompt],
        prompt_len=args.prompt_len,
        ctx_len=args.ctx_len,
    )

    # ── HF PyTorch inference ───────────────────────────────────────────────────
    print("\n--- Original HF Model Outputs (Torch CPU) ---")
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)
    hf_tokens = runner.run_hf_model_on_pytorch(hf_model)
    print(hf_tokens)

    # ── QEff PyTorch inference ─────────────────────────────────────────────────
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(args.model_name, config=config)
    print("\n--- QEff Transformed HF Model Outputs (Torch CPU) ---")
    pt_tokens = runner.run_kv_model_on_pytorch(qeff_model.model)
    print(pt_tokens)

    # ── Regular export (dynamo + optional subfunctions) ────────────────────────
    profiler = QEffMemoryProfiler(output_file=str(args.profile_output), verbose=True)
    profiler.start_monitoring()
    profiler.mark_operation("Export")
    export_start = time.perf_counter()
    try:
        onnx_path = qeff_model.export(
            use_dynamo=args.use_dynamo,
            use_onnx_subfunctions=args.use_onnx_subfunctions,
        )
    finally:
        profiler.stop_monitoring()

    export_elapsed = time.perf_counter() - export_start
    print(f"[TIMING] qeff_model.export: {export_elapsed:.3f} seconds")
    print(f"[MEMORY] export peak RSS: {profiler.peak_rss:.2f} MB")
    print(f"[ARTIFACT] onnx_path={onnx_path}")
    print(profiler.get_memory_report())
    profiler.generate_memory_graph(str(args.profile_output))
    print(f"[MEMORY] export profile graph saved to: {args.profile_output}")

    # ── ORT inference ──────────────────────────────────────────────────────────
    print("\n--- QEff Transformed Onnx Model Outputs (OnnxRuntime CPU) ---")
    ort_tokens = runner.run_kv_model_on_ort(onnx_path)
    print(ort_tokens)

    # ── Compile ────────────────────────────────────────────────────────────────
    compile_start = time.perf_counter()
    qpc_path = qeff_model.compile(
        prefill_seq_len=args.prompt_len,
        ctx_len=args.ctx_len,
        use_onnx_subfunctions=args.use_onnx_subfunctions,
        use_dynamo=args.use_dynamo,
    )
    compile_elapsed = time.perf_counter() - compile_start
    print(f"[TIMING] qeff_model.compile: {compile_elapsed:.3f} seconds")
    print(f"[ARTIFACT] qpc_path={qpc_path}")
    print("compile done")

    # ── AIC inference ──────────────────────────────────────────────────────────
    print("QEff Transformed Onnx Model Outputs(AIC Backend)")
    aic_t = None
    try:
        output = qeff_model.generate(prompts=[args.prompt], tokenizer=tokenizer, automation=True)
        print(output)
        print(output.generated_ids)
        aic_t = _to_1d_tensor(output.generated_ids)
    except RuntimeError as e:
        print(f"[AIC] Skipped (no hardware): {e}")

    # ── Compare regular outputs ────────────────────────────────────────────────
    hf_t   = _to_1d_tensor(hf_tokens)
    pt_t   = _to_1d_tensor(pt_tokens)
    ort_t  = _to_1d_tensor(ort_tokens)

    lengths = {
        "hf_tokens": hf_t.numel(),
        "pt_tokens": pt_t.numel(),
        "ort_tokens": ort_t.numel(),
    }
    if aic_t is not None:
        lengths["aic_generated_ids"] = aic_t.numel()
    min_len = min(lengths.values()) if lengths else 0
    print(f"[COMPARE] original lengths: {lengths}")

    if min_len == 0:
        print("[COMPARE] Cannot compare tokens because at least one output is empty.")
    else:
        hf_trim  = hf_t[:min_len]
        pt_trim  = pt_t[:min_len]
        ort_trim = ort_t[:min_len]

        hf_pt_match  = torch.allclose(hf_trim.float(), pt_trim.float(),  rtol=0.0, atol=0.0)
        hf_ort_match = torch.allclose(hf_trim.float(), ort_trim.float(), rtol=0.0, atol=0.0)

        print(f"[COMPARE] trimmed length used: {min_len}")
        print("[COMPARE] trimmed outputs together:")
        print(f"  hf_tokens:  {hf_trim.tolist()}")
        print(f"  pt_tokens:  {pt_trim.tolist()}")
        print(f"  ort_tokens: {ort_trim.tolist()}")

        if aic_t is not None:
            aic_trim = aic_t[:min_len]
            hf_aic_match = torch.allclose(hf_trim.float(), aic_trim.float(), rtol=0.0, atol=0.0)
            all_match = hf_pt_match and hf_ort_match and hf_aic_match
            print(f"  aic_tokens: {aic_trim.tolist()}")
        else:
            all_match = hf_pt_match and hf_ort_match

        if all_match:
            print("[COMPARE] All outputs match.")
        else:
            print("[COMPARE] Outputs do NOT match.")

    # ── Weight-free flow (optional) ────────────────────────────────────────────
    if args.use_weight_free:
        _run_weight_free_flow(args, runner, qeff_model, hf_tokens, pt_tokens, ort_tokens)


if __name__ == "__main__":
    main()
