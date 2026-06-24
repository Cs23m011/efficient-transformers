# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
from transformers import AutoModel as HFAutoModel
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModel
from QEfficient.exporter.weight_free import _default_weights_roots, load_weight_free_ort_inputs
from QEfficient.exporter.weight_spec import (
    ExternalDataFile,
    load_weight_spec,
    resolve_weight_spec_path,
    save_weight_spec,
)
from QEfficient.transformers.embeddings.embedding_utils import POOLING_MAP

sys.path.insert(0, str(Path(__file__).parents[2]))
from scripts.memory_profiling.profiler import QEffMemoryProfiler


# ─── helpers copied from examples/text_generation/compare.py ─────────────────

def _sync_embedded_extdata(onnx_path: Path, weight_spec_path: Path) -> None:
    updated_json = json.dumps(
        json.loads(weight_spec_path.read_text()), separators=(",", ":"), sort_keys=True
    )
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    for entry in onnx_model.metadata_props:
        if entry.key == "com.qti.aisw.extdata":
            entry.value = updated_json
            break
    tmp = onnx_path.with_suffix(onnx_path.suffix + ".tmp")
    onnx.save(onnx_model, str(tmp))
    tmp.replace(onnx_path)


def convert_checkpoint_to_fp32(onnx_path: Path, weight_spec_path: Path) -> None:
    """
    Extract only the tensors referenced by weight_spec inputs, cast to FP32,
    save next to the ONNX, and update weight_spec.json to point there.
    """
    spec = load_weight_spec(weight_spec_path)
    export_dir = onnx_path.parent
    candidate_roots = _default_weights_roots(weight_spec_path, spec)

    if spec.files and all(
        not Path(f.path).is_absolute() and (export_dir / f.path).is_file()
        for f in spec.files
    ):
        print("  Reusing existing local FP32 safetensors.")
        return

    needed: dict[int, set[str]] = {}
    for inp in spec.inputs:
        file_idx = int(inp.location.file)
        needed.setdefault(file_idx, set()).add(inp.location.key)

    sorted_old_idxs = sorted(needed.keys())
    old_to_new = {old: new for new, old in enumerate(sorted_old_idxs)}

    new_files = []
    for old_idx in sorted_old_idxs:
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
            raise FileNotFoundError(f"Cannot resolve external data file: {ext_file.path}")

        keys_needed = needed[old_idx]
        with safe_open(str(abs_path), framework="pt") as f:
            already_fp32 = all(f.get_slice(k).get_dtype() == "F32" for k in keys_needed)

        if already_fp32:
            new_files.append(ExternalDataFile(path=str(abs_path), format="safetensors"))
            print(f"  {abs_path.name}  ({len(keys_needed)} tensors)  -> referenced in place (already fp32)")
        else:
            tensors = load_file(str(abs_path))
            fp32_tensors = {k: v.to(torch.float32) for k, v in tensors.items() if k in keys_needed}
            out_name = f"model_{old_to_new[old_idx]:04d}.safetensors"
            save_file(fp32_tensors, str(export_dir / out_name))
            new_files.append(ExternalDataFile(path=out_name, format="safetensors"))
            print(f"  {abs_path.name}  ({len(keys_needed)}/{len(tensors)} tensors)  -> {out_name}  (fp32)")

    for inp in spec.inputs:
        inp.location.file = old_to_new[int(inp.location.file)]

    spec.files = new_files
    save_weight_spec(weight_spec_path, spec)
    _sync_embedded_extdata(onnx_path, weight_spec_path)


# ─── print helpers ────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_comparison(label, mad, max_diff, threshold):
    status = "PASS" if mad <= threshold else "FAIL"
    print(f"  {label}")
    print(f"    MAD       : {mad:.8f}  (threshold: {threshold})")
    print(f"    Max diff  : {max_diff:.8f}")
    print(f"    Status    : [{status}]")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare embedding outputs: HF PyTorch vs QEff PyTorch vs ONNX vs AI100"
    )
    parser.add_argument("--model-name", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--sentences", type=str, default="This is a test sentence")
    parser.add_argument(
        "--pooling",
        type=str,
        default="cls",
        choices=["mean", "cls", "avg", "none"],
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument(
        "--use-dynamo",
        action="store_true",
        help="Use torch.onnx dynamo export (classic path)",
    )
    parser.add_argument(
        "--use-subfunctions",
        action="store_true",
        help="Export with ONNX subfunctions",
    )
    parser.add_argument(
        "--weight-free",
        action="store_true",
        help="Weight-free export: build meta model, export without weights, "
             "then materialize FP32 safetensors. Forces --use-dynamo.",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Directory to save the exported ONNX (default: auto)",
    )
    parser.add_argument(
        "--skip-ai100",
        action="store_true",
        help="Skip AI100 compilation and inference",
    )
    parser.add_argument("--profile", action="store_true", help="Enable memory profiling")
    parser.add_argument(
        "--profile-output",
        type=str,
        default="embedding_memory_profile.png",
    )
    args = parser.parse_args()

    pooling = None if args.pooling == "none" else args.pooling

    # weight-free always requires dynamo
    if args.weight_free:
        args.use_dynamo = True

    print_section("Configuration")
    print(f"  Model       : {args.model_name}")
    print(f"  Input       : {args.sentences}")
    print(f"  Pooling     : {pooling or 'none (raw token embeddings)'}")
    print(f"  Seq len     : {args.seq_len}")
    print(f"  Dynamo      : {args.use_dynamo}")
    print(f"  Subfunctions: {args.use_subfunctions}")
    print(f"  Weight-free : {args.weight_free}")

    # ── profiler ──────────────────────────────────────────────────────────────
    profiler = None
    if args.profile:
        profiler = QEffMemoryProfiler(
            sampling_interval=0.05,
            output_file=args.profile_output,
            verbose=False,
        )
        profiler.start_monitoring()
        print(f"\n  Profiling   : enabled → {args.profile_output}")

    # ── tokenize ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    inputs = tokenizer(args.sentences, return_tensors="pt")
    print(f"\n  Token count : {inputs['input_ids'].shape[1]}")

    # ── Stage 1: HF PyTorch baseline (always full weights) ────────────────────
    print_section("Stage 1: HF PyTorch baseline")
    if profiler:
        profiler.mark_operation("Model Loading")

    hf_model = HFAutoModel.from_pretrained(args.model_name, attn_implementation="eager").eval()

    with torch.no_grad():
        hf_out = hf_model(**inputs)

    if pooling:
        hf_embeddings = POOLING_MAP[pooling](hf_out.last_hidden_state, inputs["attention_mask"])
    else:
        hf_embeddings = hf_out.last_hidden_state

    print(f"  Shape : {tuple(hf_embeddings.shape)}")
    print(f"  Mean  : {hf_embeddings.mean().item():.6f}")
    print(f"  Std   : {hf_embeddings.std().item():.6f}")

    # ── Stage 2: build QEff model ─────────────────────────────────────────────
    # Weight-free: meta model (zero RAM for weights) via init_empty_weights.
    # Normal:      reuse the already-loaded hf_model.
    print_section("Stage 2: Build QEff model")

    if args.weight_free:
        print("  Building meta model (init_empty_weights) ...")
        config = AutoConfig.from_pretrained(args.model_name)
        with init_empty_weights():
            meta_model = HFAutoModel.from_config(config, attn_implementation="eager")
        qeff_model = QEFFAutoModel(
            meta_model,
            pretrained_model_name_or_path=args.model_name,
            pooling=pooling,
        )
        print("  Meta model built — no weights in RAM.")
        print("  Skipping QEff-PT inference (meta tensors have no values).")
        mad_hf_vs_qeff = None
        max_hf_vs_qeff = None
    else:
        qeff_model = QEFFAutoModel(
            hf_model,
            pretrained_model_name_or_path=args.model_name,
            pooling=pooling,
        )
        qeff_pt_out = qeff_model.generate(inputs=inputs, runtime_ai100=False)
        qeff_pt_embeddings = qeff_pt_out if pooling else qeff_pt_out[0]
        print(f"  Shape : {tuple(qeff_pt_embeddings.shape)}")
        mad_hf_vs_qeff = torch.mean(torch.abs(hf_embeddings - qeff_pt_embeddings)).item()
        max_hf_vs_qeff = torch.max(torch.abs(hf_embeddings - qeff_pt_embeddings)).item()
        print_comparison("HF PT  vs  QEff PT", mad_hf_vs_qeff, max_hf_vs_qeff, threshold=0.0)

    # ── Stage 3: ONNX export ──────────────────────────────────────────────────
    print_section("Stage 3: ONNX export")
    if profiler:
        profiler.mark_operation("Export")

    export_dir = Path(args.export_dir) if args.export_dir else None

    onnx_path = Path(
        qeff_model.export(
            export_dir=str(export_dir) if export_dir else None,
            use_dynamo=args.use_dynamo,
            use_onnx_subfunctions=args.use_subfunctions,
            use_weight_free_export=args.weight_free,
            offload_pt_weights=not args.weight_free,
        )
    )
    print(f"  ONNX path   : {onnx_path}")

    # Weight-free: materialise FP32 safetensors next to the ONNX
    if args.weight_free:
        weight_spec_path = resolve_weight_spec_path(onnx_path)
        print(f"  Weight spec : {weight_spec_path}")
        print("  Converting checkpoint to FP32 (one-time) ...")
        convert_checkpoint_to_fp32(onnx_path, weight_spec_path)

    # ── Stage 3b: ORT inference ───────────────────────────────────────────────
    print_section("Stage 3b: ORT inference")
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    onnx_runtime_inputs = {
        "input_ids": np.array(inputs["input_ids"]),
        "attention_mask": np.array(inputs["attention_mask"]),
    }

    if args.weight_free:
        # Inject weight tensors from safetensors into ORT inputs
        onnx_runtime_inputs = load_weight_free_ort_inputs(weight_spec_path, onnx_runtime_inputs)

    onnx_out = ort_session.run(None, onnx_runtime_inputs)
    onnx_embeddings = onnx_out[0]
    print(f"  Shape : {onnx_embeddings.shape}")

    hf_np = hf_embeddings.detach().numpy()
    mad_hf_vs_onnx = float(np.mean(np.abs(hf_np - onnx_embeddings)))
    max_hf_vs_onnx = float(np.max(np.abs(hf_np - onnx_embeddings)))
    print_comparison("HF PT  vs  ONNX", mad_hf_vs_onnx, max_hf_vs_onnx, threshold=1e-5)

    # ── Stage 4: AI100 ────────────────────────────────────────────────────────
    mad_onnx_vs_ai100 = mad_hf_vs_ai100 = None
    max_onnx_vs_ai100 = max_hf_vs_ai100 = None

    if args.skip_ai100:
        print_section("Stage 4: AI100 (skipped)")
    else:
        print_section("Stage 4: AI100 hardware inference")
        if profiler:
            profiler.mark_operation("Compilation")

        qeff_model.compile(
            onnx_path=str(onnx_path),
            num_cores=args.num_cores,
            seq_len=args.seq_len,
            use_dynamo=args.use_dynamo,
            use_onnx_subfunctions=args.use_subfunctions,
            use_weight_free_export=args.weight_free,
        )

        if profiler:
            profiler.mark_operation("Generation")

        ai100_out = qeff_model.generate(inputs=inputs)
        # AI100 always pads output to the compiled seq_len.
        # With pooling: output is (B, hidden) — no trim needed.
        # Without pooling: output is (B, seq_len_compiled, hidden) — trim to actual token count.
        if pooling:
            ai100_embeddings = ai100_out["output"]
        else:
            actual_len = inputs["input_ids"].shape[1]
            ai100_embeddings = ai100_out["output"][:, :actual_len, :]
        print(f"  Shape : {ai100_embeddings.shape}")

        mad_onnx_vs_ai100 = float(np.mean(np.abs(onnx_embeddings - ai100_embeddings)))
        max_onnx_vs_ai100 = float(np.max(np.abs(onnx_embeddings - ai100_embeddings)))
        mad_hf_vs_ai100 = float(np.mean(np.abs(hf_np - ai100_embeddings)))
        max_hf_vs_ai100 = float(np.max(np.abs(hf_np - ai100_embeddings)))

        print_comparison("ONNX   vs  AI100", mad_onnx_vs_ai100, max_onnx_vs_ai100, threshold=1e-2)
        print_comparison("HF PT  vs  AI100", mad_hf_vs_ai100, max_hf_vs_ai100, threshold=1e-2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_section("Summary")
    if mad_hf_vs_qeff is not None:
        print_comparison("HF PT  vs  QEff PT", mad_hf_vs_qeff, max_hf_vs_qeff, threshold=0.0)
    else:
        print("  HF PT  vs  QEff PT : N/A (weight-free — meta model has no values)")
    print_comparison("HF PT  vs  ONNX   ", mad_hf_vs_onnx, max_hf_vs_onnx, threshold=1e-5)
    if mad_onnx_vs_ai100 is not None:
        print_comparison("ONNX   vs  AI100  ", mad_onnx_vs_ai100, max_onnx_vs_ai100, threshold=1e-2)
        print_comparison("HF PT  vs  AI100  ", mad_hf_vs_ai100, max_hf_vs_ai100, threshold=1e-2)

    # ── Stop profiler ─────────────────────────────────────────────────────────
    if profiler:
        profiler.stop_monitoring()
        profiler.generate_memory_graph(args.profile_output)
        print(profiler.get_memory_report())
        print(f"\n  Memory profile graph saved → {args.profile_output}")
    print()


if __name__ == "__main__":
    main()
