# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
import threading
import time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
import psutil
from accelerate import init_empty_weights
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
# import torch._dynamo
# torch._dynamo.config.verbose=True 
# import logging
# logging.getLogger("torch._dynamo").setLevel(logging.DEBUG)
from safetensors import safe_open
from QEfficient.exporter.weight_free import _default_weights_roots, load_weight_free_ort_inputs
from QEfficient.exporter.weight_spec import (
    ExternalDataFile,
    load_weight_spec,
    resolve_weight_spec_path,
    save_weight_spec,
)
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner


class PeakRAMTracker:
    """Background-thread peak RSS tracker for the current process + optional children."""

    def __init__(self, interval: float = 0.2):
        self._proc = psutil.Process(os.getpid())
        self._interval = interval
        self._peak_bytes = 0
        self._running = False
        self._thread = None
        self._include_children = False

    def start(self, include_children: bool = False):
        self._include_children = include_children
        self._peak_bytes = self._rss()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _rss(self) -> int:
        try:
            rss = self._proc.memory_info().rss
            if self._include_children:
                for child in self._proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            return rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    def _loop(self):
        while self._running:
            rss = self._rss()
            if rss > self._peak_bytes:
                self._peak_bytes = rss
            time.sleep(self._interval)

    def stop(self) -> float:
        """Stop tracking and return peak RAM in GB."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return self._peak_bytes / 1024 ** 3


def convert_checkpoint_to_fp32(onnx_path: Path, weight_spec_path: Path) -> None:
    """
    Extract only the tensors referenced by weight_spec inputs, cast to FP32,
    save next to the ONNX, and update weight_spec.json to point there.

    This ensures the compiler sees matching dtypes between the ONNX (FLOAT)
    and the safetensors files (also FLOAT after conversion).
    Only files/tensors actually used by the exported graph are written —
    for a 2-layer model sliced from a 30-shard checkpoint this avoids
    loading the irrelevant shards entirely.
    """
    spec = load_weight_spec(weight_spec_path)
    export_dir = onnx_path.parent
    candidate_roots = _default_weights_roots(weight_spec_path, spec)

    # Reuse only when spec.files already points to local relative files written by
    # a previous convert run.  Checking by filename alone is wrong: a fresh export
    # rewrites weight_spec.json to the original checkpoint paths, so stale local
    # files from the last run would silently miss tensors added in the new spec.
    if spec.files and all(
        not Path(f.path).is_absolute() and (export_dir / f.path).is_file()
        for f in spec.files
    ):
        print("Reusing existing local FP32 safetensors.")
        return

    # Build: old file index -> set of tensor keys actually needed.
    needed: dict[int, set[str]] = {}
    for inp in spec.inputs:
        file_idx = int(inp.location.file)
        needed.setdefault(file_idx, set()).add(inp.location.key)

    sorted_old_idxs = sorted(needed.keys())
    # Compact the file list: map old index -> new sequential index.
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
            # Checkpoint shard is already FP32 (download_fp32.py) — reference in place, zero copy.
            new_files.append(ExternalDataFile(path=str(abs_path), format="safetensors"))
            print(f"  {abs_path.name}  ({len(keys_needed)} tensors)  ->  referenced in place (already fp32)")
        else:
            tensors = load_file(str(abs_path))
            fp32_tensors = {k: v.to(torch.float32) for k, v in tensors.items() if k in keys_needed}
            out_name = f"model_{old_to_new[old_idx]:04d}.safetensors"
            save_file(fp32_tensors, str(export_dir / out_name))
            new_files.append(ExternalDataFile(path=out_name, format="safetensors"))
            print(f"  {abs_path.name}  ({len(keys_needed)}/{len(tensors)} tensors)  ->  {out_name}  (float32)")

    for inp in spec.inputs:
        inp.location.file = old_to_new[int(inp.location.file)]

    spec.files = new_files
    save_weight_spec(weight_spec_path, spec)
    _sync_embedded_extdata(onnx_path, weight_spec_path)


def _sync_embedded_extdata(onnx_path: Path, weight_spec_path: Path) -> None:
    # Keep the embedded external-data metadata aligned with weight_spec.json so
    # compiler and ORT verification resolve the same files.
    updated_json = json.dumps(json.loads(weight_spec_path.read_text()), separators=(",", ":"), sort_keys=True)
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    for entry in onnx_model.metadata_props:
        if entry.key == "com.qti.aisw.extdata":
            entry.value = updated_json
            break
    tmp = onnx_path.with_suffix(onnx_path.suffix + ".tmp")
    onnx.save(onnx_model, str(tmp))
    tmp.replace(onnx_path)


PROMPT = "what is faith ?"

#model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
#model_name = "ibm-granite/granite-3.0-3b-a800m-instruct"
#model_name="Qwen/Qwen3-8B-A3B-Instruct-2507"
#model_name="Qwen/Qwen3-235B-A22B-Instruct-2507"
#model_name = "openai/gpt-oss-20b"
#model_name="/home/amarshar/weightfree-tf5/gpt-oss-20b-dequant"
#model_name = "meta-llama/Llama-3.2-1B"
#model_name="Qwen/Qwen3-32B"
#model_name="meta-llama/Llama-3.3-70B-Instruct"
#model_name="Qwen/Qwen3-30B-A3B-Instruct-2507"
#model_name="tiny-random/glm-5.1"
model_name="/home/huggingface_hub/glm51-fp32-stacked"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
config.num_hidden_layers = 6
config.dtype = torch.float32
print(config)

CONTINUOUS_BATCHING = False
FULL_BATCH_SIZE = 4  # slots in the KV cache; active batch_size stays at 1 here # NOT VERIFIED, WIP

runner = ApiRunner(
    batch_size=1,
    tokenizer=tokenizer,
    config=config,
    prompt=[PROMPT],
    prompt_len=8,
    ctx_len=256,
    full_batch_size=FULL_BATCH_SIZE if CONTINUOUS_BATCHING else None,
)
with init_empty_weights():
    meta_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
#meta_model=AutoModelForCausalLM.from_pretrained(model_name,config=config)
qeff_model = QEFFAutoModelForCausalLM(
    meta_model,
    pretrained_model_name_or_path=model_name,
    continuous_batching=CONTINUOUS_BATCHING,
)

export_dir = Path("test_models/weightfree_from_config")
print("Exporting ...")
_ram = PeakRAMTracker()
_ram.start()
export_start = time.perf_counter()
onnx_path = Path(
    qeff_model.export(
        export_dir=export_dir,
        use_dynamo=True,
        use_onnx_subfunctions=True,
        use_weight_free_export=True,
        offload_pt_weights=False,
    )
)
export_elapsed = time.perf_counter() - export_start
export_peak_ram = _ram.stop()
weight_spec_path = resolve_weight_spec_path(onnx_path)

print(f"Weight-free export time : {export_elapsed:.3f} sec")
print(f"Export peak RAM         : {export_peak_ram:.2f} GB")

print("Converting checkpoint to FP32 (one-time local materialization) ...")
_ram3=PeakRAMTracker()
fp32_convert_time_start = time.perf_counter()
convert_checkpoint_to_fp32(onnx_path, weight_spec_path)
export_peak_ram=_ram3.stop()
fp32_convert_time = time.perf_counter() - fp32_convert_time_start
print(f"fp32 convert time: {fp32_convert_time:.3f} sec")
print(f"Export peak fp32 RAM  : {export_peak_ram:.2f} GB")
print("Compiling weight-free ONNX ...")
_ram2 = PeakRAMTracker()
_ram2.start(include_children=True)   # track qaic-compile subprocess too
compile_start = time.perf_counter()
qpc_path = qeff_model.compile(
    onnx_path=str(onnx_path),
    compile_dir=str(onnx_path.parent / "qpc"),
    prefill_seq_len=1,
    ctx_len=256,
    num_devices=4,
    mxfp6_matmul=False,
    mxint8_kv_cache=False,
    use_dynamo=True,
    use_onnx_subfunctions=True,
    use_weight_free_export=True,
)
compile_elapsed = time.perf_counter() - compile_start
compile_peak_ram = _ram2.stop()
print(f"compile time            : {compile_elapsed:.3f} sec")
print(f"Compile peak RAM        : {compile_peak_ram:.2f} GB")
print(f"QPC: {qpc_path}")

# #── OnnxRT inference ──────────────────────────────────────────────────────────
print("\n--- OnnxRT inference ---")
session = ort.InferenceSession(str(onnx_path))
ort_inputs = load_weight_free_ort_inputs(weight_spec_path, runner.input_handler.prepare_ort_inputs())
ort_outputs = runner.run_ort_session(ort_inputs, session)
ort_outputs = runner.input_handler.update_ort_outputs(ort_outputs)

ort_generated_ids = []
for _ in range(1, runner.gen_len):
    ort_generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
    ort_inputs = runner.input_handler.update_ort_inputs(ort_inputs, ort_outputs)
    ort_inputs = load_weight_free_ort_inputs(weight_spec_path, ort_inputs)
    ort_outputs = runner.run_ort_session(ort_inputs, session)
    ort_outputs = runner.input_handler.update_ort_outputs(ort_outputs)

ort_generated_ids.append(ort_outputs["logits"].argmax(-1).reshape(-1, 1))
ort_generated_ids = np.concatenate(ort_generated_ids, axis=1)
ort_generated_text = tokenizer.batch_decode(ort_generated_ids, skip_special_tokens=True)

# # #── PyTorch inference ─────────────────────────────────────────────────────────
pt_config = AutoConfig.from_pretrained("/home/huggingface_hub/glm51-fp32-stacked")
pt_config.num_hidden_layers = 6
print("\n--- PyTorch inference ---")
pt_model = AutoModelForCausalLM.from_pretrained(
      "/home/huggingface_hub/glm51-fp32-stacked",
      config=pt_config,
      torch_dtype=torch.float32,
      ignore_mismatched_sizes=True,
  )
pt_model.eval()

input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
with torch.no_grad():
    pt_out = pt_model.generate(
        input_ids,
        max_new_tokens=runner.gen_len,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

#Keep only the newly generated tokens to match ORT output shape
pt_generated_ids = pt_out[:, input_ids.shape[1]:].numpy()
pt_generated_text = tokenizer.batch_decode(pt_generated_ids, skip_special_tokens=True)

# #── QPC inference ─────────────────────────────────────────────────────────────
print("\n--- QPC inference ---")
qpc_generated_ids = None
qpc_generated_text = None
try:
    exec_info = qeff_model.generate(
        prompts=[PROMPT],
        tokenizer=tokenizer,
        automation=True,
        generation_len=runner.gen_len,
    )
    qpc_generated_ids = np.asarray(exec_info.generated_ids[0]).reshape(1, -1)
    qpc_generated_text = tokenizer.batch_decode(qpc_generated_ids, skip_special_tokens=True)
    print(exec_info)
except RuntimeError as exc:
    print(f"Skipping QPC generate: {exc}")

#── Token comparison ──────────────────────────────────────────────────────────
# print("\n========== Token Comparison ==========")
# print(f"Prompt: {PROMPT!r}")
# print()
print(f"ORT  generated_ids : {ort_generated_ids}")
print(f"ORT  generated_text: {ort_generated_text}")
print()
print(f"PT   generated_ids : {pt_generated_ids}")
print(f"PT   generated_text: {pt_generated_text}")
print()
if qpc_generated_ids is not None:
    print(f"QPC  generated_ids : {qpc_generated_ids}")
    print(f"QPC  generated_text: {qpc_generated_text}")
else:
    print("QPC  generated_ids : (skipped)")

# Per-token match table
print("\n--- Per-token match (ORT vs PT vs QPC) ---")
max_len = max(
    pt_generated_ids.shape[1],
    qpc_generated_ids.shape[1] if qpc_generated_ids is not None else 0,
)
header = f"{'Step':>5}  {'ORT':>8}  {'PT':>8}  {'QPC':>8}  {'ORT==PT':>8}  {'ORT==QPC':>9}"
print(header)
print("-" * len(header))
for i in range(max_len):
    ort_tok = int(ort_generated_ids[0, i]) if i < ort_generated_ids.shape[1] else -1
    pt_tok  = int(pt_generated_ids[0, i])  if i < pt_generated_ids.shape[1]  else -1
    qpc_tok = int(qpc_generated_ids[0, i]) if (qpc_generated_ids is not None and i < qpc_generated_ids.shape[1]) else -1
    ort_eq_pt  = "✓" if ort_tok == pt_tok  else "✗"
    ort_eq_qpc = "✓" if (qpc_generated_ids is not None and ort_tok == qpc_tok) else ("N/A" if qpc_generated_ids is None else "✗")
    print(f"{i:>5}  {ort_tok:>8}  {pt_tok:>8}  {qpc_tok:>8}  {ort_eq_pt:>8}  {ort_eq_qpc:>9}")
