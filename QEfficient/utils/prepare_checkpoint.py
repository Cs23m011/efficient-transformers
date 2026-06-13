# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
"""
Prepare a HuggingFace MoE checkpoint for weight-free QAIC export.

Downloads each shard from the hub one at a time, converts BF16→FP32, stacks
per-expert gate_proj / up_proj / down_proj weights into a single
gate_up_proj / down_proj tensor, and writes a self-contained FP32 checkpoint
directory.  The BF16 source shards are never kept on disk; each is deleted
immediately after processing.

Usage
-----
python -m QEfficient.utils.prepare_checkpoint \\
    --repo  zai-org/GLM-5 \\
    --out   /data/glm5-fp32-stacked

The output directory can be passed directly to QEFFAutoModelForCausalLM.from_pretrained().
For weight-free export the _convert_checkpoint_to_fp32 step is a no-op because the
files are already FP32 and co-located with the ONNX.
"""

import argparse
import json
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

import psutil
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open
from safetensors.torch import save_file


class PeakRAMTracker:
    """Background-thread peak RSS tracker (mirrors dynamo.py)."""

    def __init__(self, interval: float = 0.2):
        self._proc = psutil.Process(os.getpid())
        self._interval = interval
        self._peak_bytes = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._peak_bytes = self._proc.memory_info().rss
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            rss = self._proc.memory_info().rss
            if rss > self._peak_bytes:
                self._peak_bytes = rss
            time.sleep(self._interval)

    def stop(self) -> float:
        """Stop and return peak RAM in MB."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return self._peak_bytes / (1024 ** 2)

# Auxiliary files to copy verbatim from the repo
AUX_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
]

# Matches:  model.layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}.weight
EXPERT_RE = re.compile(
    r"^(model\.layers\.(\d+)\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
# Matches the layer index for MTP-layer filtering
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


class LayerStacker:
    """
    Accumulates per-expert tensors for one MoE layer and fuses them into:
      gate_up_proj: [E, 2*I, H]  (gate and up concatenated along dim=1)
      down_proj:    [E, H,   I]

    All tensors are stored as float32 regardless of source dtype.
    """

    def __init__(self, prefix: str, num_experts: int):
        self.prefix = prefix        # e.g. "model.layers.7.mlp.experts"
        self.num_experts = num_experts
        self.gate_up: torch.Tensor | None = None
        self.down: torch.Tensor | None = None
        self.filled = 0

    def add(self, expert_idx: int, kind: str, t: torch.Tensor) -> None:
        t = t.to(torch.float32)
        if kind in ("gate_proj", "up_proj"):
            I, H = t.shape
            if self.gate_up is None:
                self.gate_up = torch.empty(self.num_experts, 2 * I, H, dtype=torch.float32)
            offset = 0 if kind == "gate_proj" else I
            self.gate_up[expert_idx, offset : offset + I, :] = t
        else:  # down_proj  shape: [H, I]
            H, I = t.shape
            if self.down is None:
                self.down = torch.empty(self.num_experts, H, I, dtype=torch.float32)
            self.down[expert_idx] = t
        self.filled += 1

    @property
    def complete(self) -> bool:
        # gate + up + down = 3 tensors per expert
        return self.filled == 3 * self.num_experts

    def tensors(self) -> dict:
        return {
            f"{self.prefix}.gate_up_proj": self.gate_up,
            f"{self.prefix}.down_proj": self.down,
        }


def _atomic_save(tensors: dict, dst: Path) -> None:
    """Write tensors to a .tmp file then rename to dst (crash-safe)."""
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp))
    tmp.replace(dst)


def _safe_delete(src: Path, tmp_root: str) -> None:
    """
    Delete the downloaded shard and its underlying blob.
    Only deletes the blob if it lives inside the tmp_root directory to avoid
    corrupting the real HuggingFace cache (~/.cache/huggingface).
    """
    blob = src.resolve()
    src.unlink(missing_ok=True)
    if blob != src and str(blob).startswith(tmp_root):
        blob.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--repo", required=True, help="HuggingFace repo id, e.g. zai-org/GLM-5")
    ap.add_argument("--out", required=True, type=Path, help="Output directory for the FP32 checkpoint")
    ap.add_argument("--revision", default="main", help="Branch / tag / commit (default: main)")
    ap.add_argument(
        "--keep-mtp",
        action="store_true",
        help="Keep MTP / next-N predict weights (dropped by default)",
    )
    ap.add_argument(
        "--no-stack",
        action="store_true",
        help="Skip expert stacking; only convert BF16→FP32 (per-expert layout preserved)",
    )
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    ram_tracker = PeakRAMTracker()
    ram_tracker.start()

    # ── Download auxiliary (non-weight) files ─────────────────────────────────
    repo_files = list(list_repo_files(args.repo, revision=args.revision))
    for name in AUX_FILES:
        if name in repo_files and not (out / name).exists():
            shutil.copy2(
                hf_hub_download(args.repo, name, revision=args.revision),
                out / name,
            )
            print(f"[aux] {name}")

    # ── Read model config ──────────────────────────────────────────────────────
    config = json.loads((out / "config.json").read_text())
    num_layers = int(config["num_hidden_layers"])
    num_experts = int(config.get("n_routed_experts") or config.get("num_local_experts"))
    drop_mtp = not args.keep_mtp

    # ── Build shard plan ───────────────────────────────────────────────────────
    index_path = out / "model.safetensors.index.json"
    if index_path.exists():
        weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]
    else:
        # Single-file model (tiny/test models) — treat all keys as in one shard
        single = next(f for f in repo_files if f.endswith(".safetensors"))
        weight_map = {"*": single}

    shard_order = sorted(set(weight_map.values()))
    new_weight_map: dict[str, str] = {}
    stackers: dict[int, LayerStacker] = {}

    def is_mtp(key: str) -> bool:
        """True if key belongs to an MTP/speculative layer beyond num_layers."""
        m = LAYER_RE.match(key)
        return drop_mtp and m is not None and int(m.group(1)) >= num_layers

    # ── Process shards ─────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="qeff_ckpt_") as tmp_cache:
        for si, shard in enumerate(shard_order):
            base_out = out / (
                f"base-{Path(shard).stem}.safetensors" if not args.no_stack else shard
            )

            # Determine if all output files for this shard already exist
            expert_layers_this_shard = {
                int(EXPERT_RE.match(k).group(2))
                for k, v in weight_map.items()
                if v == shard and EXPERT_RE.match(k)
            } if not args.no_stack and "*" not in weight_map else set()

            already_done = base_out.exists() and all(
                (out / f"experts-layer-{li:05d}.safetensors").exists()
                for li in expert_layers_this_shard
            )
            if already_done:
                print(f"[{si+1}/{len(shard_order)}] {shard}  (already done, skip)")
                # Rebuild new_weight_map entries so the index is consistent
                for key, v in weight_map.items():
                    if v != shard or is_mtp(key):
                        continue
                    m = EXPERT_RE.match(key)
                    if m and not args.no_stack:
                        li = int(m.group(2))
                        new_weight_map[f"{m.group(1)}.gate_up_proj"] = f"experts-layer-{li:05d}.safetensors"
                        new_weight_map[f"{m.group(1)}.down_proj"] = f"experts-layer-{li:05d}.safetensors"
                    else:
                        new_weight_map[key] = base_out.name
                continue

            # Download shard to temp dir (never touches the real HF cache)
            dl_start = time.perf_counter()
            src = Path(
                hf_hub_download(args.repo, shard, revision=args.revision, cache_dir=tmp_cache)
            )
            dl_elapsed = time.perf_counter() - dl_start
            shard_size_mb = src.stat().st_size / (1024 ** 2)
            print(f"[TIMING] download {shard}: {dl_elapsed:.1f}s  ({shard_size_mb:.0f} MB)")

            proc_start = time.perf_counter()
            base_tensors: dict[str, torch.Tensor] = {}
            layers_stacked_this_shard = 0
            with safe_open(str(src), framework="pt") as f:
                for key in f.keys():
                    if is_mtp(key):
                        continue
                    m = EXPERT_RE.match(key)
                    if m and not args.no_stack:
                        # Expert weight → accumulate in stacker
                        li, ei, kind = int(m.group(2)), int(m.group(3)), m.group(4)
                        st = stackers.setdefault(li, LayerStacker(m.group(1), num_experts))
                        st.add(ei, kind, f.get_tensor(key))
                        new_weight_map[f"{st.prefix}.gate_up_proj"] = f"experts-layer-{li:05d}.safetensors"
                        new_weight_map[f"{st.prefix}.down_proj"] = f"experts-layer-{li:05d}.safetensors"
                        if st.complete:
                            stack_start = time.perf_counter()
                            _atomic_save(st.tensors(), out / f"experts-layer-{li:05d}.safetensors")
                            stack_elapsed = time.perf_counter() - stack_start
                            saved_mb = (out / f"experts-layer-{li:05d}.safetensors").stat().st_size / (1024 ** 2)
                            print(
                                f"    [STACKED] layer {li}: "
                                f"gate_up {tuple(st.gate_up.shape)}, "
                                f"down {tuple(st.down.shape)}  "
                                f"-> {saved_mb:.0f} MB  ({stack_elapsed:.1f}s)"
                            )
                            layers_stacked_this_shard += 1
                            del stackers[li]
                    else:
                        # Non-expert weight → BF16→FP32 and pass through
                        t = f.get_tensor(key)
                        base_tensors[key] = t.to(torch.float32) if t.is_floating_point() else t
                        new_weight_map[key] = base_out.name

            if base_tensors:
                _atomic_save(base_tensors, base_out)

            # Delete the BF16 shard; keep only if it is outside tmp_cache
            _safe_delete(src, tmp_cache)

            proc_elapsed = time.perf_counter() - proc_start
            base_mb = base_out.stat().st_size / (1024 ** 2) if base_out.exists() else 0
            n_inflight = len(stackers)
            print(
                f"[{si+1}/{len(shard_order)}] {shard} → {base_out.name} "
                f"({len(base_tensors)} base tensors, {base_mb:.0f} MB written, "
                f"{layers_stacked_this_shard} layer(s) stacked, "
                f"{n_inflight} in-flight)  proc: {proc_elapsed:.1f}s"
            )

    # ── Sanity check ──────────────────────────────────────────────────────────
    if stackers:
        raise RuntimeError(
            f"Incomplete expert layers after all shards: {sorted(stackers.keys())} "
            f"— checkpoint is missing keys."
        )

    # ── Update config.json dtype ───────────────────────────────────────────────
    for k in ("dtype", "torch_dtype"):
        if k in config:
            config[k] = "float32"
    (out / "config.json").write_text(json.dumps(config, indent=2))

    # ── Write new model.safetensors.index.json ─────────────────────────────────
    output_files = sorted(set(new_weight_map.values()))
    total_size = sum((out / f).stat().st_size for f in output_files)
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(new_weight_map.items())),
    }
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    peak_ram_mb = ram_tracker.stop()
    total_elapsed = time.perf_counter() - total_start

    print(f"\n{'='*60}")
    print(f"[SUMMARY] FP32 stacked checkpoint at: {out}")
    print(f"[SUMMARY] Output files     : {len(output_files)}")
    print(f"[SUMMARY] Total size       : {total_size/1e9:.2f} GB")
    print(f"[SUMMARY] Total time       : {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")
    print(f"[SUMMARY] Peak RAM         : {peak_ram_mb:.0f} MB  ({peak_ram_mb/1024:.2f} GB)")
    print(f'[SUMMARY] Use as           : model_name_or_path = "{out.resolve()}"')
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
