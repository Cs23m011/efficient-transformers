# -----------------------------------------------------------------------------
# One-time offline dequantization of a gpt_oss (MXFP4) checkpoint into a plain
# float checkpoint, so that weight_free_export_from_config.py can treat it
# EXACTLY like a dense Llama model.
#
# NOTE on the save path:
#   We do NOT use model.save_pretrained(). On the QEff-wrapped gpt_oss experts
#   module, save_pretrained's parameter collection DROPS the `down_proj` weight
#   (keeps down_proj_bias but loses the weight). We instead save the raw
#   model.state_dict() and ASSERT every required expert weight is present, so an
#   incomplete checkpoint can never ship silently.
#
# Run this ONCE per model. It is the only RAM-heavy step.
# -----------------------------------------------------------------------------

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from transformers import AutoTokenizer


def _shard_and_save(state_dict, out_dir: Path, max_shard_bytes: int):
    shards = []
    current, current_bytes = {}, 0
    for k, v in state_dict.items():
        nbytes = v.numel() * v.element_size()
        if current and current_bytes + nbytes > max_shard_bytes:
            shards.append(current)
            current, current_bytes = {}, 0
        current[k] = v
        current_bytes += nbytes
    if current:
        shards.append(current)

    total = len(shards)
    weight_map = {}
    if total == 1:
        fname = "model.safetensors"
        save_file(shards[0], str(out_dir / fname), metadata={"format": "pt"})
        for k in shards[0]:
            weight_map[k] = fname
    else:
        for i, shard in enumerate(shards, 1):
            fname = f"model-{i:05d}-of-{total:05d}.safetensors"
            save_file(shard, str(out_dir / fname), metadata={"format": "pt"})
            for k in shard:
                weight_map[k] = fname

    total_size = sum(v.numel() * v.element_size() for v in state_dict.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--out", default="gpt-oss-20b-dequant")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--num-layers", type=int, default=0,
                    help="if >0, truncate to this many hidden layers for a smoke test")
    ap.add_argument("--shard-gb", type=float, default=5.0)
    ap.add_argument("--keep-fused", action="store_true",
                    help="keep redundant fused gate_up_proj tensors (default: drop them)")
    args = ap.parse_args()

    save_dtype = getattr(torch, args.dtype)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading + dequantizing {args.model} via QEfficient from_pretrained ...")
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(args.model, continuous_batching=False)
    model = qeff_model.model

    if args.num_layers and args.num_layers > 0:
        try:
            n = args.num_layers
            model.model.layers = model.model.layers[:n]
            model.config.num_hidden_layers = n
            if getattr(model.config, "layer_types", None) is not None:
                model.config.layer_types = model.config.layer_types[:n]
            print(f"  (smoke test) truncated to {n} layers")
        except Exception as exc:
            print(f"  (smoke test) could not truncate layers: {exc}")

    print(f"Casting to {args.dtype} ...")
    model = model.to(dtype=save_dtype)

    if getattr(model.config, "quantization_config", None) is not None:
        try:
            delattr(model.config, "quantization_config")
        except Exception:
            model.config.quantization_config = None

    print("Collecting state_dict directly from model (preserves down_proj) ...")
    sd = {}
    for k, v in model.state_dict().items():
        t = v.detach()
        if t.is_floating_point():
            t = t.to(save_dtype)
        sd[k] = t.contiguous()

    if not args.keep_fused:
        before = len(sd)
        sd = {k: v for k, v in sd.items()
              if not (k.endswith("experts.gate_up_proj") or k.endswith("experts.gate_up_proj_bias"))}
        print(f"  dropped {before - len(sd)} redundant fused gate_up_proj tensors")

    num_layers = model.config.num_hidden_layers
    missing = []
    for layer in range(num_layers):
        for req in ("experts.gate_proj", "experts.up_proj", "experts.down_proj"):
            key = f"model.layers.{layer}.mlp.{req}"
            if key not in sd:
                missing.append(key)
    if missing:
        raise RuntimeError(
            f"Refusing to save incomplete checkpoint. Missing {len(missing)} required "
            f"expert weights, e.g.: {missing[:5]}"
        )
    print(f"  verified gate_proj / up_proj / down_proj present for all {num_layers} layers")

    print(f"Saving plain-float checkpoint to {out_dir} ...")
    n_shards = _shard_and_save(sd, out_dir, int(args.shard_gb * 1e9))
    model.config.save_pretrained(str(out_dir))
    AutoTokenizer.from_pretrained(args.model).save_pretrained(str(out_dir))

    print(f"\nDone. Wrote {n_shards} shard(s) to: {out_dir.resolve()}")
    print(f'Point your export script at: model_name = "{out_dir}"')
    for p in sorted(out_dir.glob("*.safetensors")):
        print(f"    {p.name}  ({p.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()