# Diagnostic: compare HF base vs QEff model on a single prefill forward pass.
# Both run in FP32 on CPU. Tells us WHERE the QEff model diverges from HF base.

import torch
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/huggingface_hub/glm51-fp32-stacked"
PROMPT = "what is faith ?"
NUM_LAYERS = 6

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
config = AutoConfig.from_pretrained(MODEL_PATH)
config.num_hidden_layers = NUM_LAYERS

input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
print(f"Prompt tokens: {input_ids.shape[1]} → {input_ids.tolist()}")

# ── HF base model ─────────────────────────────────────────────────────────────
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, config=config, torch_dtype=torch.float32, ignore_mismatched_sizes=True
)
hf_model.eval()

with torch.no_grad():
    hf_out = hf_model(input_ids, use_cache=False, output_hidden_states=True)

hf_logits = hf_out.logits            # [1, seq, vocab]
hf_next_token = hf_logits[0, -1].argmax().item()
print(f"\nHF  next token: {hf_next_token}")
print(f"HF  logit top5: {hf_logits[0,-1].topk(5).indices.tolist()}")
if hf_out.hidden_states:
    for i, h in enumerate(hf_out.hidden_states):
        print(f"  HF hidden[{i}] last-pos norm: {h[0,-1].norm().item():.4f}")

# ── QEff patched model ────────────────────────────────────────────────────────
# Load the same weights but apply QEff patches.
from QEfficient.transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    QEffGlmMoeDsaForCausalLM,
    QEffGlmMoeDsaModel,
    QEffGlmMoeDsaAttention,
    QEffGlmMoeDsaIndexer,
    QEffGlmMoeDsaMoE,
    QEffGlmMoeDsaTopkRouter,
    QEffGlmMoeDsaDenseDecoderLayer,
    QEffGlmMoeDsaSparseDecoderLayer,
)

qeff_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, config=config, torch_dtype=torch.float32, ignore_mismatched_sizes=True
)
# Patch classes
qeff_model.__class__ = QEffGlmMoeDsaForCausalLM
qeff_model.model.__class__ = QEffGlmMoeDsaModel
layer_types = getattr(config, "mlp_layer_types", [])
for i, layer in enumerate(qeff_model.model.layers):
    lt = layer_types[i] if i < len(layer_types) else "sparse"
    layer.__class__ = QEffGlmMoeDsaDenseDecoderLayer if lt == "dense" else QEffGlmMoeDsaSparseDecoderLayer
    layer.self_attn.__class__ = QEffGlmMoeDsaAttention
    layer.self_attn.indexer.__class__ = QEffGlmMoeDsaIndexer
    layer.self_attn.__qeff_init__()
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
        layer.mlp.gate.__class__ = QEffGlmMoeDsaTopkRouter
        layer.mlp.__class__ = QEffGlmMoeDsaMoE
        layer.mlp.__qeff_init__()
qeff_model.model.__qeff_init__()
qeff_model.eval()

# The QEff model requires:
# 1. A pre-allocated 256-slot dummy KV cache so past_seen_tokens=256 → proper 256-wide causal mask.
# 2. Explicit position_ids — with a pre-allocated cache, past_seen_tokens=256 so the model would
#    otherwise compute position_ids starting at 256, which is out-of-bounds for the 256-slot cache.
dummy_pkv = qeff_model.get_dummy_pkv_cache(config, batch_size=1, seq_len=256)
position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)   # [[0, 1, ..., 7]]

with torch.no_grad():
    qeff_out = qeff_model(
        input_ids,
        position_ids=position_ids,
        past_key_values=dummy_pkv,
        use_cache=True,
        output_hidden_states=True,
    )

qeff_logits = qeff_out.logits
qeff_next_token = qeff_logits[0, -1].argmax().item()
print(f"\nQEff next token: {qeff_next_token}")
print(f"QEff logit top5: {qeff_logits[0,-1].topk(5).indices.tolist()}")
if qeff_out.hidden_states:
    for i, h in enumerate(qeff_out.hidden_states):
        print(f"  QEff hidden[{i}] last-pos norm: {h[0,-1].norm().item():.4f}")

# ── Comparison ────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Match: {hf_next_token == qeff_next_token}")
print(f"HF  vs QEff logit max diff: {(hf_logits[0,-1] - qeff_logits[0,-1]).abs().max().item():.6f}")
print(f"HF  vs QEff logit cos-sim:  {torch.cosine_similarity(hf_logits[0,-1:], qeff_logits[0,-1:]).item():.6f}")
if hf_out.hidden_states and qeff_out.hidden_states:
    for i, (hh, qh) in enumerate(zip(hf_out.hidden_states, qeff_out.hidden_states)):
        diff = (hh[0,-1] - qh[0,-1]).abs()
        print(f"  Layer {i}: hidden diff max={diff.max().item():.6f}  mean={diff.mean().item():.6f}")
