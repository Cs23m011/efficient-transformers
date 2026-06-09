from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import numpy as np

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

model_name = "tiny-random/glm-5.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
print(config)

runner = ApiRunner(
    batch_size=1,
    tokenizer=tokenizer,
    config=config,
    prompt=["My name is"],
    prompt_len=8,
    ctx_len=32,
)

# ── HF baseline ──────────────────────────────────────────────────────────────
hf_model = AutoModelForCausalLM.from_pretrained(model_name)
hf_tokens = runner.run_hf_model_on_pytorch(hf_model)
print("\nOriginal HF Model Outputs (Torch CPU):")
print(f"Prompt: {runner.prompt}")
print(f"Completion: {tokenizer.decode(hf_tokens[0], skip_special_tokens=True)!r}")
print(f"Token IDs: {hf_tokens}")

# ── QEff PT (KV) ──────────────────────────────────────────────────────────────
qeff_model = QEFFAutoModelForCausalLM(hf_model)
pt_tokens = runner.run_kv_model_on_pytorch(qeff_model.model)
print(f"\nQEff PT (KV) tokens: {pt_tokens}")

# ── Non-Dynamo export + ORT ───────────────────────────────────────────────────
print("\n--- Non-Dynamo Export (use_dynamo=False) ---")
onnx_path_nodynamo = qeff_model.export(use_dynamo=False, use_onnx_subfunctions=True)
ort_tokens_nodynamo = runner.run_kv_model_on_ort(onnx_path_nodynamo)
print(f"Non-Dynamo ORT tokens: {ort_tokens_nodynamo}")

# ── Dynamo export + ORT ───────────────────────────────────────────────────────
print("\n--- Dynamo Export (use_dynamo=True) ---")
onnx_path_dynamo = qeff_model.export(use_dynamo=True, use_onnx_subfunctions=True)
ort_tokens_dynamo = runner.run_kv_model_on_ort(onnx_path_dynamo)
print(f"Dynamo ORT tokens:     {ort_tokens_dynamo}")

# ── Comparison ────────────────────────────────────────────────────────────────
print("\n========== Token Comparison ==========")
n = min(len(hf_tokens[0]), len(pt_tokens[0]), len(ort_tokens_nodynamo[0]), len(ort_tokens_dynamo[0]))
hf   = np.array(hf_tokens[0][:n])
pt   = np.array(pt_tokens[0][:n])
nd   = np.array(ort_tokens_nodynamo[0][:n])
dy   = np.array(ort_tokens_dynamo[0][:n])

print(f"{'Step':>4} | {'HF':>8} | {'PT(KV)':>8} | {'ORT(no-dyn)':>12} | {'ORT(dynamo)':>12} | {'ND==Dyn':>8}")
print("-" * 68)
for i in range(n):
    match = "✓" if nd[i] == dy[i] else "✗"
    print(f"{i:>4} | {hf[i]:>8} | {pt[i]:>8} | {nd[i]:>12} | {dy[i]:>12} | {match:>8}")

mad = np.mean(np.abs(nd.astype(float) - dy.astype(float)))
exact = (nd == dy).sum()
print(f"\nNon-Dynamo vs Dynamo ORT — exact: {exact}/{n} ({100*exact/n:.1f}%)  MAD: {mad:.4f}")
