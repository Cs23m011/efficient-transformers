# Weight-Free Causal LM Export

`QEFFAutoModelForCausalLM.export(..., use_weight_free_export=True, use_dynamo=True)` exports a graph-only ONNX model and writes `weight_spec.json` beside it.

Current scope:

- Supports causal LM Torch-to-ONNX export verification.
- Rebuilds a meta-device QEff model from the original `from_pretrained()` checkpoint path.
- Promotes parameters and buffers to graph inputs and records their binding contract in `weight_spec.json`.
- Includes an ORT binding helper at `QEfficient.exporter.load_weight_free_ort_inputs`.

Current limits:

- AI100 compile integration is not wired yet for weight-free graphs.
- Quantized checkpoints are not supported yet.
- The path currently requires `use_dynamo=True`.

Example:

```python
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

qeff_model = QEFFAutoModelForCausalLM.from_pretrained("/path/to/model")
onnx_path = qeff_model.export(
    use_dynamo=True,
    use_weight_free_export=True,
    offload_pt_weights=False,
)
print(onnx_path)
print(qeff_model.weight_spec_path)
```
