# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient.exporter.weight_free import load_weight_free_ort_inputs
from QEfficient.exporter.weight_spec import load_weight_spec, resolve_weight_spec_path
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils import get_padding_shape_from_config

MODEL_KWARGS = {"attn_implementation": "eager"}


def _create_local_llama_checkpoint(model_dir: Path) -> AutoConfig:
    config = AutoConfig.for_model(
        "llama",
        max_position_embeddings=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=128,
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_config(config, **MODEL_KWARGS)
    model.save_pretrained(model_dir)
    return config


def _create_runtime_inputs(config: AutoConfig, batch_size: int = 2, seq_len: int = 5, ctx_len: int = 16):
    padding_shape = get_padding_shape_from_config(config=config, batch_size=batch_size, seq_len=ctx_len)
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.int64).reshape(batch_size, seq_len) % config.vocab_size
    position_ids = torch.arange(seq_len, dtype=torch.int64).reshape(1, seq_len).repeat(batch_size, 1)
    past_key_values = []
    for _ in range(config.num_hidden_layers):
        past_key_values.append(
            (
                torch.zeros(padding_shape, dtype=torch.float32),
                torch.zeros(padding_shape, dtype=torch.float32),
            )
        )

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "past_key_values": tuple(past_key_values),
    }


def _to_ort_inputs(runtime_inputs):
    ort_inputs = {
        "input_ids": runtime_inputs["input_ids"].numpy(),
        "position_ids": runtime_inputs["position_ids"].numpy(),
    }
    for layer_idx, (past_key, past_value) in enumerate(runtime_inputs["past_key_values"]):
        ort_inputs[f"past_key.{layer_idx}"] = past_key.numpy()
        ort_inputs[f"past_value.{layer_idx}"] = past_value.numpy()
    return ort_inputs


def test_causal_lm_weight_free_export_and_parity(tmp_path):
    checkpoint_dir = tmp_path / "llama-checkpoint"
    config = _create_local_llama_checkpoint(checkpoint_dir)
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(checkpoint_dir)

    onnx_path = Path(
        qeff_model.export(
            tmp_path / "weight-free-export",
            use_dynamo=True,
            use_weight_free_export=True,
            offload_pt_weights=False,
        )
    )
    spec_path = resolve_weight_spec_path(onnx_path)

    assert onnx_path.is_file()
    assert spec_path.is_file()
    assert Path(qeff_model.weight_spec_path) == spec_path

    spec = load_weight_spec(spec_path)
    assert spec.model_id == str(checkpoint_dir)
    assert spec.files
    assert spec.inputs

    onnx_model = onnx.load(onnx_path, load_external_data=False)
    metadata_props = {entry.key: entry.value for entry in onnx_model.metadata_props}
    assert "com.qti.aisw.extdata" in metadata_props
    assert json.loads(metadata_props["com.qti.aisw.extdata"]) == json.loads(spec_path.read_text())

    initializer_names = {initializer.name for initializer in onnx_model.graph.initializer}
    assert not initializer_names.intersection({entry.name for entry in spec.inputs})

    runtime_inputs = _create_runtime_inputs(config)
    with torch.no_grad():
        pt_outputs = qeff_model.model(**runtime_inputs)

    ort_inputs = load_weight_free_ort_inputs(spec_path, _to_ort_inputs(runtime_inputs))
    ort_session = ort.InferenceSession(str(onnx_path))
    ort_outputs = ort_session.run(["logits"], ort_inputs)

    np.testing.assert_allclose(
        pt_outputs["logits"].detach().cpu().numpy(),
        ort_outputs[0],
        atol=1e-4,
        rtol=1e-4,
    )


def test_weight_free_export_requires_dynamo(tmp_path):
    checkpoint_dir = tmp_path / "llama-checkpoint"
    _create_local_llama_checkpoint(checkpoint_dir)
    qeff_model = QEFFAutoModelForCausalLM.from_pretrained(checkpoint_dir)

    with pytest.raises(NotImplementedError, match="use_dynamo=True"):
        qeff_model.export(tmp_path / "weight-free-export", use_weight_free_export=True)
