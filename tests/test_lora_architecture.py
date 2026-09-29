from types import SimpleNamespace

from peft import TaskType

import train.common as common


def fake_model(is_encoder_decoder):
    return SimpleNamespace(config=SimpleNamespace(is_encoder_decoder=is_encoder_decoder))


def test_architecture_defaults():
    seq2seq = fake_model(True)
    causal = fake_model(False)

    assert common.model_is_encoder_decoder(seq2seq)
    assert common.default_lora_targets(seq2seq) == ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
    assert common.default_lora_task_type(seq2seq) == TaskType.SEQ_2_SEQ_LM

    assert not common.model_is_encoder_decoder(causal)
    assert common.default_lora_targets(causal) == common.LoRA_TARGETS_ATT_MLP
    assert common.default_lora_task_type(causal) == TaskType.CAUSAL_LM
def test_apply_lora_builds_architecture_aware_config(monkeypatch):
    captured = {}

    def fake_get_peft_model(model, config):
        captured["model"] = model
        captured["config"] = config
        return "wrapped"

    monkeypatch.setattr(common, "get_peft_model", fake_get_peft_model)
    model = fake_model(True)

    assert common.apply_lora_everywhere(model, r=4, alpha=8, dropout=0.1) == "wrapped"
    config = captured["config"]
    assert config.r == 4
    assert config.lora_alpha == 8
    assert config.task_type == TaskType.SEQ_2_SEQ_LM
    assert set(config.target_modules) == {"q", "k", "v", "o", "wi_0", "wi_1", "wo"}


def test_layer_index_helpers():
    assert list(common.middle_third_indices(9)) == [3, 4, 5]
    assert list(common.last_n_indices(6, 2)) == [4, 5]
    assert list(common.last_n_indices(2, 5)) == [0, 1]


class FakeParam:
    def __init__(self):
        self.requires_grad = True


class NamedParamModel:
    def __init__(self):
        self.params = {
            "encoder.block.0.layer.0.q.lora_A.weight": FakeParam(),
            "encoder.block.1.layer.0.q.lora_A.weight": FakeParam(),
            "model.layers.2.self_attn.q_proj.lora_A.weight": FakeParam(),
            "model.layers.3.self_attn.q_proj.lora_A.weight": FakeParam(),
            "lm_head.weight": FakeParam(),
        }

    def named_parameters(self):
        return self.params.items()


def test_freeze_lora_outside_handles_t5_and_causal_layer_names():
    model = NamedParamModel()
    common.freeze_lora_outside(model, {1, 3})

    assert not model.params["encoder.block.0.layer.0.q.lora_A.weight"].requires_grad
    assert model.params["encoder.block.1.layer.0.q.lora_A.weight"].requires_grad
    assert not model.params["model.layers.2.self_attn.q_proj.lora_A.weight"].requires_grad
    assert model.params["model.layers.3.self_attn.q_proj.lora_A.weight"].requires_grad
    assert model.params["lm_head.weight"].requires_grad
