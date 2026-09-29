from types import SimpleNamespace

import train.common as common
import rag_t5.train.trainer as high


class FakeArgs:
    def __init__(
        self,
        output_dir,
        per_device_train_batch_size,
        gradient_accumulation_steps,
        learning_rate,
        num_train_epochs,
        weight_decay,
        warmup_ratio,
        label_smoothing_factor,
        gradient_checkpointing,
        bf16,
        logging_steps,
        save_strategy,
        report_to,
        load_best_model_at_end,
        eval_strategy,
    ):
        self.output_dir = output_dir
        self.eval_strategy = eval_strategy


class FakeTrainer:
    def __init__(self, model, args, train_dataset, eval_dataset, data_collator, processing_class):
        self.args = args
def test_build_trainer_filters_unsupported_training_arguments(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "TrainingArguments", FakeArgs)
    monkeypatch.setattr(common, "Trainer", FakeTrainer)
    monkeypatch.setattr(common, "resolve_bf16", lambda requested: False)

    cfg = common.TrainConfig(
        model_id="fake",
        train_path="train",
        valid_path="valid",
        out_dir=str(tmp_path),
        overwrite_output_dir=True,
        evaluation_strategy="steps",
    )

    trainer, args = common.build_trainer(
        object(), object(), [], [], cfg
    )

    assert isinstance(trainer, FakeTrainer)
    assert args.output_dir == str(tmp_path)
    assert args.eval_strategy == "steps"
def test_high_level_train_forwards_architecture_and_limits(monkeypatch, tmp_path):
    captured = {}
    fake_model = SimpleNamespace(
        config=SimpleNamespace(is_encoder_decoder=True),
        lm_head=SimpleNamespace(parameters=lambda: []),
    )

    monkeypatch.setattr(high, "resolve_bf16", lambda value: False)
    monkeypatch.setattr(high, "load_tokenizer", lambda model_id: object())
    monkeypatch.setattr(high, "load_fp_model", lambda model_id, dtype: fake_model)
    monkeypatch.setattr(high, "apply_lora_everywhere", lambda model, **kwargs: model)
    monkeypatch.setattr(high, "unfreeze_lm_head", lambda model: None)

    def fake_make_dataset(tokenizer, train_path, valid_path, max_length, **kwargs):
        captured.update(kwargs)
        return [], []

    monkeypatch.setattr(high, "make_dataset", fake_make_dataset)
    monkeypatch.setattr(high, "run_trainer", lambda *args, **kwargs: {"ok": True})
    cfg = high.TrainConfig(
        model_id="fake",
        train_path="train.jsonl",
        valid_path="valid.jsonl",
        out_dir=str(tmp_path),
        last_n_lora_layers=None,
        train_limit=7,
        valid_limit=3,
    )

    _, _, result = high.train(cfg)

    assert result == {"ok": True}
    assert captured == {
        "is_encoder_decoder": True,
        "train_limit": 7,
        "valid_limit": 3,
    }
