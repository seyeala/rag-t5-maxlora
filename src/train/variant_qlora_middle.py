import argparse

from .common import (
    LoRA_TARGETS_ATT_MLP,
    TrainConfig,
    apply_lora_everywhere,
    freeze_lora_outside,
    get_num_layers_and_attr,
    load_4bit_model,
    load_tokenizer,
    make_dataset,
    middle_third_indices,
    prepare_model_for_kbit_training,
    run_trainer,
    unfreeze_lm_head,
)


def main(
    model_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
    train_path="data/processed/alpaca_train.jsonl",
    valid_path="data/processed/alpaca_valid.jsonl",
    out_dir="outputs/v2_qlora_middle",
    max_length=512,
    epochs=1.0,
    bs=1,
    accum=16,
    lr=2e-4,
    r=16,
    alpha=32,
    dropout=0.05,
    max_steps=None,
    train_limit=None,
    valid_limit=None,
):
    tokenizer = load_tokenizer(model_id)
    model = load_4bit_model(model_id)
    model = prepare_model_for_kbit_training(model)
    model = apply_lora_everywhere(
        model, r=r, alpha=alpha, dropout=dropout, targets=LoRA_TARGETS_ATT_MLP
    )

    num_layers, _ = get_num_layers_and_attr(model)
    middle_layers = set(middle_third_indices(num_layers))
    freeze_lora_outside(model, middle_layers)
    unfreeze_lm_head(model)

    config = TrainConfig(
        model_id=model_id,
        train_path=train_path,
        valid_path=valid_path,
        out_dir=out_dir,
        max_length=max_length,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=accum,
        learning_rate=lr,
        max_steps=max_steps,
        train_limit=train_limit,
        valid_limit=valid_limit,
    )

    train_dataset, valid_dataset = make_dataset(
        tokenizer,
        train_path,
        valid_path,
        max_length,
        is_encoder_decoder=model.config.is_encoder_decoder,
        train_limit=train_limit,
        valid_limit=valid_limit,
    )
    stats = run_trainer(model, tokenizer, train_dataset, valid_dataset, config)
    print("Efficiency:", stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    parser.add_argument("--train_path", default="data/processed/alpaca_train.jsonl")
    parser.add_argument("--valid_path", default="data/processed/alpaca_valid.jsonl")
    parser.add_argument("--out_dir", default="outputs/v2_qlora_middle")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--train_limit", type=int)
    parser.add_argument("--valid_limit", type=int)
    args = parser.parse_args()
    main(**vars(args))
