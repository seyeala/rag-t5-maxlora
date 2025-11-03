import argparse

from .common import (
    LoRA_TARGETS_ATT_MLP,
    TrainConfig,
    apply_lora_everywhere,
    freeze_lora_outside,
    get_num_layers_and_attr,
    last_n_indices,
    load_fp_model,
    load_tokenizer,
    make_dataset,
    run_trainer,
    unfreeze_lm_head,
)


def main(
    model_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
    train_path="data/processed/alpaca_train.jsonl",
    valid_path="data/processed/alpaca_valid.jsonl",
    out_dir="outputs/v3_tiny_last2_lora",
    max_length=512,
    epochs=1.0,
    bs=1,
    accum=16,
    lr=2e-4,
    last_n=2,
):
    tokenizer = load_tokenizer(model_id)
    model = load_fp_model(model_id)
    model = apply_lora_everywhere(
        model, r=16, alpha=32, dropout=0.05, targets=LoRA_TARGETS_ATT_MLP
    )

    num_layers, _ = get_num_layers_and_attr(model)
    allowed = set(last_n_indices(num_layers, last_n))
    freeze_lora_outside(model, allowed)
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
    )

    train_dataset, valid_dataset = make_dataset(
        tokenizer, train_path, valid_path, max_length
    )
    stats = run_trainer(model, tokenizer, train_dataset, valid_dataset, config)
    print("Efficiency:", stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="deepseek-ai/DeepSeek-V2-Lite-Chat")
    parser.add_argument("--train_path", default="data/processed/alpaca_train.jsonl")
    parser.add_argument("--valid_path", default="data/processed/alpaca_valid.jsonl")
    parser.add_argument("--out_dir", default="outputs/v3_tiny_last2_lora")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--last_n", type=int, default=2)
    args = parser.parse_args()
    main(**vars(args))
