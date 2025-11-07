import argparse

from .common import (
    TrainConfig,
    get_num_layers_and_attr,
    load_fp_model,
    load_tokenizer,
    make_dataset,
    middle_third_indices,
    run_trainer,
    unfreeze_lm_head,
)


def main(
    model_id="deepseek-ai/DeepSeek-V2-Lite-Chat",
    train_path="data/processed/alpaca_train.jsonl",
    valid_path="data/processed/alpaca_valid.jsonl",
    out_dir="outputs/v1_middle_ft",
    max_length=512,
    epochs=1.0,
    bs=1,
    accum=16,
    lr=1e-5,
    max_steps=None,
    train_limit=None,
    valid_limit=None,
):
    tokenizer = load_tokenizer(model_id)
    model = load_fp_model(model_id)

    for param in model.parameters():
        param.requires_grad = False

    num_layers, attr_path = get_num_layers_and_attr(model)
    middle_layers = set(middle_third_indices(num_layers))

    layer_container = model
    for attr in attr_path.split("."):
        layer_container = getattr(layer_container, attr)

    for layer_idx in middle_layers:
        for param in layer_container[layer_idx].parameters():
            param.requires_grad = True

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
    parser.add_argument("--out_dir", default="outputs/v1_middle_ft")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--train_limit", type=int)
    parser.add_argument("--valid_limit", type=int)
    args = parser.parse_args()
    main(**vars(args))
