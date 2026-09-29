import argparse
from pathlib import Path

from peft import PeftConfig, PeftModel
from transformers import AutoConfig, AutoTokenizer

from rag_t5.models.inference import _model_class


def export_merged_adapter(adapter_path: str, out_dir: str):
    adapter = Path(adapter_path)
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"PEFT adapter_config.json not found in {adapter}")

    peft_config = PeftConfig.from_pretrained(str(adapter))
    base_id = peft_config.base_model_name_or_path
    config = AutoConfig.from_pretrained(base_id)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter), use_fast=True)
    base = _model_class(config).from_pretrained(base_id, config=config)
    model = PeftModel.from_pretrained(base, str(adapter))
    merged = model.merge_and_unload()

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output)
    tokenizer.save_pretrained(output)
    return output


def main():
    parser = argparse.ArgumentParser(description="Merge a PEFT adapter into its base model.")
    parser.add_argument("adapter_path")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    print(export_merged_adapter(args.adapter_path, args.out_dir))


if __name__ == "__main__":
    main()
