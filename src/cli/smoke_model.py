import torch
import typer

from rag_t5.models.loader import load_base
from rag_t5.models.lora import LoraArgs, apply_lora
from rag_t5.utils.seed import set_seed

app = typer.Typer()


@app.command()
def main(model_name: str = "google/flan-t5-small") -> None:
    set_seed(42)
    model, tok, device = load_base(model_name)
    model = apply_lora(model, LoraArgs())  # MAX profile
    # Show trainable params percentage (should be small)
    model.print_trainable_parameters()

    prompt = "translate English to German: The house is wonderful."
    inputs = tok(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32)
    print("\n=== OUTPUT ===")
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    app()
