import torch
import typer

from rag_t5.config import load_settings
from rag_t5.models.loader import load_base
from rag_t5.models.lora import LoraArgs, apply_lora
from rag_t5.utils.seed import set_seed

app = typer.Typer()


@app.command()
def main(
    config: str = typer.Option("configs/defaults.toml", "--config", "-c"),
    override: str | None = typer.Option(None, "--override", "-o"),
    adapter: str | None = typer.Option(
        None,
        "--adapter",
        "-a",
        help="Optional path to LoRA adapter weights",
    ),
    notebook_file: str | None = typer.Option(  # noqa: ARG001
        None,
        "--file",
        "-f",
        hidden=True,
        help="Compatibility shim for IPython which injects a -f argument.",
    ),
) -> None:
    set_seed(42)
    cfg = load_settings(config, override)

    adapter_path = adapter or cfg.lora.adapter_path
    if not adapter_path:
        raise typer.BadParameter(
            "Provide the path to the fine-tuned LoRA adapter via --adapter or the config.",
            param_name="adapter",
        )

    model, tok, device = load_base(cfg.base_model)
    try:
        model = apply_lora(
            model,
            LoraArgs(
                r=cfg.lora.r,
                alpha=cfg.lora.alpha,
                dropout=cfg.lora.dropout,
                target_modules=cfg.lora.target_modules,
                adapter_path=adapter_path,
            ),
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc), param_name="adapter") from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_name="adapter") from exc
    model.print_trainable_parameters()

    prompt = "translate English to German: The house is wonderful."
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=cfg.serve.max_new_tokens)
    print("\n=== OUTPUT ===")
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    app()
