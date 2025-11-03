import typer

from synth.gen_qa_offline import generate_sft_examples, preview_examples, save_sft_dataset

app = typer.Typer(help="Generate synthetic QA data from document chunks.")


@app.command()
def main(
    chunks_path: str = typer.Option("data/processed/chunks.jsonl", "--chunks", "-c"),
    out_path: str = typer.Option("data/processed/sft.jsonl", "--out", "-o"),
    use_api: bool = typer.Option(False, "--use-api", help="Use an external API for question drafting."),
    seed: int = typer.Option(13, "--seed", help="Random seed for shuffling."),
    min_examples: int = typer.Option(100, "--min", help="Minimum number of QA pairs to emit."),
):
    """Generate synthetic QA pairs for supervised fine-tuning."""
    try:
        examples = generate_sft_examples(
            chunks_path,
            use_api=use_api,
            seed=seed,
            min_examples=min_examples,
        )
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    except ValueError as exc:
        typer.echo(f"Generation failed: {exc}", err=True)
        raise typer.Exit(code=1)

    save_sft_dataset(examples, out_path)
    typer.echo(f"Generated {len(examples)} QA pairs → {out_path}")

    for idx, sample in enumerate(preview_examples(examples, n=3), start=1):
        typer.echo(f"Example {idx}: Q: {sample['question']} | A: {sample['answer']}")


if __name__ == "__main__":
    app()
