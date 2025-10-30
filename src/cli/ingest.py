import typer
from rag_t5.config import load_settings
from rag_t5.data.ingest import ingest_dir

app = typer.Typer()


@app.command()
def main(
    input_dir: str = typer.Option("data/raw", "--input-dir", "-i"),
    output: str = typer.Option("data/processed/docs.jsonl", "--output", "-o"),
    index_out: str = typer.Option("artifacts/models/docs_index.json", "--index-out"),
    min_chars: int = typer.Option(100, "--min-chars"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive"),
    config: str = typer.Option("configs/defaults.toml", "--config", "-c"),
    override: str = typer.Option(None, "--override", "-O"),
):
    # Keep the pattern consistent (even if cfg not needed yet)
    _ = load_settings(config, override)
    stats = ingest_dir(input_dir, output, index_out, min_chars=min_chars, recursive=recursive)
    typer.echo(f"Ingest complete → {stats}")


if __name__ == "__main__":
    app()
