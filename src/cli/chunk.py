import typer
from rag_t5.config import load_settings
from rag_t5.data.chunk import chunk_docs

app = typer.Typer()


@app.command()
def main(
    docs_path: str = typer.Option("data/processed/docs.jsonl", "--docs", "-d"),
    out_path: str = typer.Option("data/processed/chunks.jsonl", "--out", "-o"),
    config: str = typer.Option("configs/defaults.toml", "--config", "-c"),
    override: str = typer.Option(None, "--override", "-O"),
):
    cfg = load_settings(config, override)
    stats = chunk_docs(docs_path, out_path, cfg)
    typer.echo(f"Chunking complete → {stats}")


if __name__ == "__main__":
    app()
