"""Colophon CLI - stylometric and linguistic pattern analysis."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="colophon",
    help="Stylometric and linguistic pattern analysis CLI.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="Path to a PDF, EPUB, TXT, or MD file."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path for the ingested document JSON."),
    no_segment: bool = typer.Option(False, "--no-segment", help="Treat the entire document as one segment."),
    segment_pattern: Optional[str] = typer.Option(
        None, "--segment-pattern", help="Regex pattern for article segmentation."
    ),
) -> None:
    """Ingest a document and extract normalized text."""
    from colophon.ingestion import ingest as do_ingest

    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(1)

    doc = do_ingest(path, no_segment=no_segment, segment_pattern=segment_pattern)

    out_path = output or path.with_suffix(".colophon.json")
    out_path.write_text(doc.model_dump_json(indent=2))

    console.print(f"[green]Ingested:[/green] {path.name}")
    console.print(f"  Segments: {len(doc.segments)}")
    console.print(f"  Words: {len(doc.text.split()):,}")
    console.print(f"  Saved to: {out_path}")


@app.command()
def analyze(
    path: Path = typer.Argument(..., help="Path to a file or .colophon.json."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path for analysis JSON."),
    analyzers: Optional[str] = typer.Option(None, "--analyzers", help="Comma-separated list of analyzers to run."),
    model: str = typer.Option("en_core_web_sm", "--model", help="spaCy model to use."),
    lang: str = typer.Option("en", "--lang", help="Language code (en, fr, de, es, it, nl, pl, ru, ja, ko, zh, ...)."),
) -> None:
    """Run stylometric analysis on a document."""
    from colophon.analysis.pipeline import analyze as do_analyze
    from colophon.models.document import Document
    from colophon.reporting.console import print_analysis_summary

    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(1)

    if path.suffix == ".json":
        doc = Document.model_validate_json(path.read_text())
    else:
        from colophon.ingestion import ingest as do_ingest
        doc = do_ingest(path)

    analyzer_list = analyzers.split(",") if analyzers else None
    profile = do_analyze(doc, analyzers=analyzer_list, spacy_model=model, lang=lang)

    out_path = output or path.with_suffix(".analysis.json")
    out_path.write_text(profile.model_dump_json(indent=2))

    print_analysis_summary(profile)
    console.print(f"\n[dim]Saved to: {out_path}[/dim]")


@app.command()
def compare(
    corpus: Path = typer.Argument(..., help="Directory with subdirectories per author."),
    unknown: Path = typer.Argument(..., help="File to identify."),
    features: int = typer.Option(300, "--features", "-n", help="Number of features for Burrows' Delta."),
    model: str = typer.Option("en_core_web_sm", "--model", help="spaCy model to use."),
) -> None:
    """Compare an unknown text against a corpus of known authors."""
    from colophon.comparison import compare as do_compare
    from colophon.reporting.console import print_comparison_summary

    if not corpus.is_dir():
        console.print(f"[red]Corpus directory not found:[/red] {corpus}")
        raise typer.Exit(1)
    if not unknown.exists():
        console.print(f"[red]File not found:[/red] {unknown}")
        raise typer.Exit(1)

    result = do_compare(corpus, unknown, n_features=features, spacy_model=model)
    print_comparison_summary(result)


@app.command()
def report(
    path: Path = typer.Argument(..., help="Path to an .analysis.json file."),
    output: Path = typer.Option("report.html", "--output", "-o", help="Output path for the report."),
    format: str = typer.Option("html", "--format", "-f", help="Output format: html or json."),
    standalone: bool = typer.Option(False, "--standalone", help="Embed Plotly.js for offline use (~3MB)."),
) -> None:
    """Generate a visualization report from analysis results."""
    from colophon.models.features import StyleProfile

    if not path.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(1)

    profile = StyleProfile.model_validate_json(path.read_text())

    if format == "json":
        from colophon.reporting.json_export import export_json
        export_json(profile, output)
    else:
        from colophon.reporting.html import generate_report
        generate_report(profile, output, standalone=standalone)

    console.print(f"[green]Report saved to:[/green] {output}")


@app.command()
def stylize(
    article: Path = typer.Argument(..., help="Path to the article to restyle (TXT, MD, PDF)."),
    style: Path = typer.Argument(..., help="Path to an .analysis.json (the target style)."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path for the restyled article."),
    source: Optional[Path] = typer.Option(
        None, "--source", "-s", help="Path to source .colophon.json for few-shot examples."
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="LLM provider: anthropic, openai, gemini, openrouter."
    ),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", help="Model name (auto-detected if omitted)."),
    no_examples: bool = typer.Option(False, "--no-examples", help="Skip few-shot examples from source text."),
    no_diff: bool = typer.Option(False, "--no-diff", help="Skip fingerprint diff analysis."),
    show_prompt: bool = typer.Option(False, "--show-prompt", help="Print the generated system prompt and exit."),
) -> None:
    """Rewrite an article in a target author's style using AI."""
    from colophon.models.features import StyleProfile
    from colophon.stylize.prompt import build_style_prompt

    if not article.exists():
        console.print(f"[red]File not found:[/red] {article}")
        raise typer.Exit(1)
    if not style.exists():
        console.print(f"[red]Style profile not found:[/red] {style}")
        raise typer.Exit(1)

    # Load the style profile
    profile = StyleProfile.model_validate_json(style.read_text())

    # Load the article text
    if article.suffix == ".json":
        from colophon.models.document import Document
        doc = Document.model_validate_json(article.read_text())
        article_text = doc.text
    elif article.suffix in {".pdf", ".epub"}:
        from colophon.ingestion import ingest as do_ingest
        doc = do_ingest(article)
        article_text = doc.text
    else:
        article_text = article.read_text(encoding="utf-8")

    # Few-shot examples from source text
    examples: list[str] | None = None
    if not no_examples and source and source.exists():
        from colophon.models.document import Document
        from colophon.stylize.examples import select_examples
        source_doc = Document.model_validate_json(source.read_text())
        examples = select_examples(source_doc, profile)
        if examples:
            console.print(f"[dim]Selected {len(examples)} example passages from source text[/dim]")

    # Fingerprint diff
    fingerprint_diff: str | None = None
    if not no_diff:
        try:
            from colophon.analysis.pipeline import analyze as do_analyze
            from colophon.models.document import Document
            from colophon.stylize.diff import compute_fingerprint_diff
            input_doc = Document.from_text(text=article_text, source_path=str(article))
            input_profile = do_analyze(input_doc, analyzers=["sentences", "contractions", "paragraphs",
                                                              "punctuation", "sentence_openers", "dialogue"])
            fingerprint_diff = compute_fingerprint_diff(input_profile, profile)
            if fingerprint_diff:
                console.print("[dim]Computed style delta between input and target[/dim]")
        except Exception:
            pass  # Fingerprint diff is best-effort

    system_prompt = build_style_prompt(profile, examples=examples, fingerprint_diff=fingerprint_diff)

    if show_prompt:
        console.print(system_prompt)
        raise typer.Exit(0)

    console.print(f"[bold]Style:[/bold] {profile.document_title}")
    console.print(f"[bold]Article:[/bold] {article.name} ({len(article_text.split()):,} words)")

    from colophon.stylize.llm import stylize_text
    from colophon.stylize.providers import get_provider
    llm = get_provider(provider, model=llm_model)
    console.print(f"[dim]Sending to {llm.name} ({llm.model})...[/dim]")

    result = stylize_text(system_prompt, article_text, provider=provider, model=llm_model)

    out_path = output or article.with_suffix(".restyled.md")
    out_path.write_text(result)

    console.print(f"\n[green]Restyled article saved to:[/green] {out_path}")
    console.print(f"[dim]Original: {len(article_text.split()):,} words | "
                  f"Restyled: {len(result.split()):,} words[/dim]")
