"""Rich console output for analysis results."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from colophon.models.features import ComparisonResult, StyleProfile

console = Console()


def print_analysis_summary(profile: StyleProfile) -> None:
    """Print a formatted analysis summary to the console."""
    console.print()
    console.print(Panel(
        f"[bold]{profile.document_title}[/bold]\n"
        f"Words: {profile.word_count:,} | Document ID: {profile.document_id}",
        title="Colophon Analysis",
    ))

    if profile.readability:
        _print_readability(profile)
    if profile.sentences:
        _print_sentences(profile)
    if profile.vocabulary:
        _print_vocabulary(profile)
    if profile.function_words:
        _print_function_words(profile)
    if profile.pos:
        _print_pos(profile)
    if profile.punctuation:
        _print_punctuation(profile)
    if profile.ngrams:
        _print_ngrams(profile)


def _print_readability(profile: StyleProfile) -> None:
    r = profile.readability
    if not r:
        return
    table = Table(title="Readability", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Interpretation", style="dim")

    table.add_row("Flesch Reading Ease", f"{r.flesch_reading_ease:.1f}", _flesch_interpretation(r.flesch_reading_ease))
    table.add_row("Flesch-Kincaid Grade", f"{r.flesch_kincaid_grade:.1f}", f"Grade {r.flesch_kincaid_grade:.0f}")
    table.add_row("Gunning Fog", f"{r.gunning_fog:.1f}", f"Grade {r.gunning_fog:.0f}")
    table.add_row("Coleman-Liau", f"{r.coleman_liau:.1f}", f"Grade {r.coleman_liau:.0f}")
    table.add_row("ARI", f"{r.ari:.1f}", f"Grade {r.ari:.0f}")
    if r.smog is not None:
        table.add_row("SMOG", f"{r.smog:.1f}", f"Grade {r.smog:.0f}")
    else:
        table.add_row("SMOG", "N/A", "Needs 30+ sentences")

    console.print(table)


def _flesch_interpretation(score: float) -> str:
    if score >= 90:
        return "Very easy"
    elif score >= 80:
        return "Easy"
    elif score >= 70:
        return "Fairly easy"
    elif score >= 60:
        return "Standard"
    elif score >= 50:
        return "Fairly difficult"
    elif score >= 30:
        return "Difficult"
    return "Very confusing"


def _print_sentences(profile: StyleProfile) -> None:
    s = profile.sentences
    if not s:
        return
    table = Table(title="Sentence Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Count", f"{s.count:,}")
    table.add_row("Mean length", f"{s.mean_length:.1f} words")
    table.add_row("Median length", f"{s.median_length:.1f} words")
    table.add_row("Std deviation", f"{s.stdev_length:.1f}")
    table.add_row("Skewness", f"{s.skewness:.2f}")
    table.add_row("Shortest", f"{s.min_length} words")
    table.add_row("Longest", f"{s.max_length} words")

    console.print(table)

    # Text histogram
    if s.length_distribution:
        console.print("\n[bold]Sentence Length Distribution[/bold] (words per sentence)")
        max_count = max(s.length_distribution) if s.length_distribution else 1
        bar_width = 40
        for i, count in enumerate(s.length_distribution):
            if count == 0:
                continue
            label = f"{i * 5:3d}-{i * 5 + 4:3d}"
            bar_len = int((count / max_count) * bar_width)
            bar = "\u2588" * bar_len
            console.print(f"  {label} | {bar} {count}")


def _print_vocabulary(profile: StyleProfile) -> None:
    v = profile.vocabulary
    if not v:
        return
    table = Table(title="Vocabulary Richness", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Note", style="dim")

    table.add_row("Total tokens", f"{v.total_tokens:,}", "")
    table.add_row("Unique types", f"{v.unique_types:,}", "")
    table.add_row("Type-token ratio", f"{v.ttr:.4f}", "Length-biased")
    table.add_row("Hapax legomena", f"{v.hapax_legomena:,}", f"{v.hapax_ratio:.1%} of types")
    table.add_row("Yule's K", f"{v.yules_k:.2f}", "Length-independent richness")
    if v.honores_r is not None:
        table.add_row("Honore's R", f"{v.honores_r:.2f}", "Rewards rare words")
    else:
        table.add_row("Honore's R", "N/A", "All types are hapax")

    console.print(table)


def _print_function_words(profile: StyleProfile) -> None:
    fw = profile.function_words
    if not fw or not fw.frequencies:
        return
    table = Table(title=f"Top {fw.top_n} Function Words (per 1000 words)", show_header=True)
    table.add_column("Word", style="cyan")
    table.add_column("Frequency", justify="right")
    table.add_column("", min_width=20)

    max_freq = max(fw.frequencies.values()) if fw.frequencies else 1
    for word, freq in fw.frequencies.items():
        bar_len = int((freq / max_freq) * 20)
        bar = "\u2588" * bar_len
        table.add_row(word, f"{freq:.1f}", bar)

    console.print(table)


def _print_pos(profile: StyleProfile) -> None:
    p = profile.pos
    if not p:
        return
    table = Table(title="POS Tag Distribution", show_header=True)
    table.add_column("Tag", style="cyan")
    table.add_column("Proportion", justify="right")
    table.add_column("", min_width=20)

    max_prop = max(p.tag_distribution.values()) if p.tag_distribution else 1
    for tag, prop in list(p.tag_distribution.items())[:12]:
        bar_len = int((prop / max_prop) * 20)
        bar = "\u2588" * bar_len
        table.add_row(tag, f"{prop:.2%}", bar)

    console.print()
    console.print(f"  Adjective/Noun ratio: [bold]{p.adjective_noun_ratio:.3f}[/bold]")
    console.print(f"  Adverb density: [bold]{p.adverb_density:.1f}%[/bold]")
    console.print(f"  Verb density: [bold]{p.verb_density:.1f}%[/bold]")
    console.print(table)


def _print_punctuation(profile: StyleProfile) -> None:
    p = profile.punctuation
    if not p:
        return
    table = Table(title="Punctuation Patterns (per 1000 words)", show_header=True)
    table.add_column("Mark", style="cyan")
    table.add_column("Rate", justify="right")

    table.add_row("Comma", f"{p.comma:.1f}")
    table.add_row("Period", f"{p.period:.1f}")
    table.add_row("Semicolon", f"{p.semicolon:.1f}")
    table.add_row("Colon", f"{p.colon:.1f}")
    table.add_row("Dash (em/en)", f"{p.dash:.1f}")
    table.add_row("Exclamation", f"{p.exclamation:.1f}")
    table.add_row("Question", f"{p.question:.1f}")
    table.add_row("Ellipsis", f"{p.ellipsis:.1f}")
    table.add_row("Parenthesis", f"{p.parenthesis:.1f}")
    table.add_row("Quotation", f"{p.quotation:.1f}")

    console.print(table)


def _print_ngrams(profile: StyleProfile) -> None:
    ng = profile.ngrams
    if not ng:
        return
    # Show top 15 word bigrams
    if ng.word_bigrams:
        table = Table(title="Top 15 Word Bigrams", show_header=True)
        table.add_column("Bigram", style="cyan")
        table.add_column("Count", justify="right")
        for bigram, count in list(ng.word_bigrams.items())[:15]:
            table.add_row(bigram, f"{count:,}")
        console.print(table)


def print_comparison_summary(result: ComparisonResult) -> None:
    """Print authorship comparison results."""
    console.print()
    console.print(Panel(
        f"[bold]Unknown:[/bold] {result.unknown_document_title}\n"
        f"Method: {result.method} | Features: {result.n_features}",
        title="Author Comparison",
    ))

    table = Table(show_header=True)
    table.add_column("Rank", justify="right", style="bold")
    table.add_column("Author", style="cyan")
    table.add_column("Delta Score", justify="right")
    table.add_column("", min_width=20)

    if result.ranked_authors:
        max_delta = max(d for _, d in result.ranked_authors)
        for i, (author, delta) in enumerate(result.ranked_authors):
            bar_len = int((1 - delta / max_delta) * 20) if max_delta > 0 else 20
            bar = "\u2588" * bar_len
            style = "bold green" if i == 0 else ""
            table.add_row(str(i + 1), f"[{style}]{author}[/{style}]" if style else author, f"{delta:.4f}", bar)

    console.print(table)
    if result.ranked_authors:
        console.print(f"\n  [bold green]Most likely author: {result.ranked_authors[0][0]}[/bold green]")
