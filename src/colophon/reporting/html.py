"""HTML report generation with Plotly charts."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from colophon.models.features import StyleProfile


def generate_report(profile: StyleProfile, output: Path, *, standalone: bool = False) -> None:
    """Generate a self-contained HTML report with interactive Plotly charts."""
    from jinja2 import Template

    plotly_js = True if standalone else "cdn"

    # Build charts
    readability_chart = _readability_chart(profile, plotly_js) if profile.readability else None
    sentence_chart = _sentence_chart(profile, plotly_js) if profile.sentences else None
    punctuation_chart = _punctuation_chart(profile, plotly_js) if profile.punctuation else None
    function_word_chart = _function_word_chart(profile, plotly_js) if profile.function_words else None
    pos_chart = _pos_chart(profile, plotly_js) if profile.pos else None

    # Load template
    template_text = resources.files("colophon.reporting.templates").joinpath("report.html").read_text()
    template = Template(template_text)

    ngram_items = list(profile.ngrams.word_bigrams.items())[:20] if profile.ngrams else []

    html = template.render(
        title=profile.document_title,
        word_count=profile.word_count,
        readability=profile.readability,
        sentences=profile.sentences,
        vocabulary=profile.vocabulary,
        punctuation=profile.punctuation,
        ngrams=profile.ngrams,
        readability_chart=readability_chart,
        sentence_chart=sentence_chart,
        punctuation_chart=punctuation_chart,
        function_word_chart=function_word_chart,
        pos_chart=pos_chart,
        ngram_items=ngram_items,
        version="0.1.0",
        computed_at=profile.computed_at.strftime("%Y-%m-%d %H:%M"),
    )

    output.write_text(html)


def _chart_html(fig: object, include_plotlyjs: bool | str) -> str:
    """Convert a Plotly figure to embeddable HTML."""
    return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)  # type: ignore[union-attr]


def _dark_layout() -> dict:
    """Common dark theme layout settings."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e4e4e4", size=12),
        margin=dict(l=40, r=20, t=40, b=40),
    )


def _readability_chart(profile: StyleProfile, plotly_js: bool | str) -> str:
    import plotly.graph_objects as go

    r = profile.readability
    if not r:
        return ""

    categories = ["Flesch-Kincaid", "Gunning Fog", "Coleman-Liau", "ARI"]
    values = [r.flesch_kincaid_grade, r.gunning_fog, r.coleman_liau, r.ari]
    if r.smog is not None:
        categories.append("SMOG")
        values.append(r.smog)

    fig = go.Figure(go.Bar(
        x=categories,
        y=values,
        marker_color=["#e94560", "#533483", "#4ecca3", "#0f3460", "#e94560"][:len(categories)],
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        **_dark_layout(),
        yaxis_title="Grade Level",
        showlegend=False,
        height=350,
    )
    return _chart_html(fig, plotly_js)


def _sentence_chart(profile: StyleProfile, plotly_js: bool | str) -> str:
    import plotly.graph_objects as go

    s = profile.sentences
    if not s or not s.length_distribution:
        return ""

    bin_labels = [f"{i * 5}-{i * 5 + 4}" for i in range(len(s.length_distribution))]

    fig = go.Figure(go.Bar(
        x=bin_labels,
        y=s.length_distribution,
        marker_color="#4ecca3",
        marker_line_width=0,
    ))
    fig.update_layout(
        **_dark_layout(),
        xaxis_title="Words per Sentence",
        yaxis_title="Count",
        height=350,
    )
    # Add mean/median lines
    mean_bin = int(s.mean_length / 5)
    median_bin = int(s.median_length / 5)
    fig.add_vline(x=mean_bin, line_dash="dash", line_color="#e94560",
                  annotation_text=f"Mean: {s.mean_length:.1f}")
    fig.add_vline(x=median_bin, line_dash="dot", line_color="#533483",
                  annotation_text=f"Median: {s.median_length:.1f}")
    return _chart_html(fig, False)  # plotly.js already loaded


def _punctuation_chart(profile: StyleProfile, plotly_js: bool | str) -> str:
    import plotly.graph_objects as go

    p = profile.punctuation
    if not p:
        return ""

    marks = ["Comma", "Period", "Semicolon", "Colon", "Dash", "Exclamation", "Question", "Ellipsis", "Quotes"]
    values = [p.comma, p.period, p.semicolon, p.colon, p.dash, p.exclamation, p.question, p.ellipsis, p.quotation]

    fig = go.Figure(go.Bar(
        x=marks,
        y=values,
        marker_color="#e94560",
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        **_dark_layout(),
        yaxis_title="Per 1000 Words",
        showlegend=False,
        height=300,
    )
    return _chart_html(fig, False)


def _function_word_chart(profile: StyleProfile, plotly_js: bool | str) -> str:
    import plotly.graph_objects as go

    fw = profile.function_words
    if not fw or not fw.frequencies:
        return ""

    # Reverse for horizontal bar chart (top at top)
    words = list(reversed(list(fw.frequencies.keys())))
    freqs = list(reversed(list(fw.frequencies.values())))

    fig = go.Figure(go.Bar(
        y=words,
        x=freqs,
        orientation="h",
        marker_color="#4ecca3",
        text=[f"{v:.1f}" for v in freqs],
        textposition="outside",
    ))
    fig.update_layout(
        **_dark_layout(),
        xaxis_title="Per 1000 Words",
        height=max(400, len(words) * 22),
        showlegend=False,
    )
    return _chart_html(fig, False)


def _pos_chart(profile: StyleProfile, plotly_js: bool | str) -> str:
    import plotly.graph_objects as go

    p = profile.pos
    if not p or not p.tag_distribution:
        return ""

    tags = list(p.tag_distribution.keys())[:10]
    proportions = [p.tag_distribution[t] * 100 for t in tags]

    colors = ["#e94560", "#533483", "#4ecca3", "#0f3460", "#e94560",
              "#533483", "#4ecca3", "#0f3460", "#e94560", "#533483"]

    fig = go.Figure(go.Pie(
        labels=tags,
        values=proportions,
        hole=0.4,
        marker=dict(colors=colors[:len(tags)]),
        textinfo="label+percent",
        textfont_size=11,
    ))
    fig.update_layout(
        **_dark_layout(),
        height=400,
        showlegend=True,
        legend=dict(font=dict(size=11)),
    )
    return _chart_html(fig, False)
