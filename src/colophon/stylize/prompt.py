"""Build a system prompt from a Colophon StyleProfile."""

from __future__ import annotations

from colophon.models.features import StyleProfile


def build_style_prompt(
    profile: StyleProfile,
    *,
    examples: list[str] | None = None,
    fingerprint_diff: str | None = None,
) -> str:
    """Convert a StyleProfile into a detailed system prompt for style transfer."""
    sections: list[str] = []

    sections.append(_role_section(profile))
    if examples:
        from colophon.stylize.examples import format_examples_prompt
        sections.append(format_examples_prompt(examples))
    sections.append(_voice_section(profile))
    sections.append(_sentence_section(profile))
    sections.append(_vocabulary_section(profile))
    sections.append(_punctuation_section(profile))
    sections.append(_function_word_section(profile))
    sections.append(_pos_section(profile))
    sections.append(_contraction_section(profile))
    sections.append(_opener_section(profile))
    sections.append(_paragraph_section(profile))
    sections.append(_dialogue_section(profile))
    sections.append(_syntax_section(profile))
    if fingerprint_diff:
        sections.append(fingerprint_diff)
    sections.append(_rules_section())

    return "\n\n".join(s for s in sections if s)


def _role_section(profile: StyleProfile) -> str:
    return f"""<role>
You are a ghostwriter who has deeply studied the writing style of the author behind \
"{profile.document_title}". Your job is to rewrite a given article in this author's \
distinctive voice while preserving the original article's facts, quotes, and reporting. \
You are not summarizing — you are restyling. Every fact from the original must appear \
in your output.
</role>"""


def _voice_section(profile: StyleProfile) -> str:
    r = profile.readability
    if not r:
        return ""

    # Characterize the voice from readability data
    if r.flesch_reading_ease >= 70:
        accessibility = "accessible, conversational, written for a general newspaper audience"
    elif r.flesch_reading_ease >= 50:
        accessibility = "moderately complex, aimed at an educated general reader"
    else:
        accessibility = "dense and literary, with complex sentence constructions"

    grade = r.flesch_kincaid_grade
    return f"""<voice>
The target style reads at a {grade:.0f}th-grade level (Flesch Reading Ease: \
{r.flesch_reading_ease:.0f}/100). The prose is {accessibility}. \
Write at this same level — do not elevate the vocabulary or sentence complexity \
beyond what these metrics indicate. If the original article is written at a higher \
reading level, simplify it down. If lower, you may lightly elevate.
</voice>"""


def _sentence_section(profile: StyleProfile) -> str:
    s = profile.sentences
    if not s:
        return ""

    # Characterize rhythm
    if s.mean_length < 15:
        rhythm = "very short, punchy sentences — machine-gun prose"
    elif s.mean_length < 20:
        rhythm = "crisp, direct sentences with occasional longer constructions for rhythm"
    elif s.mean_length < 25:
        rhythm = "moderate-length sentences with natural variation"
    else:
        rhythm = "long, flowing sentences with embedded clauses"

    skew_note = ""
    if s.skewness > 2:
        skew_note = (
            " The distribution is heavily right-skewed — most sentences are short, "
            "with occasional long ones for emphasis or narrative buildup."
        )
    elif s.skewness > 1:
        skew_note = " There's a moderate right skew — favor shorter sentences but mix in longer ones."

    return f"""<sentence-rhythm>
Target sentence statistics:
- Mean length: {s.mean_length:.1f} words per sentence
- Median length: {s.median_length:.1f} words (the "typical" sentence)
- Range: {s.min_length} to {s.max_length} words
- Standard deviation: {s.stdev_length:.1f} (this controls variety — match it)

The rhythm is {rhythm}.{skew_note}

Vary your sentence lengths to match this distribution. Do NOT write uniformly-sized \
sentences. Mix short punches (5-10 words) with medium builds (15-20) and occasional \
long sweeps (30+). The SHORT sentences carry the most stylistic weight — they land \
the jokes, the observations, the verdicts.
</sentence-rhythm>"""


def _vocabulary_section(profile: StyleProfile) -> str:
    v = profile.vocabulary
    if not v:
        return ""

    if v.ttr < 0.15:
        richness = "deliberately limited, working-class vocabulary — repeating key words for rhythm and emphasis"
    elif v.ttr < 0.25:
        richness = "moderate vocabulary range — accessible but not dumbed down"
    else:
        richness = "rich, varied vocabulary with many unique words"

    return f"""<vocabulary>
Vocabulary profile:
- Type-token ratio: {v.ttr:.4f} ({richness})
- Unique words: {v.unique_types:,} across {v.total_tokens:,} total tokens
- Hapax legomena: {v.hapax_ratio:.1%} of unique words appear only once
- Yule's K: {v.yules_k:.1f} (lower = more diverse vocabulary)

This author uses {richness}. Match this register. Do not reach for SAT words \
or literary flourishes unless the original data supports it. Prefer concrete, \
specific nouns over abstract ones. Prefer active, muscular verbs over passive \
constructions.
</vocabulary>"""


def _punctuation_section(profile: StyleProfile) -> str:
    p = profile.punctuation
    if not p:
        return ""

    habits: list[str] = []

    if p.dash > 3:
        habits.append(
            f"Em-dashes are a signature mark ({p.dash:.1f}/1000 words) — use them "
            "for parenthetical asides, abrupt shifts, and dramatic pauses."
        )
    if p.question > 4:
        habits.append(
            f"Rhetorical questions appear frequently ({p.question:.1f}/1000 words) — "
            "use them to challenge the reader, express mock incredulity, or set up a punchline."
        )
    if p.quotation > 30:
        habits.append(
            f"Heavy use of direct quotation ({p.quotation:.1f}/1000 words) — this author "
            "lets characters speak in their own voice. Use dialogue liberally. Let people "
            "talk. Quote them, then comment."
        )
    if p.exclamation > 2:
        habits.append(f"Exclamation marks appear at {p.exclamation:.1f}/1000 — use sparingly for emphasis.")
    elif p.exclamation < 1:
        habits.append("Exclamation marks are rare — this author lets the content do the shouting.")
    if p.semicolon < 1.5:
        habits.append("Semicolons are almost absent — prefer periods and dashes over semicolons.")
    if p.ellipsis > 1:
        habits.append(f"Ellipses appear at {p.ellipsis:.1f}/1000 — use for trailing off, hesitation, or irony.")

    if not habits:
        return ""

    habit_lines = "\n".join(f"- {h}" for h in habits)
    return f"""<punctuation>
Punctuation fingerprint (rates per 1000 words):
- Commas: {p.comma:.1f} | Periods: {p.period:.1f} | Semicolons: {p.semicolon:.1f}
- Dashes: {p.dash:.1f} | Questions: {p.question:.1f} | Exclamations: {p.exclamation:.1f}
- Quotation marks: {p.quotation:.1f} | Ellipses: {p.ellipsis:.1f}

Key habits:
{habit_lines}
</punctuation>"""


def _function_word_section(profile: StyleProfile) -> str:
    fw = profile.function_words
    if not fw or not fw.frequencies:
        return ""

    # Highlight distinctive patterns
    top_5 = list(fw.frequencies.items())[:5]
    freq_lines = ", ".join(f'"{w}" ({f:.1f})' for w, f in top_5)

    # Check for high first-person usage
    i_freq = fw.frequencies.get("i", 0)
    you_freq = fw.frequencies.get("you", 0)

    person_note = ""
    if i_freq > 10:
        person_note = (
            "This author writes heavily in first person ('I' appears at "
            f"{i_freq:.1f}/1000). The column voice is personal, opinionated, "
            "and unafraid to insert the author as a character in the story."
        )
    if you_freq > 5:
        person_note += (
            f" Direct address ('you' at {you_freq:.1f}/1000) is common — "
            "the author talks TO the reader, not at them."
        )

    return f"""<function-words>
Most frequent function words (per 1000): {freq_lines}
{person_note}

Match these proportions naturally. They reflect unconscious speech patterns \
and are the strongest stylistic fingerprint.
</function-words>"""


def _pos_section(profile: StyleProfile) -> str:
    p = profile.pos
    if not p:
        return ""

    notes: list[str] = []
    if p.adjective_noun_ratio < 0.3:
        notes.append("Low adjective-to-noun ratio — this author trusts nouns to do the work. Don't over-modify.")
    elif p.adjective_noun_ratio > 0.5:
        notes.append("High adjective density — this author paints with modifiers.")

    if p.adverb_density < 4:
        notes.append("Low adverb density — verbs carry their own weight. 'He walked' not 'He walked slowly.'")
    elif p.adverb_density > 6:
        notes.append("Higher adverb usage — qualifiers and manner words are part of the voice.")

    if p.verb_density > 15:
        notes.append("High verb density — the prose is action-oriented and dynamic.")

    if not notes:
        return ""

    note_lines = "\n".join(f"- {n}" for n in notes)
    return f"""<grammar>
Part-of-speech profile:
- Adjective/Noun ratio: {p.adjective_noun_ratio:.3f}
- Adverb density: {p.adverb_density:.1f}%
- Verb density: {p.verb_density:.1f}%

{note_lines}
</grammar>"""


def _contraction_section(profile: StyleProfile) -> str:
    c = profile.contractions
    if not c:
        return ""

    if c.rate_per_1000 > 15:
        tone = (
            "This author uses contractions heavily — the voice is informal and conversational. "
            "Always use contractions: \"don't\" not \"do not\", \"it's\" not \"it is\", "
            "\"can't\" not \"cannot\". Writing without contractions will sound stiff and wrong."
        )
    elif c.rate_per_1000 > 5:
        tone = "Moderate contraction usage — use contractions naturally but not in every possible spot."
    else:
        tone = (
            "This author rarely uses contractions — the voice is more formal. "
            "Prefer \"do not\" over \"don't\", \"it is\" over \"it's\"."
        )

    top_3 = list(c.top_contractions.items())[:3]
    top_str = ", ".join(f'"{w}" ({n}x)' for w, n in top_3) if top_3 else ""

    return f"""<contractions>
Contraction rate: {c.rate_per_1000:.1f} per 1000 words ({c.total:,} total).
Most common: {top_str}

{tone}
</contractions>"""


def _opener_section(profile: StyleProfile) -> str:
    o = profile.sentence_openers
    if not o:
        return ""

    notes: list[str] = []

    if o.conjunction_start_rate > 8:
        notes.append(
            f"{o.conjunction_start_rate:.1f}% of sentences start with a coordinating conjunction "
            "(But, And, So). This is a DEFINING trait. Start many sentences with 'But' or 'And' — "
            "it creates forward momentum and a conversational, rule-breaking energy."
        )
    elif o.conjunction_start_rate > 3:
        notes.append(
            f"{o.conjunction_start_rate:.1f}% of sentences start with conjunctions. "
            "Occasionally start sentences with 'But' or 'And' for emphasis."
        )

    top_5 = list(o.top_words.items())[:5]
    opener_str = ", ".join(f'"{w}" ({n}x)' for w, n in top_5)

    if not notes:
        return ""

    note_lines = "\n".join(f"- {n}" for n in notes)
    return f"""<sentence-openers>
Most common sentence starters: {opener_str}

{note_lines}
</sentence-openers>"""


def _paragraph_section(profile: StyleProfile) -> str:
    p = profile.paragraphs
    if not p:
        return ""

    notes: list[str] = []

    if p.one_sentence_ratio > 0.35:
        notes.append(
            f"{p.one_sentence_ratio:.0%} of paragraphs are a single sentence. "
            "This is a SIGNATURE technique. Use one-sentence paragraphs constantly — "
            "for punchlines, for transitions, for emphasis. A paragraph break is a "
            "beat of silence. Use it like a comedian uses a pause."
        )
    elif p.one_sentence_ratio > 0.15:
        notes.append(
            f"{p.one_sentence_ratio:.0%} of paragraphs are a single sentence. "
            "Use one-sentence paragraphs for emphasis."
        )

    if p.median_length < 30:
        notes.append(
            f"Paragraphs are short — median {p.median_length:.0f} words. "
            "Keep paragraphs tight. Two or three sentences max in most cases. "
            "Long blocks of text are alien to this style."
        )

    if not notes:
        return ""

    note_lines = "\n".join(f"- {n}" for n in notes)
    return f"""<paragraph-structure>
Paragraph statistics:
- Count: {p.count:,} paragraphs
- Mean length: {p.mean_length:.1f} words | Median: {p.median_length:.1f} words
- One-sentence paragraphs: {p.one_sentence_ratio:.0%}

{note_lines}
</paragraph-structure>"""


def _dialogue_section(profile: StyleProfile) -> str:
    d = profile.dialogue
    if not d:
        return ""

    notes: list[str] = []

    if d.quoted_word_ratio > 0.4:
        notes.append(
            f"{d.quoted_word_ratio:.0%} of this author's text is direct quotation. "
            "The author is a LISTENER — they let their subjects speak, then react. "
            "When rewriting, preserve all original quotes. You may rearrange them or "
            "break them up with authorial commentary, but the voices must be heard."
        )
    elif d.quoted_word_ratio > 0.2:
        notes.append(
            f"{d.quoted_word_ratio:.0%} of text is quoted speech. "
            "Use dialogue to break up narration."
        )
    elif d.quoted_word_ratio < 0.1:
        notes.append(
            "Very little quoted speech — this author's voice dominates. "
            "The style is more essayistic than reportorial."
        )

    top_verb = list(d.top_attribution_verbs.items())[0] if d.top_attribution_verbs else None
    if top_verb and top_verb[0] == "say":
        notes.append(
            f'The dominant attribution verb is "said/says" ({top_verb[1]}x) — '
            "invisible, workmanlike attribution. Do NOT use fancy speech verbs "
            '("exclaimed", "opined", "remarked"). Just "said" or "says".'
        )

    if not notes:
        return ""

    note_lines = "\n".join(f"- {n}" for n in notes)
    return f"""<dialogue>
Dialogue ratio: {d.quoted_word_ratio:.0%} quoted speech, {d.narration_word_ratio:.0%} narration.

{note_lines}
</dialogue>"""


def _syntax_section(profile: StyleProfile) -> str:
    s = profile.syntax
    if not s:
        return ""

    notes: list[str] = []

    if s.mean_tree_depth < 5:
        notes.append(
            f"Mean dependency tree depth of {s.mean_tree_depth:.1f} — syntactically simple. "
            "Favor main clauses over nested subordination. Subject-verb-object. "
            "Don't bury the point inside a dependent clause."
        )
    elif s.mean_tree_depth > 6:
        notes.append(
            f"Mean tree depth of {s.mean_tree_depth:.1f} — complex syntax with layered clauses."
        )

    # Tense mix
    past = s.tense_distribution.get("Past", 0)
    pres = s.tense_distribution.get("Pres", 0)
    if past > 0.3 and pres > 0.25:
        notes.append(
            f"Tense splits nearly even: {past:.0%} past, {pres:.0%} present. "
            "This author shifts between storytelling (past tense) and commentary (present tense). "
            "Match this: narrate events in past tense, deliver opinions in present."
        )
    elif past > 0.5:
        notes.append(f"Primarily past tense ({past:.0%}) — narrative, storytelling mode.")
    elif pres > 0.5:
        notes.append(f"Primarily present tense ({pres:.0%}) — immediate, urgent voice.")

    # Sentence type mix
    interrog = s.sentence_type_mix.get("interrogative", 0)
    if interrog > 0.04:
        notes.append(
            f"{interrog:.1%} of sentences are questions — use rhetorical questions "
            "to challenge, provoke, or set up punchlines."
        )

    if not notes:
        return ""

    note_lines = "\n".join(f"- {n}" for n in notes)
    return f"""<syntax>
Syntactic complexity: mean depth {s.mean_tree_depth:.1f}, median {s.median_tree_depth:.1f}

{note_lines}
</syntax>"""


def _rules_section() -> str:
    return """<rules>
- Preserve ALL facts, names, dates, quotes, and data from the original article. Do not invent.
- You may restructure paragraphs, rewrite sentences, change word choices, and add \
rhetorical devices (irony, direct address, mock dialogue) to match the target style.
- You may add a punchy opening hook or a kicker ending if the original lacks one.
- You may add brief authorial commentary or asides in the target voice.
- Do NOT add facts, quotes, or events that weren't in the original.
- Do NOT change the fundamental angle or conclusion of the article.
- Output ONLY the rewritten article. No preamble, no explanation, no meta-commentary.
</rules>"""
