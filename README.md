# Colophon

A program can describe a writer's voice quite convincingly without learning much about it. The description might fit anyone who writes short sentences. Or it might capture what the writer talks about, which stops being useful when the subject changes.

Colophon gives those descriptions something to answer to. It measures independent pieces separately, keeps the passages behind its observations, and lets the writer correct them. You can inspect the resulting profile, export it, and test it on writing that wasn't used to build it.

The useful unit is an independent piece. Ten chunks of one essay don't give you ten examples of a writer's habits.

## Try it

Use Python 3.12 for the verified development environment. The package declares Python 3.10 and later; the current checks run on 3.12.

```bash
git clone https://github.com/Tom-R-Main/Colophon.git
cd Colophon
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[analysis,ingest]"
python -m spacy download en_core_web_sm
```

The repository includes a small synthetic corpus, so you can try the whole profile workflow without supplying anyone's writing or configuring a model API:

<!-- test:quickstart -->
```bash
colophon doctor
colophon corpus validate examples/voice-corpus/corpus.json
colophon voice extract examples/voice-corpus/corpus.json --author "Rowan (synthetic)" -o outputs/demo/writer.voice.json
colophon voice inspect outputs/demo/writer.voice.json
colophon voice export outputs/demo/writer.voice.json -o outputs/demo/writer.voice.md
colophon voice prompt outputs/demo/writer.voice.json --task generate --mode exemplars -o outputs/demo/writer.prompt.txt --receipt
```
<!-- /test:quickstart -->

You now have a JSON profile, a readable evidence report, and a prompt you can give to a writing model. The last command returns a JSON receipt with the saved file's path and hash. It checks that the prompt was written correctly. It hasn't generated an essay or established that one would sound like the writer.

The synthetic writers test the machinery. They aren't a voice-quality benchmark.

## What goes into a profile

A profile keeps several kinds of evidence together:

- **Measurements:** sentence and paragraph distributions, function words, punctuation, vocabulary, contractions, syntax and other linguistic features, with their availability and extraction versions.
- **Source boundaries:** independent works, language, genre, topic, quotations, translators and editors.
- **Observations:** literal structural markers, plus optional rhetorical interpretations supplied as annotations with exact supporting passages and exceptions.
- **Writer corrections:** accepted, rejected or replacement guidance, with its provenance retained.

A question mark is observable. Whether a question opens an investigation is an interpretation. Keeping that distinction visible makes the profile easier to challenge and improve.

Start with the [voice toolchain guide](docs/voice-toolchain.md) to prepare your own corpus. Review extraction boundaries before trusting paragraph statistics, especially with PDFs. Keep revisions and excerpts of one work in the same evaluation split.

## Test what the profile can do

Colophon exports four prompt modes: `numeric`, `exemplars`, `qualitative` and `hybrid`. They let you test what measurements, examples and interpretations contribute. Combining them doesn't guarantee a better result.

For new writing, use `voice prompt --task generate`. The default task remains `rewrite`. Exporting either prompt stays local; generation happens in your chosen writing tool.

Judge generated work on separate questions: does it resemble the voice, preserve the facts, read naturally, and avoid copying? A good answer to one doesn't settle the others. Numerical similarity is a diagnostic, not an authorship probability.

The [evaluation guide](docs/voice-toolchain.md#evaluate-on-unseen-pieces) covers held-out profiles and supplied writing outputs. Missing judgments stay unscored. Literary fidelity still needs readers; number and quotation checks don't establish factual equivalence.

## Keep the writing private

Profiles and prompts can contain source passages. Treat them as carefully as the originals. Keep real writing, annotations, extracted text, generated essays and experiments under `corpus/` or `outputs/`; both are ignored by Git. Only the small synthetic examples belong in the public test corpus.

Before committing, run the publication check described in [contributing](CONTRIBUTING.md). Ignore rules don't remove files that Git already tracks, and they don't decide whether you have permission to publish a passage.

## Other ways to use it

The document tools remain available: ingest a PDF, EPUB, HTML or text file; measure it; produce a report; or compare it with other documents. Optional rewriting, PostgreSQL vector search and a local web interface have their own dependencies.

See the [command reference](docs/commands.md) for those routes. English is the currently checked NLP path. Language configuration, a working parser, and a qualified voice model are different things.

## Documentation

- [Voice toolchain](docs/voice-toolchain.md): prepare sources, extract a profile, apply corrections and evaluate it.
- [Agent workflow](docs/agent-usage.md): machine-readable receipts, uncertainty and resuming from saved artifacts.
- [Command reference](docs/commands.md): document analysis and optional integrations.
- [Verification](docs/verification.md): executable checks and what they establish.
- [Contributing](CONTRIBUTING.md): development, documentation and publication boundaries.

## License

The code is [MIT licensed](LICENSE). That license doesn't grant rights to writing you supply.
