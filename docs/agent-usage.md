# Agent workflow

Run from the repository root with an already prepared Colophon environment. Check readiness and validate the local corpus before extracting a profile:

```bash
colophon doctor
colophon corpus validate examples/voice-corpus/corpus.json
colophon voice extract examples/voice-corpus/corpus.json --author "Rowan (synthetic)" -o rowan.voice.json
colophon voice inspect rowan.voice.json --evidence
colophon voice prompt rowan.voice.json --task generate --mode exemplars -o rowan.prompt.txt --receipt > rowan.receipt.json
```

Doctor reports package presence; it does not install anything or establish parser or voice quality. Inspect source ownership, splits, exclusions, evidence and warnings before using a real writer's profile. The synthetic example demonstrates the commands, not literary validity.

`--receipt` requires `-o`. A successful command emits one JSON object to stdout. `status: "written"` means the saved prompt was read back and matched the rendered bytes exactly. `target_path` names the absolute output path; `artifact.bytes` and `artifact.sha256` describe the readback. `source_profile.id` and `source_profile.sha256` bind the receipt to the profile bytes actually parsed. `mode` and `task` record the requested export. Verification covers **prompt artifact only**. Voice, naturalness, factuality, preservation, copying and human or literary validation stay explicitly `unverified`.

Exporting never calls a writing model. The next step is to supply the saved prompt, a brief and supported facts to an external generation tool under the user's provider authorization. Keep the resulting output and the exact provider/model setup for evaluation. A receipt is no evidence that generation occurred. Without `--receipt`, stdout remains the raw prompt when `-o` is omitted and successful file output remains silent.

Validation, source-read and source/output-alias failures return `not_written`, a nonzero exit status and recovery guidance before writing. Choose a separate output path; aliases through symlinks and hardlinks are also refused. After replacement, an unreadable readback reports `unknown` and a mismatch reports `unverified`. The artifact is retained and the command does not retry. Inspect that artifact and compare it with the source before deciding whether to rerun; do not infer that no file was written.

## Evaluate and resume

Save generated outputs as a JSON array or JSONL records with unique `id`, `variant`, `input` and `output`; optional `references`, `exemplars` and `human` ratings add evidence. A runnable fixture evaluation is:

```bash
colophon voice evaluate-writing examples/voice-corpus/writing-cases.json -o writing-results.json
```

For actual outputs, run the same command with your saved cases path. Read voice, preservation flags, copying, naturalness and factuality separately. Missing human ratings remain unscored; character-trigram voice similarity is a topic-sensitive proxy, number/quote preservation flags do not prove factual equivalence, and copying matches need interpretation. Use independent writer/reader review for recognizable voice and literary judgment.

To resume, inspect `rowan.receipt.json`, `rowan.voice.json` and `rowan.prompt.txt`; verify the saved hashes if the files may have changed. Continue external generation from the saved prompt or evaluate the saved outputs. Extraction need not be repeated when its source/configuration is unchanged. Regenerate guidance deliberately if the profile, task or mode changes. A receipt records a past observation; it does not continuously certify a file.

## Adapted principles

This small workflow adapts the Legible Loop's **L1** (state the scope and next move where the agent acts), **L2** (report observed readback), **L5** (bind source identity and refuse source/output aliases), and **L6** (limit completion claims to observed state and retain unverified artifacts). The public CLI supplies the same receipt to a person or an agent, taking inspiration from L9. These are bounded adaptations, not conformance claims, full concurrent-change protection, or measured reductions in operator effort. Footwork's experiments do not establish those outcomes for Colophon.
