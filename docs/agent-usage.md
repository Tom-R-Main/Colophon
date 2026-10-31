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

## Local profile discovery

Named profiles are a local capability that each user configures. Colophon ships the registry interface and synthetic tests, not author profiles or source texts. Discovery and prompt export do not call a writing provider or require provider credentials.

Create a private JSON registry with explicit aliases and absolute profile paths:

```json
{
  "schema_version": 1,
  "profiles": {
    "my-voice": {"path": "/absolute/private/writer.voice.json"}
  }
}
```

The default registry is `$XDG_CONFIG_HOME/colophon/profiles.json`, or `~/.config/colophon/profiles.json` when XDG is unset. `COLOPHON_PROFILES_FILE` overrides that default; `--registry` overrides both. Paths must be absolute, with optional `~` expansion. No directories are scanned, and no registry is created or changed by these commands.

Registry and profile files may live outside Git checkouts or in untracked, Git-ignored locations such as this project's `outputs/`. Tracked and nonignored files are refused, including paths reached through symlinks. Keep prompt exports private too: they can include source passages. This is a Git publication boundary, not a claim about filesystem encryption or publication rights.

Adapters must preserve the supplied absolute registry path when calling Colophon. Do not resolve or simplify symlinks first: native validation checks every encountered link, including an intermediate link inside a checkout, before following it. Collapsing the path in a wrapper would hide those locations. Consumers may use the canonical `profile_path` returned after successful resolution and must still compare the export receipt with its selected identity.

If file access, Git metadata or safe file-open support cannot be verified, discovery returns `privacy_check_failed`. Check the reported environment and access rather than disabling the checks. The current integration checks exercise macOS and Linux; this is not a claim of qualified Windows registry support.

```bash
colophon voice profiles list
colophon voice profiles resolve my-voice
# Or select a project-local ignored registry explicitly:
colophon voice profiles resolve my-voice --registry "$PWD/outputs/local-profiles/profiles.json"
```

`list` returns configured alias names without reading profile contents. An absent registry returns `status: "not_configured"` and an empty list. `resolve` validates the selected multi-document `VoiceProfile` and returns `status: "resolved"`, its `profile_path`, `profile_id`, SHA-256 of parsed bytes, language and document count. It emits no source passages, author name or trait statements. Invalid/unavailable selections exit nonzero with JSON `error.code` and recovery instructions; do not substitute another writer silently.

After resolving, pass `profile_path` to `voice prompt --task rewrite --mode exemplars -o PRIVATE_PROMPT --receipt`. Before using the guidance, compare the receipt's source ID and SHA-256 with the resolution and recompute the saved prompt hash. A mismatch means the profile or artifact changed; stop and resolve/export again deliberately. A receipt records a past observation, not a lock on the file.

For a README, first establish the repository's actual claims, commands and links. Use the exported guidance with the agent's existing writing tool to produce a separate candidate. Preserve code blocks, identifiers, links, numbers, qualifications and the supported argument, then inspect the diff and run documentation checks. Candidate preservation is a separate check from profile discovery or voice fidelity. The legacy `stylize` command expects a single-document `.analysis.json` `StyleProfile`; it is not the consumer for a multi-document `.voice.json` `VoiceProfile`.

## Evaluate and resume

Save generated outputs as a JSON array or JSONL records with unique `id`, `variant`, `input` and `output`; optional `references`, `exemplars` and `human` ratings add evidence. A runnable fixture evaluation is:

```bash
colophon voice evaluate-writing examples/voice-corpus/writing-cases.json -o writing-results.json
```

For actual outputs, run the same command with your saved cases path. Read voice, preservation flags, copying, naturalness and factuality separately. Missing human ratings remain unscored; character-trigram voice similarity is a topic-sensitive proxy, number/quote preservation flags do not prove factual equivalence, and copying matches need interpretation. Use independent writer/reader review for recognizable voice and literary judgment.

To resume, inspect `rowan.receipt.json`, `rowan.voice.json` and `rowan.prompt.txt`; verify the saved hashes if the files may have changed. Continue external generation from the saved prompt or evaluate the saved outputs. Extraction need not be repeated when its source/configuration is unchanged. Regenerate guidance deliberately if the profile, task or mode changes. A receipt records a past observation; it does not continuously certify a file.

## Adapted principles

This small workflow adapts the Legible Loop's **L1** (state the scope and next move where the agent acts), **L2** (report observed readback), **L5** (bind source identity and refuse source/output aliases), and **L6** (limit completion claims to observed state and retain unverified artifacts). The public CLI supplies the same receipt to a person or an agent, taking inspiration from L9. These are bounded adaptations, not conformance claims, full concurrent-change protection, or measured reductions in operator effort. Footwork's experiments do not establish those outcomes for Colophon.
