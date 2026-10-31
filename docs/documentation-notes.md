# Documentation decisions

The documentation has three jobs: help someone choose a task, let them complete it, and show what was actually checked. We keep those jobs small and link their owners rather than repeating a full manual in the README.

| Owner | Responsibility | Local check |
|---|---|---|
| README | Scope, prerequisites, one synthetic first run, links to deeper tasks | `tests/test_readme.py` executes the marked commands and checks artifact contents |
| Voice and agent guides | Corpus rules, ordered actions, outputs, receipt scope and recovery | CLI, corpus and receipt scenario tests |
| Installed `--help` | Exact option syntax and defaults | Public CLI parsing and behavior tests |
| Command reference | Route selection and optional dependency boundaries | Links and help reviewed against the actual commands |
| Verification guide | What a check proves and what remains unverified | Promise-to-check map plus the executed suite |

Two reference mechanisms informed this arrangement:

- uv routes its [README](https://github.com/astral-sh/uv/blob/564a2074b02eb9de5aaaa9491ed74357ade0332f/README.md#L78-L118) to a task guide and command help. Its [project guide](https://github.com/astral-sh/uv/blob/564a2074b02eb9de5aaaa9491ed74357ade0332f/docs/guides/projects.md#L12-L73) names the created files and expected output. Colophon uses the same division with a much smaller Markdown surface. We did not adopt uv's reference generator; a generation step alone would not prove committed documentation is current.
- Inspect AI keeps [examples as runnable tasks](https://github.com/UKGovernmentBEIS/inspect_ai/blob/aee6aaae3b915f48e9227007eb3a6c27824982ba/examples/security_guide.py#L1-L17). Its captured [example test](https://github.com/UKGovernmentBEIS/inspect_ai/blob/aee6aaae3b915f48e9227007eb3a6c27824982ba/tests/test_examples.py#L8-L11) illustrates a useful seam, with limited proof: it uses a mock and is skipped in GitHub Actions. Colophon's README smoke test instead checks the saved training IDs, held-out exclusion and receipt hashes. Neither test establishes real-writer quality.

These are source-supported design transfers, not claims that upstream tests were run or that agent effort fell. The source snapshots, counterexamples and reusable cards are local research under ignored `outputs/repo-learning/documentation-library/`. Public documentation contains original explanations and source links, not copied reference-repository excerpts or private corpus material.
