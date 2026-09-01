# Artifacts

Shareable HTML reports for supervisor updates and thesis figures. One file per
milestone; the data in each is copied from that milestone's `results/*/results.json`.

| File | Milestone | Live page |
|---|---|---|
| `m2-baseline.html` | M2 — FedAvg LoRA baseline on SST-2 (85.21%, S=2.9583 MB, 591.7 MB) | https://claude.ai/code/artifact/da7bce2c-43b5-4299-8a57-99abfa9c7032 |

## Notes

- These files are **artifact sources**: the published page wraps them in a
  `<!doctype html><head>…</head><body>` skeleton, so they intentionally start at
  `<title>` with no `<html>`/`<body>` tags of their own. Opening one directly in a
  browser still renders (browsers insert the missing structure), but the published
  page is the canonical view.
- Fonts load from Google Fonts; everything else — data, CSS, chart code — is inline,
  so a file is self-contained and diffable.
- **Updating a page keeps its URL** only if the same artifact URL is targeted. Ask
  Claude Code to republish using the URL in the table above; publishing without it
  creates a second, unrelated page.
- Numbers here must match the corresponding `results.json`. If a run is repeated,
  update both or the report silently drifts from the evidence.
