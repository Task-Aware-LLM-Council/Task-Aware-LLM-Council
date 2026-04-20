# Benchmarking Pipeline TODO

- Define normalized benchmark inputs around `BenchmarkExample` and `BenchmarkDataset`.
- Use `llm_gateway.ProviderConfig` directly instead of duplicating provider settings.
- Convert normalized examples into `llm_gateway.PromptRequest` with traceable request metadata.
- Run dataset-model pairs through one reusable gateway client per run.
- Persist a run manifest with config snapshot and dataset inventory.
- Persist per-example predictions in deterministic JSONL files under one run directory.
- Capture latency, usage, request id, finish reason, provider, and prompt version per prediction.
- Keep raw provider payload persistence behind a config flag.
- Support bounded concurrency for provider calls.
- Continue after per-example failures and record structured error payloads.
- Skip already recorded example ids when resuming a named run.
- Keep dataset loading out of scope for this package.
- Keep evaluation, aggregation, ranking, and council logic out of scope for this package.
- Cover prompt building, persistence, partial failures, and resume behavior with unit tests.
