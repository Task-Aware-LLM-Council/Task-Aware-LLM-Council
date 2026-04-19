# Policy Runner Notes

## Problem
The policies in `packages/council_policies` were written in different shapes:

- some were already close to `PromptRequest -> response`
- some mixed policy logic with orchestration flow
- `p2` was still more dataset-oriented than prompt-oriented

That made it hard to build one clean benchmarking path that could:

- load prompts from a unified data pipeline
- run all policies the same way
- return results in one consistent format
- plug metric calculation on top later

There was also an important runtime constraint:

- specialist models and synthesizer models cannot be loaded on GPU at the same time
- loading/unloading them repeatedly per example or per policy is too expensive

## Mitigation
The mitigation is to separate **policy decision logic** from **benchmark execution flow**.

The new layer does this:

1. Run all specialist calls first for the whole batch.
2. Cache those specialist responses.
3. Tear down the specialist orchestrator.
4. Start the synthesizer orchestrator only after that.
5. Let policies that need synthesis consume the cached specialist outputs.

This keeps GPU usage in two phases and avoids repeated reloads.

## Approach Taken
We added an additive runner layer in `packages/council_policies`:

- `PolicyRuntime`
- `CouncilBenchmarkRunner`
- prompt-level adapters for `p2`, `p3`, and `p4`

These adapters make each policy look like a simple:

- input: `PromptRequest`
- output: `PromptResponse`

The runner also preserves per-policy metadata so downstream metrics and analysis can still inspect routing, voting, synthesis usage, and cache provenance.

## Important Boundary
This work does **not** rewrite policy logic itself.

It is intentionally lightweight:

- existing policy entrypoints remain available
- existing policy behavior is preserved
- this mainly streamlines how policies are executed in benchmark runs

So the goal is not to change what the policies decide.  
The goal is to create a clean execution surface so a unified dataloader pipeline and metric calculator can be built on top.

## Result
The repo now has a common benchmark-facing path where:

- all policies can be run from the same request format
- all policies return the same response format
- specialist outputs can be cached and reused
- synthesis happens only after specialist phase is complete
- metrics can be added later without further policy-specific orchestration hacks
