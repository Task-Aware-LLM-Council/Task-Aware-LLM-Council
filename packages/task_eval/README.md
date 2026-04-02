# task_eval

`task_eval` is the reusable evaluation layer for benchmark tasks. It owns:

- dataset-specific row normalization
- conversion into benchmark cases
- response answer extraction
- reference normalization
- metric calculation

It is designed to be used by `benchmark_runner`, but the dataset profiles and
scoring helpers are reusable outside that package as well.

## Public API

- `DatasetProfile`
- `MetricCalculator`
- `EvaluationCase`
- `MetricResult`
- `get_dataset_profile(...)`
- `list_dataset_profiles()`
- dataset profiles:
  - `MusiqueProfile`
  - `QualityProfile`
  - `FeverProfile`
  - `HardMathProfile`
  - `HumanEvalPlusProfile`

## Current Dataset Profiles

- `musique`
- `quality`
- `fever`
- `hardmath`
- `humaneval_plus`

Each profile:

- streams dataset rows with `load_dataset(..., streaming=True)` by default
- converts rows into `EvaluationCase`
- extracts dataset-aware answers from raw model output
- computes normalized metric values

## Integration

`benchmark_runner` can use these profiles directly because they provide both:

- `iter_cases()`
- `score(...)`

That means a dataset profile can act as both the case source and the metric
calculator for one dataset.
