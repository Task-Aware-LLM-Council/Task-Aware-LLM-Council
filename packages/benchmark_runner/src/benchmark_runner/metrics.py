from task_eval import MetricCalculator as Metric
from task_eval import MetricResult
from task_eval.scoring import aggregate_numeric_metrics as aggregate_score_records


class NullMetric:
    name = "null_metric"

    def score(self, *, case, prediction, dataset_metadata=None):
        return MetricResult(values={})
