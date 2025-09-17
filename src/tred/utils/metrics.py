from typing import Callable, Sequence

from numpy import searchsorted
from sklearn.metrics import roc_auc_score, roc_curve


def recall_at_fpr(labels: Sequence, scores: Sequence, max_fpr: float) -> float:
    if not 0.0 < max_fpr < 1.0:
        raise ValueError(f"`max_fpr` should be in the ]0,1[ range, but got {max_fpr}.")
    fpr, tpr, _ = roc_curve(labels, scores)
    # Get index of smallest fpr value that is strictly larger than max_fpr
    idx = searchsorted(fpr, max_fpr, "right")
    # Since fpr list always starts with 0 and ends with 1, idx will be in [1, len(fpr)-1]
    return tpr[idx - 1].item()


METRIC_MAPPING: dict[str, Callable] = {
    "recall_at_fpr": recall_at_fpr,
    "roc_auc_score": roc_auc_score,
}


def metric_wrapper(metric_name: str, **kwargs) -> Callable[[Sequence, Sequence], float]:
    """Returns a metric function with pre-applied keyword arguments."""
    metric = METRIC_MAPPING.get(metric_name)

    if metric is None:
        raise ValueError(
            f"Unknown metric: '{metric_name}'. Available metrics: {list(METRIC_MAPPING.keys())}"
        )

    return lambda labels, predictions: metric(labels, predictions, **kwargs)
