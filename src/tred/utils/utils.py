from datetime import datetime, timezone
from importlib import import_module
from typing import Any

from numpy import clip
from numpy.typing import NDArray
from torch.autograd import Function

from tred.datasets import CSVDataset


def csv_distance(
    dataset: CSVDataset, anchor: Any | None = None, anchor_idx: int | None = None
) -> NDArray:
    """
    Compute the distance between each point in a dataset and a given anchor point.

    Combines the standardized Euclidean distance for numerical features
    and Hamming distance over the categorical features.

    Parameters
    ----------
    dataset: CSVDataset
        The dataset containing the data points.
    anchor: Any, optional
        A specific data point to serve as the anchor. If None, `anchor_idx` must be provided.
    anchor_idx: int, optional
        The index of the data point in the dataset to serve as the anchor. Used if `anchor` is None.

    Returns
    -------
    NDArray
        An array of distances from each data point in the dataset to the anchor point.
    """
    if anchor is None:
        if anchor_idx is None:
            raise ValueError(
                "`anchor` and `anchor_idx` cannot be None at the same time."
            )
        anchor = dataset[anchor_idx]["X"]

    X = dataset.get_X()
    n_num = len(dataset.numerical_columns)
    X_std = X[:, :n_num].std(axis=0) + 1e-6
    stand_diff = clip(abs(anchor[:n_num] - X[:, :n_num]) / X_std, 0, 10)
    num_dist = (stand_diff**2).sum(axis=1)
    cat_dist = (anchor[n_num:] != X[:, n_num:]).astype(float).sum(axis=1)

    return num_dist + cat_dist


def load_class(class_path: str) -> Any:
    """Dynamically loads a class from a path"""
    if "." in class_path:
        module_str, component = class_path.rsplit(".", 1)
        module = import_module(module_str)
        class_ = getattr(module, component)
        return class_
    else:
        raise ValueError("`load_class` - Classpath provided does not specify module")


def ts_to_date(timestamp: int | None) -> str:
    if timestamp is None:
        return "xxxx-xx-xx"
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")


class GRL(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad.neg()
