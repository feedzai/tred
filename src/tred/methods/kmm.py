from adapt.instance_based import KMM as KMMAdapt
from lightgbm import LGBMRegressor
from numpy import hstack, vstack, where
from numpy.random import choice
from numpy.typing import NDArray

from tred.datasets import CSVDataset

from .base_method import BaseMethod

Dataset = CSVDataset
DatasetMap = dict[str, CSVDataset]


class KMM(BaseMethod):
    """Kernel Mean Matching (KMM)"""

    requires_target_data = True
    requires_target_labels = False

    def __init__(
        self,
        model_confs: dict,
        positive_ratio: float,
        xs_size: int,
        xt_size: int,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the method.

        Parameters
        ----------
        model_confs: dict
            Configurations for the LGBM model.
        positive_ratio : float
            Ratio of positive labels to include in each batch.
        xs_size : int
            Number of source samples to use for training.
        xt_size : int
            Number of target samples to use for training.
        verbose : bool, default=False
            Whether to print training progress.
        """
        self.model: LGBMRegressor = LGBMRegressor(verbose=-1, **model_confs)
        self.model_confs: dict = model_confs
        self.positive_ratio: float = positive_ratio
        self.verbose: bool = verbose
        self.xs_size: int = xs_size
        self.xt_size: int = xt_size

    def fit(  # type: ignore[override]
        self,
        train_L: DatasetMap,
        target_name: str,
        reset_model: bool = False,
        train_U: DatasetMap | None = None,
        validation: DatasetMap | None = None,
    ) -> None:
        X_t = train_U[target_name].get_X()[-self.xt_size :]  # type: ignore[index]
        l_X_s = []
        l_y_s = []
        n = self.xs_size // len(train_L)

        for name in train_L:
            x = train_L[name].get_X()
            y = train_L[name].get_y()
            pos_indices = where(y == 1)[0]
            neg_indices = where(y == 0)[0]
            pos_indices = choice(
                pos_indices,
                int(n * self.positive_ratio),
                replace=len(pos_indices) < int(n * self.positive_ratio),
            )
            neg_indices = choice(
                neg_indices,
                int(n * (1 - self.positive_ratio)),
                replace=len(neg_indices) < int(n * (1 - self.positive_ratio)),
            )
            l_X_s += [x[pos_indices], x[neg_indices]]
            l_y_s += [y[pos_indices], y[neg_indices]]  # type: ignore[index]
        X_s = vstack(l_X_s)
        y_s = hstack(l_y_s)

        kmm = KMMAdapt(verbose=0)
        weights = kmm.fit_weights(X_s, X_t)
        if self.verbose:
            for i, name in enumerate(train_L):
                print(f"Domain {name} total weight: {weights[n*i:n*(i+1)].sum()}")
            print(f"Positives total weight: {weights[y_s == 1].sum()}")
            print(f"Negatives total weight: {weights[y_s == 0].sum()}")

        self.model.fit(X_s, y_s, sample_weight=weights)

    def predict(self, dataset: Dataset, domain_name: str) -> NDArray:  # type: ignore[override]
        return self.model.predict(dataset.get_X()).clip(0, 1)
