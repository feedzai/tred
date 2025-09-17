from lightgbm import LGBMRegressor
from numpy.typing import NDArray

from tred.datasets import CSVDataset

from .base_method import BaseMethod

Dataset = CSVDataset
DatasetMap = dict[str, CSVDataset]


class Lgbm(BaseMethod):

    requires_target_data = True
    requires_target_labels = True

    def __init__(
        self,
        max_dataset_size: int,
        model_confs: dict = {},
    ) -> None:
        self.max_dataset_size: int = max_dataset_size
        self.model: LGBMRegressor = LGBMRegressor(verbose=-1, **model_confs)
        self.model_confs: dict = model_confs

    def fit(  # type: ignore[override]
        self,
        train_L: DatasetMap,
        target_name: str,
        reset_model: bool = False,
        train_U: DatasetMap | None = None,
        validation: DatasetMap | None = None,
    ) -> None:
        dataset = train_L[target_name]
        self.model.fit(
            dataset.get_X()[: self.max_dataset_size],
            dataset.get_y()[: self.max_dataset_size],  # type: ignore[index]
            feature_name=(dataset.numerical_columns + dataset.categorical_columns),
            categorical_feature=dataset.categorical_columns,
        )

    def predict(self, dataset: Dataset, domain_name: str) -> NDArray:  # type: ignore[override]
        return self.model.predict(dataset.get_X()).clip(0, 1)
