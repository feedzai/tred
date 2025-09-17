from abc import ABC, abstractmethod

from numpy.typing import NDArray

from tred.datasets import BaseDataset

Dataset = BaseDataset
DatasetMap = dict[str, BaseDataset]


class BaseMethod(ABC):
    """
    Abstract base class for methods.

    This class defines the common interface all methods must implement
    (`fit` and `predict`) plus a convenience splitter that respects
    the dataset's temporal nature.

    Public attributes
    -----------------
    requires_target_data : bool
        Whether the method requires target domain data during training.
    requires_target_labels : bool
        Whether the method requires target domain labels during training.
    """

    requires_target_data: bool = True
    requires_target_labels: bool = True

    @abstractmethod
    def fit(
        self,
        train_L: DatasetMap,
        target_name: str,
        reset_model: bool = False,
        train_U: DatasetMap | None = None,
        validation: DatasetMap | None = None,
    ) -> None:
        """
        Fit the method given domain-partitioned datasets.

        Parameters
        ----------
        train_L : DatasetMap
            Labeled training sets by domain.
        target_name : str
            The name of the target domain.
        reset_model : bool, default=False
            Whether to reinitialize the model and optimizer before training.
        train_U : DatasetMap or None, default=None
            Unlabeled training sets by domain. Methods that do not use
            unlabeled data may ignore this argument.
        validation : DatasetMap or None, default=None
            Validation sets by domain. If None, the implementation may derive
            a validation split from `train_L`.
        """
        pass

    @abstractmethod
    def predict(self, dataset: Dataset, domain_name: str) -> NDArray:
        """
        Compute predictions on a dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset on which to compute the predictions.
        domain_name : str
            Name of the domain. May be used by some methods.

        Returns
        -------
        NDArray
            The method's predictions for the given dataset.
        """
        pass

    def _split_train_val(
        self, datasets: DatasetMap, val_ratio: float = 0.3
    ) -> tuple[DatasetMap, DatasetMap]:
        """
        Split the datasets of each domain into training and validation sets.

        Parameters
        ----------
        datasets : DatasetMap
            Dictionary with domain names as keys and datasets as values.
        val_ratio : float, default=0.3
            Proportion of the dataset's time span allocated to validation.

        Returns
        -------
        tuple[DatasetMap, DatasetMap]
            Training datasets and validation datasets.

        Raises
        ------
        ValueError
            If `datasets` is empty.
        """
        if not datasets:
            raise ValueError("No datasets provided to split.")

        train_datasets: DatasetMap = {}
        val_datasets: DatasetMap = {}

        for name_d, dataset in datasets.items():
            t, v = dataset.split(split_ratio=1.0 - val_ratio)
            train_datasets[name_d] = t
            val_datasets[name_d] = v

        return train_datasets, val_datasets
