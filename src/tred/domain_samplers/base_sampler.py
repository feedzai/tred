from abc import ABC, abstractmethod

from tred.datasets import BaseDataset

Dataset = BaseDataset
DatasetMap = dict[str, BaseDataset]


class BaseSampler(ABC):
    """
    Abstract base class for domain samplers.

    Subclasses implement `_sample`, which returns a mapping of new domain names
    to dataset views for a single input dataset.
    """

    def sample(
        self,
        datasets: DatasetMap,
        n_domains: int | list[int],
        selected_datasets: list[str] | None = None,
    ) -> DatasetMap:
        """
        Create a number of sub-samples from one or more datasets.

        Parameters
        ----------
        datasets : DatasetMap
            Map from dataset name to dataset object.
        n_domains : int | list[int]
            Number of domains to sample per dataset. If int, the same number is used for each dataset.
            If list, its length must match the number of datasets, in the same order.
        selected_datasets : list[str] | None, default=None
            List of dataset names to sample from. If None, all datasets will be used.

        Returns
        -------
        DatasetMap
            Map from new domain names to dataset views.

        Raises
        ------
        KeyError
            If any name in `selected_datasets` is not a key of `datasets`.
        ValueError
            If the length of `n_domains` does not match the length of `selected_datasets`.
        """
        if selected_datasets is None:
            selected_datasets = list(datasets.keys())
        else:
            missing = [k for k in selected_datasets if k not in datasets]
            if missing:
                raise KeyError(f"Unknown dataset(s) in selected_datasets: {missing}.")

        if isinstance(n_domains, int):
            n_domains = [n_domains for _ in selected_datasets]
        elif len(n_domains) != len(selected_datasets):
            raise ValueError(
                f"Length of `n_domains` must match length of `selected_datasets` "
                f"(got {len(n_domains)} and {len(selected_datasets)})."
            )

        new_datasets: DatasetMap = {}
        for name_d, n_dom in zip(selected_datasets, n_domains):
            new_datasets.update(
                self._sample(
                    dataset=datasets[name_d],
                    dataset_name=name_d,
                    n_domains=n_dom,
                )
            )
        return new_datasets

    @abstractmethod
    def _sample(
        self,
        dataset: Dataset,
        dataset_name: str,
        n_domains: int,
    ) -> DatasetMap:
        """
        Create a number of sub-samples from a single dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset from which to sample.
        dataset_name : str
            Name of dataset, to be used as prefix for domain names.
        n_domains : int
            Number of domains to sample.

        Returns
        -------
        DatasetMap
            Map from new domain names to dataset views.
        """
        pass
