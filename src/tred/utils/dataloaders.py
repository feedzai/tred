from typing import Any, Iterable, Mapping

from numpy import union1d
from torch import cat, full, long
from torch.utils.data import DataLoader

from tred.datasets import BaseDataset


class BalancedDataloader:
    def __init__(
        self,
        batch_size: int,
        datasets: Mapping[str, BaseDataset],
        positive_ratio: float = 0.5,
        drop_last: bool = False,
        shuffle: bool = False,
        stop_criterion: int | str = "max",
    ):
        if not all(d.label_available for d in datasets.values()):
            raise ValueError(
                "Trying to create a BalancedDataloader with unlabelled dataset."
            )
        if (positive_ratio <= 0) or (1 <= positive_ratio):
            raise ValueError(
                f"positive_ratio should be in the interval ]0,1[ , but found {positive_ratio}."
            )

        self.batch_size = batch_size
        self.dataset_names = sorted(datasets)
        self.shuffle = shuffle

        self._batch_counter = 0
        self._iters = []  # type: ignore[var-annotated]
        self._mini_batch_size = batch_size // len(datasets)

        if positive_ratio > 0.5:
            pos_batch_size = int(positive_ratio * self._mini_batch_size)
            neg_batch_size = self._mini_batch_size - pos_batch_size
        else:
            neg_batch_size = int((1 - positive_ratio) * self._mini_batch_size)
            pos_batch_size = self._mini_batch_size - neg_batch_size

        # Build a DataLoader for each pair of (domain,class)
        self._loaders: list[DataLoader] = []
        for name_d in self.dataset_names:
            dataset = datasets[name_d]
            start, end = dataset.time_range
            negative_view = dataset.select_on_condition(start, end, 0)
            self._loaders.append(
                DataLoader(negative_view, neg_batch_size, shuffle, drop_last=drop_last)  # type: ignore[arg-type]
            )
            if len(self._loaders[-1]) == 0:
                raise ValueError(
                    f"The dataloader from the negative class of domain {name_d} is empty."
                )
            positive_view = dataset.select_on_condition(start, end, 1)
            self._loaders.append(
                DataLoader(positive_view, pos_batch_size, shuffle, drop_last=drop_last)  # type: ignore[arg-type]
            )
            if len(self._loaders[-1]) == 0:
                raise ValueError(
                    f"The dataloader from the positive class of domain {name_d} is empty."
                )

        if isinstance(stop_criterion, int):
            self._length = stop_criterion
        else:
            # Compute the stopping criterion based on domain loader lengths.
            stop_criterion_map = {
                "min": min(len(loader) for loader in self._loaders),
                "max": max(len(loader) for loader in self._loaders),
            }
            if stop_criterion not in stop_criterion_map:
                raise ValueError(
                    f"stop_criterion should be an integer or a string in ['min', 'max']. "
                    f"Found value '{stop_criterion}'."
                )
            self._length = stop_criterion_map[stop_criterion]
            if self._length == 0:
                raise ValueError("BalancedDataloader has length 0.")

    def __iter__(self):
        self._batch_counter = 0
        self._iters = [iter(loader) for loader in self._loaders]
        return self

    def __len__(self) -> int:
        return self._length

    def __next__(self) -> dict[str, Any]:
        if self._batch_counter >= self._length:
            raise StopIteration
        self._batch_counter += 1

        batch_X = []
        batch_y = []
        batch_d = []

        for i in range(len(self.dataset_names)):
            try:
                neg_mini_batch = next(self._iters[2 * i])
            except StopIteration:
                self._iters[2 * i] = iter(self._loaders[2 * i])
                neg_mini_batch = next(self._iters[2 * i])
            batch_X.append(neg_mini_batch["X"])
            batch_y.append(neg_mini_batch["y"])
            batch_d.append(
                full(size=neg_mini_batch["y"].size(), fill_value=i, dtype=long)
            )

            try:
                pos_mini_batch = next(self._iters[2 * i + 1])
            except StopIteration:
                self._iters[2 * i + 1] = iter(self._loaders[2 * i + 1])
                pos_mini_batch = next(self._iters[2 * i + 1])
            batch_X.append(pos_mini_batch["X"])
            batch_y.append(pos_mini_batch["y"])
            batch_d.append(
                full(size=pos_mini_batch["y"].size(), fill_value=i, dtype=long)
            )

        return {
            "X": cat(batch_X, dim=0),
            "y": cat(batch_y, dim=0),
            "d": cat(batch_d, dim=0),
        }


class CombinedDataloader:
    """
    A multi-domain data loader that iterates over multiple datasets in parallel,
    returning balanced mini-batches containing data from each domain (dataset).

    This loader merges different domain-specific datasets (or different views of
    the same dataset) into a single iterator. Each returned batch contains tensors
    concatenated along the batch dimension:

    - "X": features from all domains.
    - "y": labels from all domains (if available).
    - "d": domain labels (an integer identifier for each domain).

    The size of each mini-batch is controlled by `batch_size`, which is evenly
    split among the domains. The total number of batches is determined by the
    `stop_criterion` (either the minimum or maximum number of batches among
    the domain loaders).
    """

    def __init__(
        self,
        batch_size: int,
        l_datasets: list[dict[str, BaseDataset]],
        shuffle: bool = False,
        stop_criterion: int | str = "max",
    ):
        """
        Initialize the CombinedDataloader.

        Parameters
        ----------
        batch_size : int
            The total batch size for each combined iteration.
        l_datasets : list[dict[str, BaseDataset]]
            A list of dictionaries, where each dictionary maps domain names
            (str) to a `BaseDataset` or subclass.
        shuffle : bool, default=False
            Whether to shuffle the order of samples within each domain.
        stop_criterion : {"min", "max"}, default="max"
            - "min": The total number of batches is the minimum number of batches
              among the domain loaders (i.e., iteration stops when at least one
              domain is exhausted).
            - "max": The total number of batches is the maximum number of batches
              among the domain loaders (i.e., iteration continues, re-starting the
              exhausted domain loaders, until the domain with the largest number
              of batches is exhausted).

        Raises
        ------
        ValueError
            If `stop_criterion` is not one of {"min", "max"}.
        """
        self.batch_size = batch_size
        # Extract a sorted list of all unique domain names
        self.dataset_names = sorted(set(k for d in l_datasets for k in d))
        # Determine if all datasets have labels available
        self.label_available = all(
            v.label_available for d in l_datasets for v in d.values()
        )
        self.n_datasets = len(self.dataset_names)
        self.shuffle = shuffle

        self._batch_counter = 0
        self._iters = []  # type: ignore[var-annotated]
        self._mini_batch_size = batch_size // self.n_datasets

        # Build a DataLoader for each domain
        self._loaders: list[DataLoader] = []
        for name_d in self.dataset_names:
            # Combine different datasets of the same domain in a single view
            view = self._get_combined_view(
                [d[name_d] for d in l_datasets if name_d in d]
            )
            self._loaders.append(DataLoader(view, self._mini_batch_size, shuffle))  # type: ignore[arg-type]

        if isinstance(stop_criterion, int):
            self._length = stop_criterion
        else:
            # Compute the stopping criterion based on domain loader lengths.
            stop_criterion_map = {
                "min": min(len(loader) for loader in self._loaders),
                "max": max(len(loader) for loader in self._loaders),
            }
            if stop_criterion not in stop_criterion_map:
                raise ValueError(
                    f"stop_criterion should be an integer or a string in ['min', 'max']. "
                    f"Found value '{stop_criterion}'."
                )
            self._length = stop_criterion_map[stop_criterion]
            if self._length == 0:
                raise ValueError("CombinedDataloader has length 0.")

    def __iter__(self):
        self._batch_counter = 0
        self._iters = [iter(loader) for loader in self._loaders]
        return self

    def __len__(self) -> int:
        return self._length

    def __next__(self) -> dict[str, Any]:
        """
        Fetch the next combined batch from the domain loaders.

        This method retrieves one mini-batch from each domain loader, concatenates
        them along the batch dimension, and appends a domain index. If a particular
        domain loader is exhausted and `stop_criterion="max"`, the domain loader
        restarts from the beginning.

        Returns
        -------
        batch_dict : dict[str, Any]
            Dictionary containing:
            - "X": Torch tensor of concatenated features from all domains.
            - "y": Torch tensor of concatenated labels (if available), else omitted.
            - "d": Torch tensor of domain indices, indicating which domain each sample
              belongs to.

        Raises
        ------
        StopIteration
            When the data loader has reached the defined stopping point.
        """
        if self._batch_counter >= self._length:
            raise StopIteration
        self._batch_counter += 1

        batch_X = []
        batch_y = []
        batch_d = []

        for i in range(self.n_datasets):
            try:
                mini_batch = next(self._iters[i])
            except StopIteration:
                self._iters[i] = iter(self._loaders[i])
                mini_batch = next(self._iters[i])

            batch_X.append(mini_batch["X"])
            if self.label_available:
                batch_y.append(mini_batch["y"])
            batch_d.append(
                full(size=(mini_batch["X"].size()[0],), fill_value=i, dtype=long)
            )

        if self.label_available:
            return {
                "X": cat(batch_X, dim=0),
                "y": cat(batch_y, dim=0),
                "d": cat(batch_d, dim=0),
            }
        return {
            "X": cat(batch_X, dim=0),
            "d": cat(batch_d, dim=0),
        }

    def _get_combined_view(self, l_dataset: list[BaseDataset]) -> BaseDataset:
        """
        Merge multiple dataset views of the same domain into a single view.

        This new view contains the union of the active indices of all datasets
        in the provided list, without creating copies of the underlying data.

        Parameters
        ----------
        l_dataset : list[BaseDataset]
            List of dataset objects representing different subsets or splits of
            the same domain.

        Returns
        -------
        BaseDataset
            A new view with the union of the active indices.
        """
        view = l_dataset[0]._create_view()
        for i in range(1, len(l_dataset)):
            view._active_indices = union1d(
                view._active_indices, l_dataset[i]._active_indices
            )
        view.label_available = self.label_available
        view.__iter__()
        return view
