from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Generic, Literal, Self, Sequence, TypeVar

from numpy import arange, intp
from numpy.random import default_rng
from numpy.typing import NDArray

TFeat = TypeVar("TFeat")
TLabel = TypeVar("TLabel")
TTime = TypeVar("TTime")
IndexArray = NDArray[intp]
IndexLike = int | slice | Sequence[int] | IndexArray


class BaseDataset(ABC, Generic[TFeat, TLabel, TTime]):
    """
    Abstract base class for datasets.

    Provides a common structure for loading data, selecting subsets by condition,
    splitting by time, and iterating over samples.

    Public attributes
    ----------
    label_available : bool
        Whether labels `_y` are available and may be returned.
    path : str
        Filesystem path to the dataset source.
    random_seed : int
        Seed for random number generation, used for shuffling.
    shuffle : bool
        If True, iteration shuffles the active indices at the start of each iteration.

    Internal attributes
    -------------------
    _X : Any | None
        The feature information for each instance.
    _y : Any | None
        The label information for each instance.
    _t : Any | None
        The temporal information for each instance.
    _active_indices : numpy.NDArray[intp]
        Integer indices defining the current view of the data.
    _index : int
        Iterator cursor over `_active_indices`.
    _is_loaded : bool
        Whether the data has been loaded into memory.
    _rng : numpy.random.Generator
        Random number generator instance initialized with `random_seed`.

    Notes
    -----
    - The term "view" refers to an instance of a BaseDataset subclass that shares
    the same underlying data with other instances, but with independent active indices.
    Views avoid copying data, while allowing different samples to be considered active.
    """

    _X: TFeat | None
    _y: TLabel | None
    _t: TTime | None

    def __init__(
        self,
        path: str,
        label_available: bool = True,
        random_seed: int = 0,
        shuffle: bool = False,
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        path : str
            Filesystem path to the dataset source.
        label_available : bool, default=True
            Whether labels are available and to be returned.
        random_seed : int, default=0
            Seed for random number generation, used for shuffling.
        shuffle : bool, default=False
            Whether to iterate in random order.
        """
        self.label_available: bool = label_available
        self.path: str = path
        self.random_seed: int = random_seed
        self.shuffle: bool = shuffle

        self._X = None
        self._y = None
        self._t = None
        self._active_indices: IndexArray = arange(0, dtype=intp)
        self._index: int = 0
        self._is_loaded: bool = False
        self._rng = default_rng(random_seed)

    def get_X(self) -> TFeat:
        """
        Get features from all active indices.

        Returns
        -------
        TFeat
            Slice of `_X` for active indices.

        Raises
        ------
        RuntimeError
            If the dataset is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError("Dataset not loaded. Call load_data() first.")
        return self._get_X(self._active_indices)

    def get_y(self) -> TLabel | None:
        """
        Get labels from all active indices.

        Returns
        -------
        TLabel | None
            Slice of `_y` for active indices if labels are available, or None otherwise.

        Raises
        ------
        RuntimeError
            If the dataset is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError("Dataset not loaded. Call load_data() first.")
        return self._get_y(self._active_indices) if self.label_available else None

    def get_t(self) -> TTime:
        """
        Get temporal values from all active indices.

        Returns
        -------
        TTime
            Slice of `_t` for active indices.

        Raises
        ------
        RuntimeError
            If the dataset is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError("Dataset not loaded. Call load_data() first.")
        return self._get_t(self._active_indices)

    @property
    def is_loaded(self) -> bool:
        """
        Check if the dataset is loaded into memory.
        """
        return self._is_loaded

    def load_data(self) -> None:
        """
        Load the data from disk into memory.

        Must correctly populate `_X`, `_y`, `_t`.

        Notes
        -----
        `_y` MUST be populated even if `label_available` is False,
        because other views may enable their labels later.

        Raises
        ------
        ValueError
            If the dataset is empty or any of `_X`, `_y`, `_t` was not set.
        """
        self._load_data()
        n = self._n_samples()
        if n <= 0:
            raise ValueError(f"Dataset at {self.path} is empty.")
        if self._X is None:
            raise ValueError("_X was not loaded.")
        if self._y is None:
            raise ValueError("_y was not loaded.")
        if self._t is None:
            raise ValueError("_t was not loaded.")
        self._reset_active_indices()
        self._is_loaded = True

    def select_on_condition(
        self,
        start_time: Any | None = None,
        end_time: Any | None = None,
        label_value: Any | None = None,
        inplace: bool = False,
        scope: Literal["active", "all"] = "active",
    ) -> Self:
        """
        Return a filtered view by time range and/or label value.

        By default, filtering is applied to the current active view.
        If `scope="all"`, filtering is recomputed from the full dataset.
        If `inplace=True`, the current object is modified; otherwise, a new view is returned.

        Parameters
        ----------
        start_time : Any | None, default=None
            Keep samples with `t >= start_time` (inclusive). If None, no lower bound.
        end_time : Any | None, default=None
            Keep samples with `t < end_time` (exclusive). If None, no upper bound.
        label_value : Any | None, default=None
            Keep samples whose label equals this value. Ignored if None.
        inplace : bool, default=False
            If True, mutate this view; otherwise return a new view.
        scope : Literal["active", "all"], default="active"
            If "active", filters within the current active indices.
            If "all", recomputes from the entire dataset.

        Returns
        -------
        Self
            The filtered dataset view.

        Raises
        ------
        RuntimeError
            If the dataset is not loaded.
        ValueError
            If label filter is requested but labels are not available.
        """
        if not self._is_loaded:
            raise RuntimeError("Dataset not loaded. Call load_data() first.")

        if (label_value is not None) and (not self.label_available):
            raise ValueError("Label filter requested but `label_available` is False.")

        ret = self if inplace else self._create_view()
        if scope == "all":
            ret._reset_active_indices()

        return ret._select_on_condition(start_time, end_time, label_value)

    def split(
        self,
        split_ratio: float | None = None,
        split_time: Any | None = None,
    ) -> tuple[Self, Self]:
        """
        Split the current view into two non-overlapping views.

        Exactly one of `split_ratio` or `split_time` must be provided.
        The split is performed over the *time span* of the current active indices,
        not the sample count. Samples with `t < split_time` go to the first view;
        samples with `t >= split_time` go to the second view.

        Parameters
        ----------
        split_ratio : float | None, default=None
            Fraction of the dataset's time span allocated to the first split (0 < ratio < 1).
        split_time : Any | None, default=None
            Exact timestamp at which to split the dataset.

        Returns
        -------
        (Self, Self)
            Tuple of two dataset views.

        Raises
        ------
        RuntimeError
            If the dataset is not loaded.
        ValueError
            If neither or both of `split_ratio` and `split_time` are provided,
            if `split_ratio` is not in (0, 1), or if there are no active indices.
        """
        if not self._is_loaded:
            raise RuntimeError("Dataset not loaded. Call load_data() first.")

        if (split_time is None) == (split_ratio is None):
            raise ValueError("Provide exactly one of `split_ratio` or `split_time`.")
        if split_ratio is not None and not (0 < split_ratio < 1):
            raise ValueError("`split_ratio` must be in the range ]0, 1[.")
        if len(self._active_indices) == 0:
            raise ValueError("No active indices to split.")

        return self._split(split_ratio, split_time)

    def _create_view(self) -> Self:
        """
        Create a lightweight view of the dataset.

        Shares the same underlying data (_X, _y, _t) as the original dataset,
        but has independent active indices, iterator state, and random generator.
        """
        view = copy(self)
        view._active_indices = self._active_indices.copy()
        view._index = 0
        view._is_loaded = self._is_loaded
        view._rng = default_rng(self.random_seed)
        return view

    @abstractmethod
    def _get_X(self, real_idx: IndexLike) -> TFeat:
        """
        Get features at the real indices.
        """
        pass

    @abstractmethod
    def _get_y(self, real_idx: IndexLike) -> TLabel:
        """
        Get labels at the real indices.
        """
        pass

    @abstractmethod
    def _get_t(self, real_idx: IndexLike) -> TTime:
        """
        Get temporal values at the real indices.
        """
        pass

    @abstractmethod
    def _load_data(self) -> None:
        """
        Load the data from file into memory.
        """
        pass

    @abstractmethod
    def _n_samples(self) -> int:
        """
        Get the total number of samples in the dataset.
        """
        pass

    def _reset_active_indices(self) -> None:
        """
        Reset the active indices to include all samples.
        """
        self._active_indices = arange(self._n_samples(), dtype=intp)
        self._index = 0

    @abstractmethod
    def _select_on_condition(
        self,
        start_time: Any | None,
        end_time: Any | None,
        label_value: Any | None,
    ) -> Self:
        """
        Filter the dataset by time range and/or label value.
        """
        pass

    def _shuffle_data(self) -> None:
        """
        Shuffle the active indices in place.
        """
        perm = self._rng.permutation(len(self))
        self._active_indices = self._active_indices[perm]

    @abstractmethod
    def _split(
        self,
        split_ratio: float | None,
        split_time: Any | None,
    ) -> tuple[Self, Self]:
        """
        Split the dataset into two non-overlapping views.
        """
        pass

    def __len__(self) -> int:
        """
        Number of active instances in the dataset.

        Raises
        ------
        RuntimeError
            If the dataset is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError("Dataset not loaded. Call load_data() first.")
        return self._active_indices.size

    def __getitem__(self, idx: IndexLike) -> dict[str, Any]:
        """
        Get a sample or a batch of samples.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys 'X' (features), 'y' (labels, if available), and 't' (time).

        Raises
        ------
        RuntimeError
            If the dataset is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError("Dataset not loaded. Call load_data() first.")

        real_idx = self._active_indices[idx]
        return {
            "X": self._get_X(real_idx),
            "y": self._get_y(real_idx) if self.label_available else None,
            "t": self._get_t(real_idx),
        }

    def __iter__(self) -> Self:
        """
        Reset the iterator (and shuffle if enabled).

        Raises
        ------
        RuntimeError
            If the dataset is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError("Dataset not loaded. Call load_data() first.")

        self._index = 0
        if self.shuffle:
            self._shuffle_data()
        return self

    def __next__(self) -> dict[str, Any]:
        """
        Return the next instance of the dataset.

        Raises
        ------
        StopIteration
            When all active samples have been iterated.
        """
        if self._index >= self._active_indices.size:
            raise StopIteration
        value = self[self._index]
        self._index += 1
        return value
