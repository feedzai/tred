from typing import Any, Self

from numpy import datetime64, float64, int64, ones, timedelta64, where
from numpy.typing import NDArray
from pandas import read_csv, to_datetime

from .base_dataset import BaseDataset, IndexLike


class CSVDataset(BaseDataset[NDArray[float64], NDArray[int64], NDArray[datetime64]]):
    """Dataset class to load datasets in CSV format."""

    def __init__(
        self,
        categorical_cardinalities: list[int],
        categorical_columns: list[str],
        label_column: str,
        numerical_columns: list[str],
        path: str,
        time_column: str,
        time_unit: str,
        label_available: bool = True,
        random_seed: int = 0,
        standardize_numericals: bool = False,
        shuffle: bool = False,
    ) -> None:
        """
        Initialize a CSV tabular dataset.

        Parameters:
        -----------
        categorical_cardinalities: list[int]
            Number of unique values for each categorical column.
        categorical_columns: list[str]
            Names of the categorical columns.
        label_column: str
            Name of the column with label information.
        numerical_columns: list[str]
            Names of the numerical columns.
        path: str
            Path to the dataset file.
        time_column: str
            Name of the column with temporal information.
        time_unit: str
            The unit of time for the `time_column`, used by pandas to convert to datetime.
        label_available: bool, default=True
            Whether labels are available and to be returned.
        random_seed : int, default=0
            Seed for random number generation, used for shuffling.
        standardize_numericals: bool, default=False
            Whether to standardize numerical columns when loading the data.
        shuffle: bool, default=False
            Whether to iterate in random order.
        """
        assert len(categorical_cardinalities) == len(
            categorical_columns
        ), "Length of `categorical_cardinalities` must match length of `categorical_columns`."
        assert (
            label_column not in numerical_columns
            and label_column not in categorical_columns
        ), "`label_column` must be different from feature columns."
        super().__init__(path, label_available, random_seed, shuffle)

        self.categorical_cardinalities: list[int] = categorical_cardinalities
        self.categorical_columns: list[str] = categorical_columns
        self.label_column: str = label_column
        self.numerical_columns: list[str] = numerical_columns
        self.standardize_numericals: bool = standardize_numericals
        self.time_column: str = time_column
        self.time_unit: str = time_unit

    def _get_X(self, real_idx: IndexLike) -> NDArray[float64]:
        return self._X[real_idx]  # type: ignore[index]

    def _get_y(self, real_idx: IndexLike) -> NDArray[int64]:
        return self._y[real_idx]  # type: ignore[index]

    def _get_t(self, real_idx: IndexLike) -> NDArray[datetime64]:
        return self._t[real_idx]  # type: ignore[index]

    def _load_data(self) -> None:
        """
        Notes:
        - Assumes the CSV file has a header row with column names.
        - Assumes all categorical columns are already integer-encoded.
        - Features are stored as numpy.float64, the labels as numpy.int64,
          and the timestamp as numpy.datetime64.
        - Uses pandas.to_datetime to parse the time column.
          The `time_unit` parameter is used to help with parsing.
        - Standardizes numerical columns if `standardize_numericals` is True.
          This is done at the entire dataset level, not at the split level.
        """
        df = read_csv(self.path)
        if self.standardize_numericals:
            for col in self.numerical_columns:
                df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
        self._X = df[self.numerical_columns + self.categorical_columns].to_numpy(
            dtype="float64"
        )
        self._y = df[self.label_column].to_numpy(dtype="int64")
        self._t = to_datetime(df[self.time_column], unit=self.time_unit).to_numpy()

    def _n_samples(self) -> int:
        return 0 if self._X is None else self._X.shape[0]

    def _select_on_condition(
        self,
        start_time: Any | None,
        end_time: Any | None,
        label_value: Any | None,
    ) -> Self:
        cond = ones(len(self), dtype=bool)
        if start_time is not None:
            start_time = self._to_ts(start_time)
            cond &= self._t[self._active_indices] >= start_time  # type: ignore[index]
        if end_time is not None:
            end_time = self._to_ts(end_time)
            cond &= self._t[self._active_indices] < end_time  # type: ignore[index]
        if label_value is not None:
            cond &= self._y[self._active_indices] == label_value  # type: ignore[index]

        self._active_indices = self._active_indices[where(cond)[0]]
        self._index = 0
        return self

    def _split(
        self,
        split_ratio: float | None,
        split_time: Any | None,
    ) -> tuple[Self, Self]:
        if split_ratio is not None:
            t = self._t[self._active_indices]  # type: ignore[index]
            t_min, t_max = t.min(), t.max()
            span_ns = (t_max - t_min) / timedelta64(1, "ns")
            split_ns = int(span_ns * float(split_ratio))
            split_time = t_min + timedelta64(split_ns, "ns")
        else:
            split_time = self._to_ts(split_time)

        split_a = self.select_on_condition(
            None, split_time, inplace=False, scope="active"
        )
        split_b = self.select_on_condition(
            split_time, None, inplace=False, scope="active"
        )

        return split_a, split_b

    def _to_ts(self, value: Any) -> datetime64:
        """
        Convert an input value to `numpy.datetime64` using `pandas.to_datetime`.
        """
        return to_datetime(value, unit=self.time_unit).to_numpy()
