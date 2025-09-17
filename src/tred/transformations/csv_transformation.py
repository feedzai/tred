from typing import Any

from numpy import array, maximum, ndarray, pi, sin, unique, where, zeros
from numpy.random import choice, dirichlet, uniform

from tred.datasets import CSVDataset


class CSVTransformation:
    """Implements transformations on CSVDataset's."""

    def __init__(self, transformations_configs: dict[str, dict[str, Any]]) -> None:
        """Create CSVTransformation object, storing a dictionary with its configurations.

        The configurations should be structured as a dictionary, with the keys being unique
        names for each transformation, and the value being another dictionary with configurations
        for each transformation. This second dictionary should contain the following:
            columns: list of strings with the column names to be changed.
            operation: selects the operation that is going to be applied to the columns.
                Should be one of "scale_numerical", "attract_numericals",
                "resample_categoricals", "resample_label".
            time_dependency: defines how the transformation depends on time.
                Should be one of "const", "drift", "cycle".
            period: Only required for time_dependency="cycle". Defines the period of the cycle.

        There are other keys that need to be included, depending on the operation type.
        For operation="scale_numericals":
            scale_factor: [lower_bound, upper_bound]
        For operation="attract_numericals":
            anchor_value: [lower_bound, upper_bound]
            anchor_weight: [lower_bound, upper_bound]
        For operation="resample_categoricals":
            change_factor: [lower_bound, upper_bound]
            new_prob: (optional) list of floats
        For operation="resample_label":
            change_factor: [lower_bound, upper_bound]
            new_prob: list of floats
        """
        self.transformations: dict[str, dict[str, Any]] = transformations_configs

    def apply_transformations(
        self,
        datasets: dict[str, CSVDataset],
        verbose: bool = False,
    ) -> None:
        for name_d, dataset in datasets.items():
            if verbose:
                print(f"Transforming dataset {name_d}.")
            for name_t, transformation in self.transformations.items():

                if transformation["operation"] == "scale_numericals":
                    scale_factor = uniform(*(transformation["scale_factor"]))
                    for column in transformation["columns"]:
                        if verbose:
                            print(
                                f"{name_t:20s} is applying `scale_numericals` with scale factor {scale_factor:6.3f}."
                            )
                        self._scale_numerical(
                            dataset,
                            column,
                            transformation["time_dependency"],
                            transformation.get("period", 0),
                            scale_factor,
                        )

                if transformation["operation"] == "attract_numericals":
                    anchor_value = uniform(*transformation["anchor_value"])
                    anchor_weight = uniform(*transformation["anchor_weight"])
                    for column in transformation["columns"]:
                        if verbose:
                            print(
                                f"{name_t:20s} is applying `attract_numericals` with anchor value {anchor_value:6.3f} and anchor weight {anchor_weight:6.3f}."
                            )
                        self._attract_numerical(
                            dataset,
                            column,
                            transformation["time_dependency"],
                            transformation.get("period", 0),
                            anchor_value,
                            anchor_weight,
                        )

                if transformation["operation"] == "resample_categorical":
                    change_factor = uniform(*transformation["change_factor"])
                    new_prob = array(transformation.get("new_prob", []))
                    if len(new_prob) == 0:
                        column = transformation["columns"][0]
                        column_idx = dataset.categorical_columns.index(column)
                        cardinality = dataset.categorical_cardinalities[column_idx]
                        new_prob = dirichlet([1] * cardinality)
                    for column in transformation["columns"]:
                        if verbose:
                            print(
                                f"{name_t:20s} is applying `resample_categorical` with changefactor {change_factor:6.3f}."
                            )
                        self._resample_categorical(
                            dataset,
                            column,
                            transformation["time_dependency"],
                            transformation.get("period", 0),
                            change_factor,
                            new_prob,
                        )

                if transformation["operation"] == "resample_label":
                    change_factor = uniform(*transformation["change_factor"])
                    new_prob = transformation["new_prob"]
                    if verbose:
                        print(
                            f"{name_t:20s} is applying `resample_label` with changefactor {change_factor:6.3f}."
                        )
                    self._resample_label(
                        dataset,
                        transformation["time_dependency"],
                        transformation.get("period", 0),
                        change_factor,
                        new_prob,
                    )

    def _scale_numerical(
        self,
        dataset: CSVDataset,
        column: str,
        time_dependency: str,
        period,
        scale_factor: float,
    ):
        column_idx = dataset.numerical_columns.index(column)
        if time_dependency == "drift":
            t_min = dataset.get_t().min()
            t_max = dataset.get_t().max()
            scale_factor = scale_factor ** ((dataset._t - t_min) / (t_max - t_min))
        if time_dependency == "cycle":
            t_min = dataset.get_t().min()
            scale_factor = scale_factor ** (
                (sin(2 * pi * (dataset._t - t_min) / period) + 1) / 2
            )
        dataset._X[:, column_idx] *= scale_factor  # type: ignore

    def _attract_numerical(
        self,
        dataset: CSVDataset,
        column: str,
        time_dependency: str,
        period,
        anchor_value: float,
        anchor_weight: float,
    ):
        column_idx = dataset.numerical_columns.index(column)
        if time_dependency == "drift":
            t_min = dataset.get_t().min()
            t_max = dataset.get_t().max()
            anchor_weight = anchor_weight * ((dataset._t - t_min) / (t_max - t_min))
        if time_dependency == "cycle":
            t_min = dataset.get_t().min()
            anchor_weight = anchor_weight * (
                (sin(2 * pi * (dataset._t - t_min) / period) + 1) / 2
            )
        dataset._X[:, column_idx] = (1 - anchor_weight) * dataset._X[  # type: ignore
            :, column_idx
        ] + anchor_weight * anchor_value

    def _resample_categorical(
        self,
        dataset: CSVDataset,
        column: str,
        time_dependency: str,
        period,
        change_factor: float,
        new_prob: ndarray,
    ):
        # Extract original values
        column_idx = dataset.categorical_columns.index(column)
        cardinality = dataset.categorical_cardinalities[column_idx]
        old_values = dataset._X[:, len(dataset.numerical_columns) + column_idx].astype(  # type: ignore
            int
        )
        N = len(old_values)

        # Get new values to insert
        counts = unique(old_values, return_counts=True)
        all_counts = zeros(cardinality)
        all_counts[counts[0]] += counts[1]
        old_prob = (all_counts + 1) / N
        diff = maximum(0, new_prob - old_prob)
        diff /= diff.sum()
        new_values = choice(cardinality, size=N, p=diff)

        # Compute resample probabilities
        resample_prob_per_class = maximum(0, 1 - new_prob / old_prob)
        resample_prob = resample_prob_per_class[old_values]
        if time_dependency == "const":
            resample_prob *= change_factor
        elif time_dependency == "drift":
            t_min = dataset.get_t().min()
            t_max = dataset.get_t().max()
            resample_prob *= change_factor ** ((dataset._t - t_min) / (t_max - t_min))
        elif time_dependency == "cycle":
            t_min = dataset.get_t().min()
            resample_prob *= change_factor ** (
                (sin(2 * pi * (dataset._t - t_min) / period) + 1) / 2
            )
        else:
            raise ValueError(f"`time_dependency` '{time_dependency}' not supported.")

        # Transform dataset
        rand = uniform(size=N)
        dataset._X[:, len(dataset.numerical_columns) + column_idx] = where(  # type: ignore
            rand < resample_prob, new_values, old_values
        )

    def _resample_label(
        self,
        dataset: CSVDataset,
        time_dependency: str,
        period,
        change_factor: float,
        new_prob: ndarray,
    ):
        # Extract original values
        cardinality = 2
        old_values = dataset._y.astype(int)  # type: ignore
        N = len(old_values)

        # Get new values to insert
        counts = unique(old_values, return_counts=True)
        all_counts = zeros(cardinality)
        all_counts[counts[0]] += counts[1]
        old_prob = (all_counts + 1) / N
        diff = maximum(0, new_prob - old_prob)
        diff /= diff.sum()
        new_values = choice(cardinality, size=N, p=diff)

        # Compute resample probabilities
        resample_prob_per_class = maximum(0, 1 - new_prob / old_prob)
        resample_prob = resample_prob_per_class[old_values]
        if time_dependency == "const":
            resample_prob *= change_factor
        elif time_dependency == "drift":
            t_min = dataset.get_t().min()
            t_max = dataset.get_t().max()
            resample_prob *= change_factor ** ((dataset._t - t_min) / (t_max - t_min))
        elif time_dependency == "cycle":
            t_min = dataset.get_t().min()
            resample_prob *= change_factor ** (
                (sin(2 * pi * (dataset._t - t_min) / period) + 1) / 2
            )
        else:
            raise ValueError(f"`time_dependency` '{time_dependency}' not supported.")

        # Transform dataset
        rand = uniform(size=N)
        dataset._y = where(rand < resample_prob, new_values, old_values)  # type: ignore
