from typing import Optional

import numpy as np


class GenerateSyntheticData:
    """
    Class to generate synthetic datasets from multivariate Gaussian distributions.

    This class provides a flexible method to create synthetic data with controlled
    statistical properties, including spread, correlation, displacement, shape variability,
    and label complexity.
    """

    SMALL_VALUE = 1e-9
    SIGMOID_CLIP = 10

    def __init__(
        self,
        n_distrib: int,
        n_instances: int,
        n_dimensions: int,
        seed: Optional[int] = None,
    ):
        """
        Initialize the data generator with basic parameters and validate inputs.

        Parameters
        ----------
        n_distrib: int
            Number of distributions (domains) to generate.
        n_instances: int
            Number of instances (data points) per distribution.
        n_dimensions: int
            Number of features (dimensions) for each data point.
        seed: int, optional
            Random seed for reproducibility. If None, randomness will be uncontrolled.
        """
        if not isinstance(n_distrib, int) or n_distrib <= 0:
            raise ValueError("n_distrib must be a positive integer.")
        if not isinstance(n_instances, int) or n_instances <= 0:
            raise ValueError("n_instances must be a positive integer.")
        if not isinstance(n_dimensions, int) or n_dimensions <= 0:
            raise ValueError("n_dimensions must be a positive integer.")

        self.n_distrib = n_distrib
        self.n_instances = n_instances
        self.n_dimensions = n_dimensions
        self.rng = np.random.default_rng(seed)

    def generate_data(
        self,
        correlation_factor: float = 0.0,
        displacement_factor: float = 0.5,
        reshape_factor: float = 0.0,
        spreadness_factor: float = 0.5,
        label_complexity: int = 1,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate labelled synthetic data with specified properties.

        Parameters
        ----------
        correlation_factor: float
            Controls how much to bias correlation among features. Must be between 0.0 and 1.0.
        displacement_factor: float
            Controls how far the mean of the distribution is displaced. Must be non-negative.
        reshape_factor: float
            Degree of variability in the shape across domains. Must be between 0.0 and 1.0.
        spreadness_factor: float
            Controls the spread of each distribution. Must be between 0.0 and 1.0.
        label_complexity: int, default=1
            Complexity of the labeling logic. Represents the degree of polynomial used
            to generate labels. Accepts values 1, 2, or 3.

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            A list of tuples, where each tuple contains:
            - np.ndarray: Data points for the distribution (shape: `n_instances x n_dimensions`).
            - np.ndarray: Corresponding binary labels.
        """
        self._validate_generate_parameters(
            spreadness_factor,
            correlation_factor,
            displacement_factor,
            reshape_factor,
            label_complexity,
        )
        unlabeled_data = self.generate_features(
            correlation_factor, displacement_factor, reshape_factor, spreadness_factor
        )
        labels = self.generate_labels(unlabeled_data, label_complexity)
        return list(zip(unlabeled_data, labels))

    def generate_features(
        self,
        correlation_factor: float = 0.0,
        displacement_factor: float = 0.5,
        reshape_factor: float = 0.0,
        spreadness_factor: float = 0.5,
    ) -> list[np.ndarray]:
        """
        Generate unlabelled synthetic data with specified properties.

        Parameters
        ----------
        correlation_factor: float
            Controls how much to bias correlation among features. Must be between 0.0 and 1.0.
        displacement_factor: float
            Controls how far the mean of the distribution is displaced. Must be non-negative.
        reshape_factor: float
            Degree of variability in the shape across domains. Must be between 0.0 and 1.0.
        spreadness_factor: float
            Controls the spread of each distribution. Must be between 0.0 and 1.0.

        Returns
        -------
        list[np.ndarray]
            A list of np.ndarray's, each representing the data points of a domain.
        """
        self._validate_generate_parameters(
            correlation_factor, displacement_factor, reshape_factor, spreadness_factor
        )

        basis_matrix = self._get_basis_matrix(correlation_factor)
        spread_vector = self.rng.uniform(
            2 * spreadness_factor - 2, 2 * spreadness_factor, self.n_dimensions
        )
        sign_vector = self.rng.choice([-1, 1], self.n_dimensions)

        feature_sets = []
        for _ in range(self.n_distrib):
            # Generate factors of the decomposition
            Q = basis_matrix.copy()
            A = spread_vector.copy()
            if reshape_factor > 0:
                Q, A = self._perturb_matrices(
                    Q, A, correlation_factor, reshape_factor, spreadness_factor
                )
            Q *= sign_vector
            QA = Q @ np.diag(2**A)

            # Generate multivariate gaussian sample
            mu = self.rng.normal(0, 1, self.n_dimensions)
            mu /= np.linalg.norm(mu) + self.SMALL_VALUE
            mu = mu @ (QA * 3 * displacement_factor)
            Sigma = QA.T @ QA

            feature_sets.append(
                self.rng.multivariate_normal(mu, Sigma, size=self.n_instances)
            )

        return feature_sets

    def generate_labels(
        self, feature_sets: list[np.ndarray], label_complexity: int
    ) -> list[np.ndarray]:
        """
        Generate binary labels for the given feature sets.

        Parameters
        ----------
        feature_sets: list[np.ndarray]
            A list of feature sets generated for each domain.
        label_complexity: int
            Complexity of the labeling logic. Represents the degree of polynomial used
            to generate labels. Accepts values 1, 2, or 3.

        Returns
        -------
        list[np.ndarray]
            A list of binary label arrays corresponding to the feature sets.
        """
        if label_complexity not in [1, 2, 3]:
            raise ValueError("label_complexity must be an integer (1, 2, or 3).")

        label_coefficients = self.rng.normal(
            0, 1, (self.n_dimensions + 1, label_complexity)
        )
        labels = []
        for features in feature_sets:
            xx = np.hstack([np.ones((self.n_instances, 1)), features])
            cc = label_coefficients * self.rng.uniform(
                0.9, 1.1, label_coefficients.shape
            )
            px = np.prod(np.dot(xx, cc), axis=1)
            py = 1 / (1 + np.exp(np.clip(-px, -self.SIGMOID_CLIP, self.SIGMOID_CLIP)))
            labels.append((self.rng.uniform(size=self.n_instances) <= py).astype(float))
        return labels

    def _get_basis_matrix(self, correlation_factor: float) -> np.ndarray:
        """
        Generate a basis matrix with optional correlation alignment.

        Parameters
        ----------
        correlation_factor: float
            Degree of correlation among features. Must be between 0.0 and 1.0.

        Returns
        -------
        np.ndarray
            Basis matrix for feature generation.
        """
        Q = self.rng.normal(0, 1, (self.n_dimensions, self.n_dimensions))
        Q /= np.linalg.norm(Q, axis=0) + self.SMALL_VALUE
        if correlation_factor > 0:
            anchor_vector = self.rng.normal(0, 1, (self.n_dimensions, 1))
            anchor_vector /= np.linalg.norm(anchor_vector) + self.SMALL_VALUE
            Q = correlation_factor * anchor_vector + (1 - correlation_factor) * Q
            Q /= np.linalg.norm(Q, axis=0) + self.SMALL_VALUE
        return Q

    def _perturb_matrices(
        self,
        Q: np.ndarray,
        A: np.ndarray,
        correlation_factor: float,
        reshape_factor: float,
        spreadness_factor: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Adjust matrices to introduce variability across distributions.

        Parameters
        ----------
        Q: np.ndarray
            Basis matrix for the shape.
        A: np.ndarray
            Spread vector.
        correlation_factor: float
            Controls how much to bias correlation among features. Must be between 0.0 and 1.0.
        reshape_factor: float
            Degree of variability in the shape across domains. Must be between 0.0 and 1.0.
        spreadness_factor: float
            Controls the spread of each distribution. Must be between 0.0 and 1.0.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Adjusted basis matrix (Q) and spread vector (A).
        """
        Qr = self._get_basis_matrix(correlation_factor)
        Q = (1 - reshape_factor) * Q + reshape_factor * Qr
        Q /= np.linalg.norm(Q, axis=0) + self.SMALL_VALUE

        Ar = self.rng.uniform(
            2 * spreadness_factor - 2, 2 * spreadness_factor, self.n_dimensions
        )
        A = (1 - reshape_factor) * A + reshape_factor * Ar
        return Q, A

    def _validate_generate_parameters(
        self,
        correlation_factor: float,
        displacement_factor: float,
        reshape_factor: float,
        spreadness_factor: float,
        label_complexity: Optional[int] = None,
    ) -> None:
        """
        Validate parameters for feature and label generation.

        Parameters
        ----------
        correlation_factor: float
            Controls how much to bias correlation among features. Must be between 0.0 and 1.0.
        displacement_factor: float
            Controls how far the mean of the distribution is displaced. Must be non-negative.
        reshape_factor: float
            Degree of variability in the shape across domains. Must be between 0.0 and 1.0.
        spreadness_factor: float
            Controls the spread of each distribution. Must be between 0.0 and 1.0.
        label_complexity: int, optional
            Complexity of the labeling logic. If provided, must be 1, 2, or 3.
        """
        if not (0.0 <= correlation_factor <= 1.0):
            raise ValueError("correlation_factor must be between 0.0 and 1.0.")
        if not (0.0 <= displacement_factor):
            raise ValueError("displacement_factor must be non-negative.")
        if not (0.0 <= reshape_factor <= 1.0):
            raise ValueError("reshape_factor must be between 0.0 and 1.0.")
        if not (0.0 <= spreadness_factor <= 1.0):
            raise ValueError("spreadness_factor must be between 0.0 and 1.0.")
        if label_complexity is not None and label_complexity not in [1, 2, 3]:
            raise ValueError("label_complexity must be an integer (1, 2, or 3).")
