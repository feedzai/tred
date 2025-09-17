from typing import Any

import numpy as np


class AddNoiseTransformation:
    """A data transformation operation that adds random noise to
    feature values and introduces random swaps to class labels."""

    def transform_domain(
        X: Any,
        y: Any,
        change_X_factor: float = 0.0,
        change_y_factor: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform the provided data from a single domain.

        Parameters
        ----------
        X: Any
            The features of the dataset. It is assumed that all features are
            numerical and that X can be convertible to a numpy ndarray,
            in which the first dimension corresponds to different instances.
        y: Any
            The labels of the dataset. It is assumed to be a binary
            classification task and that y can be convertible to a numpy
            ndarray of 0's and 1's, in which the first dimension corresponds
            to different instances.
        change_X_factor: float, default=0.0
            Scaling factor for noise added to the features. A value of 0.0 means
            no noise, while higher values increase the noise level proportionally.
        change_y_factor: float, default=0.0
            Scaling factor for noise added to the labels. It must be between
            0.0 and 1.0, with 0.0 meaning no label flipping and 1.0 meaning
            the label will be completly random.

        Returns
        -------
        (np.ndarray, np.ndarray)
            A tuple containing the transformed features and labels,
            each represented as ndarrays.
        """

        if not (0.0 <= change_y_factor <= 1.0):
            raise ValueError("`change_y_factor` must be between 0.0 and 1.0.")

        new_X = np.array(X)
        new_y = np.array(y)

        if change_X_factor > 0:
            X_std = X.std(axis=0)
            X_noise = np.random.normal(
                loc=0,
                scale=change_X_factor * X_std,
                size=new_X.shape,
            )
            new_X += X_noise

        if change_y_factor > 0:
            py = (1 - change_y_factor) * new_y + change_y_factor * 0.5
            new_y = (np.random.uniform(size=new_y.shape) <= py).astype(float)

        return new_X, new_y
