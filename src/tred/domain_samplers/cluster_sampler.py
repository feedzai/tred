from typing import Callable

from numpy import clip, exp, where
from numpy.random import default_rng

from tred.datasets import BaseDataset

from .base_sampler import BaseSampler

Dataset = BaseDataset
DatasetMap = dict[str, BaseDataset]


class ClusterSampler(BaseSampler):
    """
    Implements a cluster-based sampling strategy to create domain-like data subsets.

    This class facilitates the creation of synthetic domains by sampling subsets of data
    based on their proximity to randomly chosen anchor points. The sampling probability decreases
    with distance from the anchor, controlled by a decay factor. The method ensures that each
    sampled subset meets specified criteria regarding minimum size and label distribution.
    """

    def __init__(
        self,
        distance_function: Callable,
        decay_factor: float = 1.0,
        limit_tries: int = 100,
        min_size: int = 0,
        min_labels: int = 0,
        random_seed: int = 0,
    ) -> None:
        """
        Initialize the sampling strategy.

        Parameters
        ----------
        distance_function: Callable
            A function to compute the distance between two data points.
        decay_factor: float, default=1.0
            A factor controlling the steepness of the probability distribution used
            for sampling. Higher values result in samples closer to the anchor.
        limit_tries: int, default=100
            Maximum number of attempts to find a valid sample that meets the
            specified criteria (minimum size and label counts).
        min_size: int, default=0
            Minimum acceptable size for the domain. This can be used as guardrails
            to avoid having domains that are too small.
        min_labels: int, default=0
            Minimum acceptable number of instances from each class for the domain.
            This can be used as guardrails to avoid having too few examples per class.
        random_seed: int, default=0
            Seed for the random number generator to ensure reproducibility.
        """
        self.decay_factor: float = decay_factor
        self.dist_f: Callable = distance_function
        self.limit_tries: int = limit_tries
        self.min_size: int = min_size
        self.min_labels: int = min_labels
        self.random_seed: int = random_seed
        self._rng = default_rng(random_seed)

    def _sample(
        self,
        dataset: Dataset,
        dataset_name: str,
        n_domains: int,
    ) -> DatasetMap:
        N = len(dataset)
        new_domains: DatasetMap = {}
        for i in range(n_domains):
            tries = 0
            while tries < self.limit_tries:
                anchor_idx = self._rng.integers(N)
                dists = self.dist_f(dataset, anchor_idx=anchor_idx)
                dists = self.decay_factor * dists / (dists.std() + 1e-12)
                probs = exp(clip(-dists, -20, 0))
                sample = where(self._rng.random(N) < probs)[0]

                size = len(sample)
                positive_labels = int(dataset[sample]["y"].sum())
                if (
                    size >= self.min_size
                    and positive_labels >= self.min_labels
                    and (size - positive_labels) >= self.min_labels
                ):
                    break
                tries += 1

            if tries == self.limit_tries:
                raise RuntimeError(
                    f"Didn't find a sample matching the requirements "
                    f"(size >= {self.min_size}, examples per class >= {self.min_labels}) "
                    f"after {self.limit_tries} attempts. Consider reducing "
                    f"`decay_factor` (currently at {self.decay_factor})."
                )

            new_dataset: Dataset = dataset._create_view()
            new_dataset._active_indices = sample
            new_domains[f"{dataset_name}_{i}"] = new_dataset
        return new_domains
