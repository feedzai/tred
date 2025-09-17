from copy import deepcopy

from numpy.typing import NDArray
from torch import Tensor, cat, device, nn, no_grad, sigmoid
from torch.optim import Adam
from torch.utils.data import DataLoader

from tred.datasets import BaseDataset
from tred.models import MLP
from tred.utils import BalancedDataloader, metric_wrapper

from .base_method import BaseMethod

Dataset = BaseDataset
DatasetMap = dict[str, BaseDataset]


class CentralMLP(BaseMethod):
    """
    Train a single MLP using data pooled across domains.

    Depending on configuration, the model can train on:
      - source domains only,
      - target domain only, or
      - both source and target.
    """

    def __init__(
        self,
        batch_size: int,
        batches_per_epoch: int,
        include_source: bool,
        include_target: bool,
        lr: float,
        max_epochs: int,
        model_confs: dict,
        patience: int,
        positive_ratio: float,
        use_device: str,
        validation_metric: dict,
        validation_ratio: float = 0.3,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the method.

        Parameters
        ----------
        batch_size : int
            Batch size for training and evaluation.
        batches_per_epoch : int
            Number of batches to train in each epoch (before computing validation loss).
        include_source : bool
            Whether to include source domains in the training.
        include_target : bool
            Whether to include the target domain in the training.
        lr : float
            Learning rate.
        max_epochs : int
            Maximum number of training epochs.
        model_confs : dict
            Configurations for the MLP model.
        patience : int
            Early-stopping patience (epochs without validation improvement).
        positive_ratio : float
            Ratio of positive labels to include in each batch.
        use_device : str
            Device to use for computation (e.g., 'cpu', 'cuda').
        validation_metric : dict
            The validation metric used for early stopping.
        validation_ratio : float, default=0.3
            Proportion of training time span used for validation when none is provided.
        verbose : bool, default=False
            Whether to print training progress.

        Raises
        ------
        ValueError
            If both `include_source` and `include_target` are False.
        """
        if not include_source and not include_target:
            raise ValueError(
                "At least one of `include_source` or `include_target` must be True."
            )
        self.requires_target_data = not include_source
        self.requires_target_labels = not include_source

        self.batch_size: int = batch_size
        self.batches_per_epoch: int = batches_per_epoch
        self.device = device(use_device)
        self.include_source: bool = include_source
        self.include_target: bool = include_target
        self.lr: float = lr
        self.max_epochs: int = max_epochs
        self.model: MLP = MLP(**model_confs).to(self.device)
        self.model_confs: dict = model_confs
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.patience: int = patience
        self.positive_ratio: float = positive_ratio
        self.val_ratio: float = validation_ratio
        self.verbose: bool = verbose

        self.BCE_loss: nn.Module = nn.BCEWithLogitsLoss()
        self.val_metric = metric_wrapper(
            validation_metric["metric"], **validation_metric.get("config", {})
        )

    def fit(
        self,
        train_L: DatasetMap,
        target_name: str,
        reset_model: bool = False,
        train_U: DatasetMap | None = None,
        validation: DatasetMap | None = None,
    ) -> None:
        # Prepare dataloader
        train_L = {
            name_d: dataset
            for name_d, dataset in train_L.items()
            if (name_d != target_name and self.include_source)
            or (name_d == target_name and self.include_target)
        }
        if len(train_L) == 0:
            print(
                "No labelled training data available to train CentralMLP. Skipping fit."
            )
            return
        if validation is None:
            train_L, validation = self._split_train_val(train_L, self.val_ratio)
        train_dataloader = BalancedDataloader(
            batch_size=self.batch_size,
            datasets=train_L,
            positive_ratio=self.positive_ratio,
            shuffle=True,
            stop_criterion=self.batches_per_epoch,
        )

        # Reset model if required
        if reset_model:
            if self.verbose:
                print("Resetting model and optimizer.")
            self.model = MLP(**self.model_confs).to(self.device)
            self.optimizer = Adam(self.model.parameters(), lr=self.lr)

        # Establish baseline validation performance
        best_epoch = -1
        best_model_state = deepcopy(self.model.state_dict())
        best_score = self._compute_val_score(validation)
        if self.verbose:
            print(f"Initial Val Score: {best_score:9.6f}")

        # Training loop with early stopping
        for epoch in range(self.max_epochs):
            train_loss = self._train_epoch(train_dataloader)
            val_score = self._compute_val_score(validation)
            if self.verbose:
                print(
                    f"Epoch {epoch + 1:2d}, "
                    f"Train Loss: {train_loss:9.6f}, "
                    f"Val Score: {val_score:9.6f}"
                )
            if val_score > best_score:
                best_epoch = epoch
                best_model_state = deepcopy(self.model.state_dict())
                best_score = val_score
            if epoch - best_epoch >= self.patience:
                if self.verbose:
                    print("Early stopping triggered.")
                break

        # Load best model state
        self.model.load_state_dict(best_model_state)
        if self.verbose:
            print(f"Final Val Score: {best_score:9.6f}")

    def predict(self, dataset: Dataset, domain_name: str) -> NDArray:
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)  # type: ignore[arg-type, var-annotated]
        preds = []
        self.model.eval()
        with no_grad():
            for batch in dataloader:
                batch = self._move_to_device(batch)
                logits = self.model(batch["X"])
                preds.append(sigmoid(logits))
        return cat(preds).cpu().numpy().squeeze(1)

    def _compute_val_score(self, datasets: DatasetMap) -> float:
        """
        Compute the average of the validation scores of the method on each domain.
        """
        scores_per_dataset = []
        for name_d, dataset in datasets.items():
            score = self.val_metric(dataset.get_y(), self.predict(dataset, name_d))  # type: ignore[arg-type]
            scores_per_dataset.append(score)
        return sum(scores_per_dataset) / len(scores_per_dataset)

    def _move_to_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Move a batch of data to the specified device.
        """
        if "X" in batch:
            batch["X"] = batch["X"].to(self.device).float()
        if "y" in batch:
            batch["y"] = batch["y"].to(self.device).float()
        return batch

    def _train_epoch(self, dataloader: BalancedDataloader) -> float:
        """
        Train the model for one epoch.

        Parameters
        ----------
        dataloader : BalancedDataloader
            DataLoader for the training data.

        Returns
        -------
        float
            Average training loss per batch.
        """
        iter(dataloader)
        self.model.train()
        epoch_loss = 0.0

        for _ in range(self.batches_per_epoch):
            self.optimizer.zero_grad()
            batch = self._move_to_device(next(dataloader))
            logits = self.model(batch["X"])
            loss = self.BCE_loss(logits, batch["y"].unsqueeze(1).float())
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        return epoch_loss / self.batches_per_epoch
