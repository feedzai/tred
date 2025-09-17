from copy import deepcopy

from numpy.typing import NDArray
from torch import Tensor, cat, device, log2, mean, nn, no_grad, sigmoid
from torch.optim import Adam
from torch.utils.data import DataLoader

from tred.datasets import BaseDataset
from tred.models import MLP
from tred.utils import GRL, BalancedDataloader, metric_wrapper

from .base_method import BaseMethod

Dataset = BaseDataset
DatasetMap = dict[str, BaseDataset]


class MME(BaseMethod):
    """Minimax Entropy"""

    requires_target_data = True
    requires_target_labels = True

    def __init__(
        self,
        batch_size: int,
        batches_per_epoch: int,
        C_confs: dict,
        E_confs: dict,
        lambda_adapt: float,
        lr: float,
        max_epochs: int,
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
        C_confs : dict
            Configurations for the classifier.
        E_confs : dict
            Configurations for the feature extractor.
        lambda_adapt : float
            Lambda hyperparameter. Weight given to the domain discriminator loss.
        lr : float
            Learning rate.
        max_epochs : int
            Maximum number of training epochs.
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
        """
        self.batch_size: int = batch_size
        self.batches_per_epoch: int = batches_per_epoch
        self.device = device(use_device)
        self.lambda_adapt: float = lambda_adapt
        self.lr: float = lr
        self.max_epochs: int = max_epochs
        self.patience: int = patience
        self.positive_ratio: float = positive_ratio
        self.val_ratio: float = validation_ratio
        self.verbose: bool = verbose

        self.C: MLP = MLP(**C_confs).to(self.device)
        self.C_confs: dict = C_confs
        self.C_optim = Adam(self.C.parameters(), lr=self.lr)

        self.E: MLP = MLP(**E_confs).to(self.device)
        self.E_confs: dict = E_confs
        self.E_optim = Adam(self.E.parameters(), lr=self.lr)

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
        # Prepare dataloaders
        if validation is None:
            train_L, validation = self._split_train_val(train_L, self.val_ratio)
        # Dataloader for the classifier (C)
        dataloader_C = BalancedDataloader(
            batch_size=self.batch_size,
            datasets=train_L,
            positive_ratio=self.positive_ratio,
            shuffle=True,
            stop_criterion=self.batches_per_epoch,
        )
        # Dataloader for the unlabelled target data, to compute the entropy
        dataloader_H = DataLoader(  # type: ignore[var-annotated]
            train_U[target_name],  # type: ignore[arg-type, index]
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Reset model if required
        if reset_model:
            if self.verbose:
                print("Resetting model and optimizer.")
            self.C = MLP(**self.C_confs).to(self.device)
            self.C_optim = Adam(self.C.parameters(), lr=self.lr)
            self.E = MLP(**self.E_confs).to(self.device)
            self.E_optim = Adam(self.E.parameters(), lr=self.lr)

        # Establish baseline validation performance
        best_epoch = -1
        best_C_state = deepcopy(self.C.state_dict())
        best_E_state = deepcopy(self.E.state_dict())
        best_score = self._compute_val_score(validation)
        if self.verbose:
            print(f"Initial Val Score: {best_score:9.6f}")

        # Training loop with early stopping
        for epoch in range(self.max_epochs):
            train_loss = self._train_epoch(dataloader_C, dataloader_H)
            val_score = self._compute_val_score(validation)
            if self.verbose:
                print(
                    f"Epoch {epoch + 1:2d}, "
                    f"Train Loss: {train_loss:9.6f}, "
                    f"Val Score: {val_score:9.6f}"
                )
            if val_score > best_score:
                best_epoch = epoch
                best_C_state = deepcopy(self.C.state_dict())
                best_E_state = deepcopy(self.E.state_dict())
                best_score = val_score
            if epoch - best_epoch >= self.patience:
                if self.verbose:
                    print("Early stopping triggered.")
                break

        # Load best model state
        self.C.load_state_dict(best_C_state)
        self.E.load_state_dict(best_E_state)
        if self.verbose:
            print(f"Final Val Score: {best_score:9.6f}")

    def predict(self, dataset: Dataset, domain_name: str) -> NDArray:
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)  # type: ignore[arg-type, var-annotated]
        preds = []
        self.C.eval()
        self.E.eval()
        with no_grad():
            for batch in dataloader:
                batch = self._move_to_device(batch)
                emb = self.E(batch["X"])
                emb = nn.functional.normalize(emb)
                logits = self.C(emb)
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
            batch["X"] = batch["X"].to(self.device)
        if "y" in batch:
            batch["y"] = batch["y"].to(self.device)
        return batch

    def _train_epoch(
        self,
        dataloader_C: BalancedDataloader,
        dataloader_H: DataLoader,
    ) -> float:
        """
        Train the method for one epoch.

        Parameters
        ----------
        dataloader_C : BalancedDataLoader
            DataLoader for the training data of the classifier.
        dataloader_H : DataLoader
            DataLoader for the unlabelled target data, to compute the entropy.

        Returns
        -------
        float
            Average training loss per batch.
        """
        iter(dataloader_C)
        iter_H = iter(dataloader_H)
        self.C.train()
        self.E.train()
        epoch_loss = 0.0

        for _ in range(self.batches_per_epoch):
            self.C_optim.zero_grad()
            self.E_optim.zero_grad()

            # Train classifier
            batch = self._move_to_device(next(dataloader_C))
            emb = self.E(batch["X"])
            emb = nn.functional.normalize(emb)
            loss_class = self.BCE_loss(self.C(emb), batch["y"].unsqueeze(1).float())
            epoch_loss += loss_class.item()

            # Train with entropy
            try:
                batch = next(iter_H)
            except StopIteration:
                iter_H = iter(dataloader_H)
                batch = next(iter_H)
            batch = self._move_to_device(batch)
            emb = self.E(batch["X"])
            emb = nn.functional.normalize(emb)
            logits = self.C(GRL.apply(emb))
            probs = sigmoid(logits)
            loss_entropy = -mean(probs * log2(probs) + (1 - probs) * log2(1 - probs))

            # Update parameters
            loss = loss_class + self.lambda_adapt * loss_entropy
            loss.backward()
            self.C_optim.step()
            self.E_optim.step()

        return epoch_loss / self.batches_per_epoch
