from copy import deepcopy

from numpy.typing import NDArray
from torch import Tensor, cat, device, nn, no_grad, sigmoid, zeros
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader

from tred.datasets import CSVDataset
from tred.models import MLP
from tred.utils import BalancedDataloader, metric_wrapper

from .base_method import BaseMethod

Dataset = CSVDataset
DatasetMap = dict[str, CSVDataset]


class MTAE(BaseMethod):
    """Multi-Task Autoencoder"""

    requires_target_data = False
    requires_target_labels = False

    def __init__(
        self,
        batch_size: int,
        batches_per_epoch: int,
        C_confs: dict,
        D_confs: dict,
        E_confs: dict,
        lr: float,
        max_epochs: int,
        max_pretrain_epochs: int,
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
        D_confs : dict
            Configurations for the decoder.
        E_confs : dict
            Configurations for the encoder.
        lr : float
            Learning rate.
        max_epochs : int
            Maximum number of training epochs.
        max_pretrain_epochs : int
            Maximum number of pretraining epochs.
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
        self.cardinalities: list[int] = E_confs["cardinalities"]
        self.device = device(use_device)
        self.input_dim_num: int = E_confs["input_dim_num"]
        self.lr: float = lr
        self.max_epochs: int = max_epochs
        self.max_pretrain_epochs: int = max_pretrain_epochs
        self.patience: int = patience
        self.positive_ratio: float = positive_ratio
        self.val_ratio: float = validation_ratio
        self.verbose: bool = verbose

        self.C: MLP = MLP(**C_confs).to(self.device)
        self.C_confs: dict = C_confs
        self.C_optim = Adam(self.C.parameters(), lr=self.lr)

        self.D: dict[str, MLP] = {}
        self.D_confs: dict = D_confs
        self.D_optim: dict[str, Adam] = {}

        self.E: MLP = MLP(**E_confs).to(self.device)
        self.E_confs: dict = E_confs
        self.E_optim = Adam(self.E.parameters(), lr=self.lr)

        self.BCE_loss: nn.Module = nn.BCEWithLogitsLoss()
        self.val_metric = metric_wrapper(
            validation_metric["metric"], **validation_metric.get("config", {})
        )

    def fit(  # type: ignore[override]
        self,
        train_L: DatasetMap,
        target_name: str,
        reset_model: bool = False,
        train_U: DatasetMap | None = None,
        validation: DatasetMap | None = None,
    ) -> None:
        # Prepare dataloaders
        train_L = {
            name_d: dataset
            for name_d, dataset in train_L.items()
            if name_d != target_name
        }
        if validation is None:
            train_L, validation = self._split_train_val(train_L, self.val_ratio)  # type: ignore[assignment, arg-type]
        dataloader_C = BalancedDataloader(
            batch_size=self.batch_size,
            datasets=train_L,
            positive_ratio=self.positive_ratio,
            shuffle=True,
            stop_criterion=self.batches_per_epoch,
        )

        # Reset model if required
        needs_pretrain = False
        if reset_model:
            if self.verbose:
                print("Resetting model and optimizer.")
            self.C = MLP(**self.C_confs).to(self.device)
            self.C_optim = Adam(self.C.parameters(), lr=self.lr)
            self.D = {}
            self.E = MLP(**self.E_confs).to(self.device)
            self.E_optim = Adam(self.E.parameters(), lr=self.lr)
            needs_pretrain = True
        for name_d in train_L:
            if name_d not in self.D:
                self.D[name_d] = MLP(**self.D_confs).to(self.device)
                self.D_optim[name_d] = Adam(self.D[name_d].parameters(), lr=self.lr)
                needs_pretrain = True

        # If the model was reset or there is a new domain, pretrain
        if needs_pretrain:
            self._pretrain_hook(train_L)

        # Establish baseline validation performance
        best_epoch = -1
        best_C_state = deepcopy(self.C.state_dict())
        best_E_state = deepcopy(self.E.state_dict())
        best_score = self._compute_val_score(validation)  # type: ignore[arg-type]
        if self.verbose:
            print(f"Initial Val Score: {best_score:9.6f}")

        # Training loop with early stopping
        for epoch in range(self.max_epochs):
            train_loss = self._train_epoch(dataloader_C)
            val_score = self._compute_val_score(validation)  # type: ignore[arg-type]
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

    def predict(self, dataset: CSVDataset, domain_name: str) -> NDArray:  # type: ignore[override]
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)  # type: ignore[arg-type, var-annotated]
        preds = []
        self.C.eval()
        self.E.eval()
        with no_grad():
            for batch in dataloader:
                batch = self._move_to_device(batch)
                logits = self.C(self.E(batch["X"]))
                preds.append(sigmoid(logits))
        return cat(preds).cpu().numpy().squeeze(1)

    def _compute_val_score(self, datasets: dict[str, CSVDataset]) -> float:
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

    def _pretrain_epoch(self, dataloaders: dict[str, BalancedDataloader]) -> float:
        domain_names = list(dataloaders.keys())
        for name_d, dl in dataloaders.items():
            iter(dl)
            self.D[name_d].train()
        self.E.train()
        epoch_loss = 0.0
        batch_i = 0

        while batch_i <= self.batches_per_epoch:
            for t in domain_names:
                self.D_optim[t].zero_grad()
                self.E_optim.zero_grad()
                for s in domain_names:
                    # Fetch next target batch
                    try:
                        batch_target = next(dataloaders[t])
                    except StopIteration:
                        iter(dataloaders[t])
                        batch_target = next(dataloaders[t])
                    batch_target = self._move_to_device(batch_target)

                    # Fetch next source batch
                    try:
                        batch_source = next(dataloaders[s])
                    except StopIteration:
                        iter(dataloaders[s])
                        batch_source = next(dataloaders[s])
                    batch_source = self._move_to_device(batch_source)

                    # Train with reconstruction loss
                    batch_emb = self.E(batch_source["X"])
                    batch_out = self.D[t](batch_emb)
                    loss, _, _ = self._reconstruction_loss(batch_out, batch_target["X"])
                    epoch_loss += loss.item()
                    loss.backward()
                    batch_i += 1

                self.D_optim[t].step()
                self.E_optim.step()

        return epoch_loss / batch_i

    def _pretrain_hook(self, train_L: DatasetMap) -> None:
        """
        Pretrain the shared encoder to minimize the loss of all decoders.
        """
        # Prepare dataloaders
        dataloaders: dict[str, BalancedDataloader] = {}
        for name_d in train_L:
            dataloaders[name_d] = BalancedDataloader(
                batch_size=self.batch_size,
                datasets={name_d: train_L[name_d]},
                positive_ratio=self.positive_ratio,
                drop_last=True,
                shuffle=True,
            )

        # Early stopping variables
        best_E_state = None
        best_epoch = -1
        best_loss = float("inf")

        # Training loop with early stopping
        for epoch in range(self.max_pretrain_epochs):
            pretrain_loss = self._pretrain_epoch(dataloaders)
            if self.verbose:
                print(f"Pre-train epoch {epoch + 1:2d}, Loss: {pretrain_loss:.3f}")
            if pretrain_loss < best_loss:
                best_E_state = deepcopy(self.E.state_dict())
                best_epoch = epoch
                best_loss = pretrain_loss
            if epoch - best_epoch >= self.patience:
                if self.verbose:
                    print("Pretrain early stopping triggered.")
                break

        # Load best model state
        self.E.load_state_dict(best_E_state)  # type: ignore[arg-type]

    def _reconstruction_loss(
        self, output: Tensor, target: Tensor
    ) -> tuple[Tensor, float, float]:
        """
        Compute the reconstruction loss.

        This combines:
          - Mean squared error (MSE) over numerical features, and
          - Cross-entropy over each categorical feature's logits.

        Parameters
        ----------
        target : Tensor
            Input tensor of shape (B, input_dim_num + num_categorical).
        output : Tensor
            Decoder output tensor of shape
            (B, input_dim_num + sum(cardinalities)), where
            the first `input_dim_num` columns are reconstructions of numerical
            features, followed by contiguous logits blocks per categorical feature.

        Returns
        -------
        (Tensor, float, float)
            - Scalar total loss (numerical + categorical), averaged over batch.
            - Numerical loss component (float).
            - Categorical loss component (float).
        """
        # Compute loss on numerical features
        output_num: Tensor = output[:, : self.input_dim_num]
        target_num: Tensor = target[:, : self.input_dim_num].detach()
        num_loss = ((output_num - target_num) ** 2).sum(dim=1)

        # Compute loss on categorical features
        idx = self.input_dim_num
        target_cat = target[:, self.input_dim_num :].long()
        cat_loss = zeros(output.shape[0], device=output.device)
        for cat_i, num_classes in enumerate(self.cardinalities):
            output_cat = output[:, idx : (idx + num_classes)]
            cat_loss += cross_entropy(
                output_cat, target_cat[:, cat_i], reduction="none"
            )
            idx += num_classes

        # Compute aggregate loss
        total_loss = num_loss + cat_loss
        return total_loss.mean(), num_loss.mean().item(), cat_loss.mean().item()

    def _train_epoch(self, dataloader: BalancedDataloader) -> float:
        """
        Train the method for one epoch.

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
        self.C.train()
        self.E.train()
        epoch_loss = 0.0

        for _ in range(self.batches_per_epoch):
            self.C_optim.zero_grad()
            self.E_optim.zero_grad()
            batch = self._move_to_device(next(dataloader))
            logits = self.C(self.E(batch["X"]))
            loss = self.BCE_loss(logits, batch["y"].unsqueeze(1).float())
            epoch_loss += loss.item()
            loss.backward()
            self.C_optim.step()
            self.E_optim.step()

        return epoch_loss / self.batches_per_epoch
