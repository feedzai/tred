from copy import deepcopy

from numpy.typing import NDArray
from torch import Tensor, cat, device, nn, no_grad, sigmoid, zeros
from torch.optim import Adam
from torch.utils.data import DataLoader

from tred.datasets import BaseDataset
from tred.models import MLP
from tred.utils import BalancedDataloader, CombinedDataloader, metric_wrapper

from .base_method import BaseMethod

Dataset = BaseDataset
DatasetMap = dict[str, BaseDataset]


class MAN(BaseMethod):
    """Multinomial Adversarial Networks"""

    requires_target_data = True
    requires_target_labels = True

    def __init__(
        self,
        batch_size: int,
        batches_per_epoch: int,
        C_confs: dict,
        D_confs: dict,
        d_iter: int,
        Ep_confs: dict,
        Es_confs: dict,
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
        D_confs : dict
            Configurations for the domain discriminator.
        d_iter : int
            Number of batches to train the domain discriminator before the classifier.
        Ep_confs : dict
            Configurations for the private feature extractors.
        Es_confs : dict
            Configurations for the shared feature extractor.
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
        self.d_iter: int = d_iter
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

        self.D: MLP = MLP(**D_confs).to(self.device)
        self.D_confs: dict = D_confs
        self.D_optim = Adam(self.D.parameters(), lr=self.lr)

        self.Es: MLP = MLP(**Es_confs).to(self.device)
        self.Es_confs: dict = Es_confs
        self.Es_optim = Adam(self.Es.parameters(), lr=self.lr)

        self.Ep_confs: dict = Ep_confs
        self.Ep: dict[str, MLP] = {}
        self.Ep_optim: dict[str, Adam] = {}
        self.Ep_output_len: int = self.Ep_confs["layer_dims"][-1]

        self.BCE_loss: nn.Module = nn.BCEWithLogitsLoss()
        self.CE_loss: nn.Module = nn.CrossEntropyLoss()
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
        # Dataloaders for the classifiers (C)
        dataloaders_C: dict[str, BalancedDataloader] = {}
        for name_d in train_L:
            dataloaders_C[name_d] = BalancedDataloader(
                batch_size=self.batch_size,
                datasets={name_d: train_L[name_d]},
                positive_ratio=self.positive_ratio,
                shuffle=True,
                stop_criterion=self.batches_per_epoch,
            )
        # Dataloader for the domain discriminator (D)
        dataloader_D = CombinedDataloader(
            batch_size=self.batch_size,
            l_datasets=[train_L] if train_U is None else [train_L, train_U],  # type: ignore[list-item]
            shuffle=True,
            stop_criterion=self.d_iter,
        )

        # Reset model if required
        if reset_model:
            if self.verbose:
                print("Resetting models and optimizers.")
            self.C = MLP(**self.C_confs).to(self.device)
            self.C_optim = Adam(self.C.parameters(), lr=self.lr)
            self.D = MLP(**self.D_confs).to(self.device)
            self.D_optim = Adam(self.D.parameters(), lr=self.lr)
            self.Es = MLP(**self.Es_confs).to(self.device)
            self.Es_optim = Adam(self.Es.parameters(), lr=self.lr)
            self.Ep = {}

        # Initialize private feature extractors and optimizers for first-time domains
        for name_d in train_L:
            if name_d not in self.Ep:
                self.Ep[name_d] = MLP(**self.Ep_confs).to(self.device)
                self.Ep_optim[name_d] = Adam(self.Ep[name_d].parameters(), lr=self.lr)

        # Establish baseline validation performance
        best_epoch = -1
        best_C_state = deepcopy(self.C.state_dict())
        best_D_state = deepcopy(self.D.state_dict())
        best_Es_state = deepcopy(self.Es.state_dict())
        best_Ep_state = {}
        for name_d in train_L:
            best_Ep_state[name_d] = deepcopy(self.Ep[name_d].state_dict())
        best_score = self._compute_val_score(validation)
        if self.verbose:
            print(f"Initial Val Score: {best_score:9.6f}")

        # Training loop with early stopping
        for epoch in range(self.max_epochs):
            train_loss = self._train_epoch(dataloaders_C, dataloader_D)
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
                best_D_state = deepcopy(self.D.state_dict())
                best_Es_state = deepcopy(self.Es.state_dict())
                for name_d in train_L:
                    best_Ep_state[name_d] = deepcopy(self.Ep[name_d].state_dict())
                best_score = val_score
            if epoch - best_epoch >= self.patience:
                if self.verbose:
                    print("Early stopping triggered.")
                break

        # Load best model state
        self.C.load_state_dict(best_C_state)
        self.D.load_state_dict(best_D_state)
        self.Es.load_state_dict(best_Es_state)
        for name_d in train_L:
            self.Ep[name_d].load_state_dict(best_Ep_state[name_d])
        if self.verbose:
            print(f"Final Val Score: {best_score:9.6f}")

    def predict(self, dataset: Dataset, domain_name: str) -> NDArray:
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)  # type: ignore[arg-type, var-annotated]
        preds = []
        self.C.eval()
        self.Es.eval()
        # If the domain was seen during training, use its private feature extractor. Otherwise, use a zero vector.
        if domain_name in self.Ep:
            self.Ep[domain_name].eval()
        else:
            fp_dummy = zeros(self.Ep_output_len, device=self.device).float()
        with no_grad():
            for batch in dataloader:
                batch = self._move_to_device(batch)
                fs = self.Es(batch["X"])
                fp = (
                    self.Ep[domain_name](batch["X"])
                    if domain_name in self.Ep
                    else fp_dummy
                )
                logits = self.C(cat((fs, fp), dim=1))
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
        if "d" in batch:
            batch["d"] = batch["d"].to(self.device)
        return batch

    def _train_epoch(
        self,
        dataloaders_C: dict[str, BalancedDataloader],
        dataloader_D: CombinedDataloader,
    ) -> float:
        """
        Train the method for one epoch.

        Parameters
        ----------
        dataloaders_C : dict[str, BalancedDataloader]
            DataLoaders for the training data of the classifier, with domain name as keys.
        dataloader_D : CombinedDataLoader
            DataLoader for the training data of the domain discriminator.

        Returns
        -------
        float
            Total training loss.
        """
        for name_d in dataloaders_C:
            iter(dataloaders_C[name_d])
            self.Ep[name_d].train()
        self.C.train()
        epoch_loss = 0.0
        batch_i = 0

        while batch_i < self.batches_per_epoch:
            # Train domain discriminator
            self.D.train()
            self.Es.eval()
            self.D_optim.zero_grad()
            for batch in dataloader_D:
                batch = self._move_to_device(batch)
                fs = self.Es(batch["X"])
                loss = self.CE_loss(self.D(fs), batch["d"])
                loss.backward()
            self.D_optim.step()

            # Train feature extractors and classifier
            self.D.eval()
            self.Es.train()
            self.C_optim.zero_grad()
            self.Es_optim.zero_grad()
            for name_d, dataloader in dataloaders_C.items():
                self.Ep_optim[name_d].zero_grad()
                batch = self._move_to_device(next(dataloader))
                fs = self.Es(batch["X"])
                fp = self.Ep[name_d](batch["X"])
                f = cat((fs, fp), dim=1)
                loss_class = self.BCE_loss(self.C(f), batch["y"].unsqueeze(1).float())
                epoch_loss += loss_class.item()
                loss_domain = -self.CE_loss(self.D(fs), batch["d"])
                loss = loss_class + self.lambda_adapt * loss_domain
                loss.backward()
                batch_i += 1
                self.Ep_optim[name_d].step()
            self.C_optim.step()
            self.Es_optim.step()

        return epoch_loss / batch_i
