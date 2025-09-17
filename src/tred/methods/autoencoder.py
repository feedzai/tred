from copy import deepcopy

from torch import Tensor, device, no_grad, zeros
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader

from tred.datasets import CSVDataset
from tred.models import MLP
from tred.utils import CombinedDataloader

Dataset = CSVDataset
DatasetMap = dict[str, CSVDataset]


class Autoencoder:
    """
    Autoencoder for pretraining a tabular feature extractor.

    The MLP encoder `E` maps inputs to a latent representation.
    The MLP decoder `D` reconstructs the original input.

    Notes
    -----
    - Input layout must be `[numerical..., categorical...]` where the categorical
      block is represented by integer-encoded category indices (one per feature).
    - Decoder output layout mirrors `[recon_numerical..., logits_cat_0..., logits_cat_1..., ...]`.
    """

    def __init__(
        self,
        batch_size: int,
        batches_per_epoch: int,
        D_confs: dict,
        E_confs: dict,
        lr: float,
        max_epochs: int,
        patience: int,
        use_device: str = "cpu",
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
        D_confs : dict
            Configurations for the decoder.
        E_confs : dict
            Configurations for the encoder. Must include:
            - "cardinalities": list[int], the categorical cardinalities in order;
            - "input_dim_num": int, number of numerical features at the start of input.
        lr : float
            Learning rate.
        max_epochs : int
            Maximum number of training epochs.
        patience : int
            Early-stopping patience (epochs without validation improvement).
        use_device : str, default="cpu"
            Device to use for computation (e.g., 'cpu', 'cuda').
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
        self.patience: int = patience
        self.val_ratio: float = validation_ratio
        self.verbose: bool = verbose

        self.D_confs: dict = D_confs
        self.D: MLP = MLP(**self.D_confs).to(self.device)
        self.D_optim = Adam(self.D.parameters(), lr=self.lr)

        self.E_confs: dict = E_confs
        self.E: MLP = MLP(**self.E_confs).to(self.device)
        self.E_optim = Adam(self.E.parameters(), lr=self.lr)

    def evaluate(
        self, datasets: dict[str, CSVDataset], print_intermediate: bool = False
    ) -> float:
        """
        Evaluate the reconstruction quality on datasets.

        Computes the mean *negative* reconstruction loss across domains
        (higher is better), where the reconstruction loss is:
          MSE over numerical features + cross-entropy over each categorical feature.

        Parameters
        ----------
        datasets : DatasetMap
            Datasets by domain.
        print_intermediate : bool, default=False
            If True and `verbose`, prints per-domain losses.

        Returns
        -------
        float
            Mean negative reconstruction loss over domains.
        """
        self.D.eval()
        self.E.eval()
        scores_per_dataset = []

        for name_d, dataset in datasets.items():
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)  # type: ignore[arg-type, var-annotated]
            full_loss = 0.0
            num_loss = 0.0
            cat_loss = 0.0
            with no_grad():
                for batch in dataloader:
                    batch_in = self._move_to_device(batch)["X"]
                    batch_out = self.D(self.E(batch_in))
                    losses = self._reconstruction_loss(batch_in, batch_out)
                    full_loss += losses[0].item()
                    num_loss += losses[1]
                    cat_loss += losses[2]
            if self.verbose and print_intermediate:
                print(
                    f"Autoencoder.evaluate: dataset {name_d}, "
                    f"full_loss {full_loss / len(dataloader):6.3f}, "
                    f"num_loss {num_loss / len(dataloader):6.3f}, "
                    f"cat_loss {cat_loss / len(dataloader):6.3f}"
                )
            scores_per_dataset.append(-full_loss / len(dataloader))

        return sum(scores_per_dataset) / len(scores_per_dataset)

    def fit(
        self,
        train: DatasetMap,
        reset_model: bool = False,
        validation: DatasetMap | None = None,
    ) -> None:
        """
        Fit the method given domain-partitioned datasets.

        Parameters
        ----------
        train : DatasetMap
            Training datasets by domain.
        reset_model : bool, default=False
            Whether to reinitialize the encoder, the decoder, and their optimizers.
        validation : DatasetMap or None, default=None
            Validation datasets by domain. If None, a split is created from `train`.
        """
        # Prepare dataloader
        if validation is None:
            train, validation = self._split_train_val(train, self.val_ratio)
        dataloader = CombinedDataloader(
            batch_size=self.batch_size,
            l_datasets=[train],  # type: ignore[list-item]
            shuffle=True,
            stop_criterion=self.batches_per_epoch,
        )

        # Reset model if required
        if reset_model:
            if self.verbose:
                print("Resetting model and optimizer.")
            self.D = MLP(**self.D_confs).to(self.device)
            self.D_optim = Adam(self.E.parameters(), lr=self.lr)
            self.E = MLP(**self.E_confs).to(self.device)
            self.E_optim = Adam(self.E.parameters(), lr=self.lr)

        # Establish baseline validation performance
        best_epoch = -1
        best_D_state = deepcopy(self.D.state_dict())
        best_E_state = deepcopy(self.E.state_dict())
        best_score = self.evaluate(validation)
        if self.verbose:
            print(f"Initial Val Score: {best_score:9.6f}")

        # Training loop with early stopping
        for epoch in range(self.max_epochs):
            train_loss = self._train_epoch(dataloader)
            val_score = self.evaluate(validation)
            if self.verbose:
                print(
                    f"Epoch {epoch + 1:2d}, "
                    f"Train Loss: {train_loss:9.6f}, "
                    f"Val Score: {val_score:9.6f}"
                )
            if val_score > best_score:
                best_epoch = epoch
                best_D_state = deepcopy(self.D.state_dict())
                best_E_state = deepcopy(self.E.state_dict())
                best_score = val_score
            if epoch - best_epoch >= self.patience:
                if self.verbose:
                    print("Early stopping triggered.")
                break

        # Load best model state
        self.D.load_state_dict(best_D_state)  # type: ignore[arg-type]
        self.E.load_state_dict(best_E_state)  # type: ignore[arg-type]
        if self.verbose:
            print(f"Final Val Score: {best_score:9.6f}")

    def _move_to_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Move a batch of data to the specified device.
        """
        if "X" in batch:
            batch["X"] = batch["X"].to(self.device)
        return batch

    def _reconstruction_loss(
        self, target: Tensor, output: Tensor
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

    def _split_train_val(
        self, datasets: DatasetMap, val_ratio: float = 0.3
    ) -> tuple[DatasetMap, DatasetMap]:
        """
        Split the datasets of each domain into training and validation sets.

        Parameters
        ----------
        datasets : DatasetMap
            Dictionary with domain names as keys and datasets as values.
        val_ratio : float, default=0.3
            Proportion of the time range used for the validation dataset.

        Returns
        -------
        tuple[DatasetMap, DatasetMap]
            Training datasets and validation datasets.
        """
        train_datasets: DatasetMap = {}
        val_datasets: DatasetMap = {}

        for name_d, dataset in datasets.items():
            t, v = dataset.split(split_ratio=1.0 - val_ratio)
            train_datasets[name_d] = t
            val_datasets[name_d] = v

        return train_datasets, val_datasets

    def _train_epoch(self, dataloader: CombinedDataloader) -> float:
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
        self.D.train()
        self.E.train()
        epoch_loss = 0.0

        for _ in range(self.batches_per_epoch):
            self.D_optim.zero_grad()
            self.E_optim.zero_grad()
            batch_in = self._move_to_device(next(dataloader))["X"]
            batch_out = self.D(self.E(batch_in))
            loss, _, _ = self._reconstruction_loss(batch_in, batch_out)
            epoch_loss += loss.item()
            loss.backward()
            self.D_optim.step()
            self.E_optim.step()

        return epoch_loss / self.batches_per_epoch
