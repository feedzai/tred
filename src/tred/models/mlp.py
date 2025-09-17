from math import log2
from typing import Literal

from torch import Tensor, cat, nn


class MLP(nn.Module):
    """
    A modular Multi-Layer Perceptron (MLP), designed for tabular data.

    Supports numerical and categorical inputs, dropout, normalisation, and residual connections.
    Each layer uses a pre-activation block
     (norm → activation → dropout → linear)
    with optional residual connection.
    """

    def __init__(
        self,
        activation: str,
        cardinalities: list[int],
        dropout: float,
        input_dim_num: int,
        layer_dims: list[int],
        norm: Literal["none", "batch_norm", "layer_norm"],
        output_activation: str,
        residual: bool,
    ) -> None:
        """
        Initialize the MLP model.

        Parameters
        ----------
        activation : str
            Activation function applied between layers. This should match the name of
            an activation function from `torch.nn.functional` (e.g., "relu", "gelu").
        cardinalities : list[int]
            Cardinalities for categorical features.
            The i-th value specifies the number of unique values in the i-th categorical feature.
            Categorical features are embedded as dense vectors.
        dropout : float
            Dropout rate applied after activation functions. If 0.0, dropout is disabled.
        input_dim_num : int
            Number of numerical features in the input.
        layer_dims : list[int]
            Number of units in each layer of the MLP (after the input).
        norm : Literal["none", "batch_norm", "layer_norm"]
            Type of normalisation to use before linear layers:
            - "none": No normalisation.
            - "batch_norm": Batch normalisation.
            - "layer_norm": Layer normalisation.
        output_activation : str
            Activation function applied to the final output. If "none", no activation is applied.
            Otherwise, this should match the name of an activation function
            from `torch.nn.functional` (e.g., "relu", "gelu").
        residual : bool
            Whether to use residual connections between layers.
        """
        super().__init__()
        self.activation = getattr(nn.functional, activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.input_dim_num = input_dim_num
        self.output_activation = (
            getattr(nn.functional, output_activation)
            if output_activation != "none"
            else None
        )
        self.residual = residual

        # Embedding layers for categorical features
        self.embedding_layers = (
            nn.ModuleList(
                [nn.Embedding(c, self.embedding_dim(c)) for c in cardinalities]
            )
            if cardinalities
            else None
        )

        # Linear layers
        input_dim = input_dim_num + sum(self.embedding_dim(c) for c in cardinalities)
        dims = [input_dim] + layer_dims
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        # Normalisation layers
        self.norms = None
        if norm == "batch_norm":
            self.norms = nn.ModuleList(
                [nn.BatchNorm1d(dims[i]) for i in range(len(dims) - 1)]
            )
        if norm == "layer_norm":
            self.norms = nn.ModuleList(
                [nn.LayerNorm(dims[i]) for i in range(len(dims) - 1)]
            )

        # Residual projections
        self.projections = None
        if self.residual:
            self.projections = nn.ModuleList(
                [
                    (
                        nn.Identity()
                        if dims[i] == dims[i + 1]
                        else nn.Linear(dims[i], dims[i + 1])
                    )
                    for i in range(len(dims) - 1)
                ]
            )

    def embed(self, x: Tensor) -> Tensor:
        """
        Apply embeddings to the categorical features and concatenate them with the numerical features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim), where the first `input_dim_num` columns
            correspond to numerical features and the remaining columns are categorical indices.

        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size, input_dim_num + sum(embedding_dims))
            containing the numerical features and embedded categorical features.
        """
        if self.embedding_layers:
            x_num = x[:, : self.input_dim_num]
            x_cat = x[:, self.input_dim_num :].int()
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embedding_layers)]
            x_emb = cat(embs, dim=1)
            x = cat((x_num, x_emb), dim=1)
        return x.float()

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim), where `input_dim` can include
            both numerical and categorical features (in this order).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, layer_dims[-1]).
        """
        x = self.embed(x)
        for i in range(len(self.layers)):
            out = x
            # Normalisation
            if self.norms:
                out = self.norms[i](out)
            # Activation
            out = self.activation(out)
            # Dropout
            if self.dropout:
                out = self.dropout(out)
            # Linear Layer
            out = self.layers[i](out)
            # Residual connection
            if self.residual:
                skip = self.projections[i](x)  # type: ignore[index]
                out = out + skip
            x = out

        if self.output_activation:
            x = self.output_activation(x)
        return x

    @staticmethod
    def embedding_dim(cardinality: int) -> int:
        """
        Compute the embedding dimension for a categorical feature.

        Parameters
        ----------
        cardinality : int
            The number of unique values in the categorical feature.

        Returns
        -------
        int
            The embedding dimension, computed as log2(cardinality) + 1 for cardinalities > 2.
        """
        return int(log2(cardinality) + 1) if cardinality > 2 else 1
