import torch
import torch.nn as nn
import collections
from typing import Callable, Optional
from torch.nn import Module
from torch.distributions import Normal
import torch.nn.init as init

def _identity(x):
    return x

class FCLayers(nn.Module):
    """A helper class to build fully-connected layers for a neural network."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        nn.Sequential(
                            nn.Linear(
                                n_in + n_cat * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
            )
        )

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def set_online_update_hooks(self, hook_first_layer=True):
        """Set online update hooks."""
        self.hooks = []

        def _hook_fn_weight(grad):
            new_grad = torch.zeros_like(grad)
            new_grad[:, -n_cat:] = grad[:, -n_cat:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def forward(self, x: torch.Tensor, cat_tensor: torch.Tensor):
        """Forward computation on ``x``."""
        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat(
                                [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                            )
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                cat_tensor_layer = cat_tensor.unsqueeze(0).expand(
                                    (x.size(0), cat_tensor.size(0), cat_tensor.size(1))
                                )
                            else:
                                cat_tensor_layer = cat_tensor
                            x = torch.cat((x, cat_tensor_layer), dim=-1)
                        x = layer(x)
        return x

    
class Encoder(nn.Module):
    """Encode data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat
        Size of cat tensor
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        Defaults to :meth:`torch.exp`.
    return_dist
        Return directly the distribution of z instead of its parameters.
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
        return_dist: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat=n_cat,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.return_dist = return_dist

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity
        self.var_activation = torch.exp if var_activation is None else var_activation
        
    def initialize_weights(self):
        # perventing nan values for loc parameter (possibly due to bad init), 
        # alternatively the lr is too high and I get exploding gradients.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, nonlinearity='relu')
                init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, cov: torch.Tensor):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for latent distribution
        q = self.encoder(x, cov)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())
        if self.return_dist:
            return dist, latent
        return q_m, q_v, latent

# Decoder SCGEN
class Decoder(nn.Module):
    """
    Decodes data from latent space to data space.

    ``n_input`` dimensions to ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    Output is the mean and variance of a multivariate Gaussian

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    kwargs
        Keyword args for :class:`~scvi.modules._base.FCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        
        kwargs.pop('var_eps', None)
        self.decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat=n_cat,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        
        self.linear_out = nn.Linear(n_hidden, n_output)
        
    def initialize_weights(self):
        # perventing nan values for loc parameter (possibly due to bad init), 
        # alternatively the lr is too high and I get exploding gradients.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, nonlinearity='relu')
                init.zeros_(module.bias)


    def forward(self, x: torch.Tensor, cov: torch.Tensor):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        Parameters
        ----------
        x
            tensor with shape ``(n_input,)``
        cat_list
            list of category membership(s) for this sample

        """
        p = self.linear_out(self.decoder(x, cov))
        return p
    
class Embedding(Module):
    def __init__(self,
                 size,
                 cov_embed_dims: int = 10,
                 normalize: bool = True
                 ):
        """
        Creates dataset embeddings.
        """
        super().__init__()
        self.normalize = normalize
        self.embedding = torch.nn.Embedding(size, cov_embed_dims)
        if self.normalize:
            # TODO this could probably be implemented more efficiently as embed gives same result for every sample in
            #  a given class. However, if we have many balanced classes there wont be many repetitions within minibatch
            self.layer_norm = torch.nn.LayerNorm(cov_embed_dims, elementwise_affine=False)

    def forward(self, x):
        x = self.embedding(x)
        if self.normalize:
            x = self.layer_norm(x)
        return x