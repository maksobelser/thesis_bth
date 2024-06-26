from typing import Optional, Dict
import numpy as np
from anndata import AnnData

import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial
from scvi.module.base import BaseModuleClass, auto_move_data
from scvi.module.base._base_module import LossOutput

torch.backends.cudnn.benchmark = True
from transVAE.nn._base_components import Encoder, Decoder, Embedding
from sklearn.metrics import r2_score

# Conditional VAE model
class VAEC(BaseModuleClass):
    """Conditional Variational auto-encoder model.

    This is an implementation of the CondSCVI model

    Parameters
    ----------
    n_input
        Number of input genes
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    dropout_rate
        Dropout rate for the encoder and decoder neural network.
    extra_encoder_kwargs
        Keyword arguments passed into :class:`~scvi.nn.Encoder`.
    extra_decoder_kwargs
        Keyword arguments passed into :class:`~scvi.nn.FCLayers`.
    """

    def __init__(
        self,
        adata: AnnData,
        n_input: int,
        n_hidden: int = 128,
        n_latent: int = 5,
        n_layers: int = 2,
        log_variational: bool = True,
        kl_weight: float = 0.005,
        dropout_rate: float = 0.05,
        px_decoder: bool = False, 
        cov_embed_dims: int = 10,
        extra_encoder_kwargs: Optional[dict] = None,
        extra_decoder_kwargs: Optional[dict] = None,
        initialize_weights: bool = False,
        validation_adatas_dict: Optional[dict] = None,
        original_adata_covariates = None,
    ):
        super().__init__()
        self.dispersion = "gene"
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dropout_rate = dropout_rate
        self.log_variational = log_variational
        self.gene_likelihood = "nb"
        self.latent_distribution = "normal"
        self.px_decoder = px_decoder
        # Automatically deactivate if useless
        self.n_batch = 0
        self.kl_weight = kl_weight
        self.val_adatas = validation_adatas_dict
        self.original_adata_covariates = original_adata_covariates
                
        # Initialize embeddings for covariates with high cardinality
        if "covariates_embed" in adata.obsm.keys():
            self.embed_cov_sizes = adata.obsm["covariates_embed"].nunique().tolist()
            self.cov_embeddings = torch.nn.ModuleList([
                Embedding(size=size, cov_embed_dims=cov_embed_dims) for size in self.embed_cov_sizes])
            self.embed_cov = True
            self.n_cov_embed = len(adata.obsm["covariates_embed"].columns)*cov_embed_dims
        else:
            self.embed_cov = False
            self.n_cov_embed = 0
        
        if "covariates" in adata.obsm.keys():
            self.n_cov = len(adata.obsm["covariates"].columns)
        else:
            self.n_cov = 0
            
        self.n_cov_total = self.n_cov_embed + self.n_cov
        if self.n_cov_total == 0:
            self.n_cov_total = None 

        self.z_encoder = Encoder(
            n_input=n_input,
            n_output=n_latent,  
            n_cat=self.n_cov_total,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate = dropout_rate,
            var_eps=1e-4
        )

        self.decoder = Decoder(
            n_input=n_latent,
            n_output=n_input,
            n_cat=self.n_cov_total,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate = dropout_rate,
            var_eps=1e-4
        )
        
        if initialize_weights:
            self.z_encoder.initialize_weights()
            self.decoder.initialize_weights()
        
    def _get_cov(self, tensors):
        
        cov_list = []

        # Check if 'covariates' key exists in tensors and its size
        if 'covariates' in tensors and tensors['covariates'].shape[1] > 0:
            cov_list.append(tensors['covariates'])
        
        # Check if 'covariates_embed' key exists in tensors
        if self.embed_cov:
            # Dynamically create embeddings if not already initialized
            if not hasattr(self, 'cov_embeddings'):
                self.embed_cov_sizes = [tensors['covariates_embed'][:, i].max().item() + 1 for i in range(tensors['covariates_embed'].shape[1])]
                self.cov_embeddings = torch.nn.ModuleList([
                    Embedding(size=size, cov_embed_dims=cov_embed_dims) for size in self.embed_cov_sizes])

            # Append the embeddings
            cov_list.extend([embedding(tensors['covariates_embed'][:, i].int()) 
                             for i, embedding in enumerate(self.cov_embeddings)])
            
        # Concatenate along dimension 1 or return None if empty
        return torch.cat(cov_list, dim=1) if cov_list else None
        
    def _get_inference_input(self, tensors, **kwargs):
        """Parse the dictionary to get appropriate args"""
        
        expr = tensors[REGISTRY_KEYS.X_KEY]
        cov = self._get_cov(tensors=tensors)
        input_dict = dict(expr=expr, cov=cov)
        
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        """
        Parse the dictionary to get appropriate args
        :param cov_replace: Replace cov from tensors with this covariate vector (for predict)
        """

        z = inference_outputs["z"]
        cov = self._get_cov(tensors=tensors)
        input_dict = dict(z=z, cov=cov)
        
        return input_dict

    @auto_move_data
    def inference(self, expr, cov):
        """High level inference method.

        Runs the inference (encoder) model.
        """
        q_m, q_v, latent = self.z_encoder(expr, cov)        
        outputs = {"z": latent, "q_m": q_m, "q_v": q_v}
        
        return outputs

    @auto_move_data
    def generative(self, z, cov):
        """Runs the generative model."""
        
        p = self.decoder(z, cov)
        EPS_EXP = 22
        EPS_LOG = 1e-8
        p = torch.exp(torch.minimum(p, torch.ones_like(p) * EPS_EXP))
        p = torch.maximum(p, torch.ones_like(p) * EPS_LOG)
        
        return {"px": p}

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 0.0005,
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]
        qz_m = inference_outputs["q_m"]
        qz_v = inference_outputs["q_v"]
        p = generative_outputs["px"]
        
        kld = kl(
            Normal(qz_m, torch.sqrt(qz_v)),
            Normal(0, 1),
        ).sum(dim=1)           
        
        rl = self.get_reconstruction_loss(p, x)
        loss = (0.5 * rl + 0.5 * (kld * self.kl_weight)).mean()

        return LossOutput(loss=loss, reconstruction_loss=rl, kl_local=kld)
        
    def get_reconstruction_loss(self, x, px) -> torch.Tensor:
        x = x[0] if isinstance(x, tuple) else x
        px = torch.tensor(px) if not isinstance(px, torch.Tensor) else px
        loss = ((x - px) ** 2).sum(dim=1)
        return loss

    def compute_validation_metrics(self):
        # device
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        validation_loss = {}
        csdl = self.val_adatas["csdl_adata_to_predict"]
        
        latent = []
        qm = []
        qv = []
        expr = []
        for tensors in csdl:
            tensors = {k: v.to(device) for k, v in tensors.items()}  # Move tensors to the device
            inference_inputs = self._get_inference_input(tensors)
            outputs = self.inference(**inference_inputs)
            z = outputs["z"]
            latent += [z]

        latent = torch.cat(latent)
        for name, data in self.val_adatas.items():
            if name == "csdl_adata_to_predict":
                continue
            cov = self._get_cov(tensors=data["generative_tensors"])
            gt = data["ground_truth"]
            px = self.generative(z = latent, cov = cov)["px"].cpu().detach().numpy().mean(axis = 0)
            r2 = r2_score(gt, px)
            
            validation_loss[f"{name}_r2"] = r2
        
        if len(validation_loss) > 1:
            validation_loss["mean_r2"] = sum(validation_loss.values()) / len(validation_loss)
        
        return validation_loss
