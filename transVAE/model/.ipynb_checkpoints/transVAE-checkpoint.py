from scvi.data.fields import CategoricalObsField, LayerField, ObsmField
from scvi.data import AnnDataManager
from anndata import AnnData
from scvi.utils import setup_anndata_dsp
from scvi import REGISTRY_KEYS
import numpy as np
from typing import Optional, List, Dict, Sequence, Tuple
import torch
from scvi.module.base import auto_move_data

from transVAE.module._base_module import VAEC
from transVAE.module._utils import prepare_metadata
from scvi.model.base import VAEMixin, BaseModelClass
from transVAE.model._training import UnsupervisedTrainingMixin

class transVAE(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Implementation of VAE model
    Parameters
    ----------
    adata
        AnnData object that has been registered. 
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    """
    def __init__(
        self,
        adata: AnnData,
        n_labels: list = 0,
        n_hidden: int = 800,
        n_latent: int = 100,
        n_layers: int = 2,
        dropout_rate: float = 0.2,
        cov_embed_dims: int = 10,
        kl_weight: float = 0.005,
        initialize_weights: bool = False,
        validation_adatas_dict: Optional[Dict] = None,
        **model_kwargs,
    ):
        super().__init__(adata)
        # assign n_input
        n_input = self.summary_stats.n_vars
        if validation_adatas_dict is not None:
            validation_adatas_dict = self.prepare_validation_data(validation_adatas_dict)
        
        # cVAE
        self.module = VAEC(
            adata = adata,
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            cov_embed_dims = cov_embed_dims,
            kl_weight = kl_weight,
            initialize_weights = initialize_weights,
            validation_adatas_dict = validation_adatas_dict,
            original_adata_covariates = self.adata.obsm,
            **model_kwargs,
        )
        self._model_summary_string = (
            "Model with the following params: n_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: {}, n_labels {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            n_labels
        )

        self.init_params_ = self._get_init_params(locals())

    def prepare_validation_data(self, 
                                validation_adatas_dict: Dict,
                                batch_size = 4048) -> Dict:
        """
        Prepares validation data for models that require embedding categorical covariates, 
        and validation of input AnnData objects. This method primarily processes and 
        augments `AnnData` objects with necessary metadata, embeddings, and data loaders 
        for downstream model validation.

        The method operates in-place, modifying `validation_adatas_dict` to include 
        data loaders and processed AnnData objects. It relies on a 'control' key being 
        present within `validation_adatas_dict`, and 'translation dict' in the `obsm` 
        attribute of AnnData objects for embedding.

        Parameters:
        - validation_adatas_dict (dict): A dictionary where keys are descriptive names and 
          values are AnnData objects intended for validation. Must include keys 'adata_to_predict', 
          'translate_dicts', and 'ground_truths' for operations within the method.

        Returns:
        - dict: Updated validation_adatas_dict with data loaders and other relevant 
          data structures needed for validation.

        Raises:
        - ValueError: If 'control' key is not present in `validation_adatas_dict`.
        - KeyError: If a key from `translate_dicts` is not present in the `obs` attribute 
          of an AnnData object.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        adata_to_predict = validation_adatas_dict["adata_to_predict"]
        translate_dicts = validation_adatas_dict["translate_dicts"]
        ground_truths = validation_adatas_dict["ground_truths"]

        # Extract covariate information from self.adata
        categorical_covariate_keys = self.adata.uns["covariates_dict"]["categorical"]
        categorical_covariate_embed_keys = self.adata.uns["covariates_dict"]["categorical_embed"]
        orders = self.adata.uns["covariate_orders"]

        # Prepare metadata for validation AnnData
        covariates, covariates_embed, orders_dict, cov_dict = prepare_metadata(
            meta_data=adata_to_predict.obs,
            cov_cat_keys=categorical_covariate_keys,
            cov_cat_embed_keys=categorical_covariate_embed_keys,
            orders=orders
        )

        # Update the uns and obsm of validation AnnData with new metadata and embeddings
        adata_to_predict.uns['covariate_orders'] = orders_dict
        adata_to_predict.uns['covariates_dict'] = cov_dict
        if categorical_covariate_keys is not None:
            adata_to_predict.obsm['covariates'] = covariates
        if categorical_covariate_embed_keys is not None:
            adata_to_predict.obsm['covariates_embed'] = covariates_embed

        # Validate the AnnData object and create a DataLoader
        adata_to_predict = self._validate_anndata(adata_to_predict)
        scdl = self._make_data_loader(
            adata=adata_to_predict, indices=None, batch_size=batch_size
        )
        
        validation_dict = dict()
        validation_dict["csdl_adata_to_predict"] = scdl
        
        for ground_truth_name, translate_dict in translate_dicts.items():
            # latent variable switching
            for column in translate_dict.keys():
                if column not in list(adata_to_predict.obs.columns):
                    raise KeyError("Dict key from translate_dict not found in adata.obs.")
                adata_to_predict.obs[column] = translate_dict[column]

            covariates, covariates_embed, orders_dict, cov_dict = prepare_metadata(
                meta_data=adata_to_predict.obs,
                cov_cat_keys=categorical_covariate_keys,
                cov_cat_embed_keys=categorical_covariate_embed_keys,
                orders=self.adata.uns["covariate_orders"]
            )

            tensors = {
                "covariates": torch.Tensor(covariates.values).to(device), 
                "covariates_embed": torch.Tensor(covariates_embed.values).to(device)
            }
            
            gt = ground_truths[ground_truth_name].X.toarray().mean(axis = 0)
                
            # Update the dictionary with the DataLoader
            validation_dict[ground_truth_name] = {"ground_truth":gt, "generative_tensors": tensors}

        return validation_dict
    
    @auto_move_data
    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        mc_samples: int = 5000,
        batch_size: Optional[int] = None,
        
    ) -> np.ndarray:
        
        """Return the latent representation for each cell.

        This is typically denoted as :math:`z_n`.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        Returns
        -------
        Low-dimensional representation for each cell.
        """
        
        self._check_if_trained(warn=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        categorical_covariate_keys = self.adata.uns["covariates_dict"]["categorical"]
        categorical_covariate_embed_keys = self.adata.uns["covariates_dict"]["categorical_embed"]
        
        covariates, covariates_embed, orders_dict, cov_dict = prepare_metadata(
            meta_data=adata.obs,
            cov_cat_keys=categorical_covariate_keys,
            cov_cat_embed_keys=categorical_covariate_embed_keys,
            orders=self.adata.uns["covariate_orders"]
        )
        
        adata.uns['covariate_orders'] = orders_dict
        adata.uns['covariates_dict'] = cov_dict
        if categorical_covariate_keys is not None:
            if 'covariates' in adata.obsm:
                del adata.obsm['covariates']
            adata.obsm['covariates'] = covariates
        if categorical_covariate_embed_keys is not None:
            if 'covariates_embed' in adata.obsm:
                del adata.obsm['covariates_embed']
            adata.obsm['covariates_embed'] = covariates_embed
        
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        
        latent = []
        for tensors in scdl:
            tensors = {k: v.to(device) for k, v in tensors.items()}  # Move tensors to the device
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            z = outputs["z"]
            latent += [z.cpu()]
            
        return torch.cat(latent).numpy()
    
    @auto_move_data
    def translate(
        self,
        adata: AnnData,
        translate_dict: Dict,
        copy: bool = False,
        
    ) -> AnnData:
        """
        Translate the given adata based on the provided translation dictionary.

        The function goes through an inference process to obtain latent representations 
        and then uses a generative process with latent varibale switching to predict cells. 
        The results are formatted and returned as an AnnData object.

        Parameters:
        - adata (AnnData): The input AnnData object.
        - translate_dict (Dict): Dictionary specifying which column in the adata should be translated to which variable.
        - copy (bool, optional): If True, a copy of the input adata will be used for processing. Defaults to False.

        Returns:
        - AnnData: An AnnData object containing the predicted cells.
        
        Examples:
        
        >> predicted = model.translate(adata_train, translate_dict= {"dataset": "chem"})
        
        """
        # find the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Make sure var names are unique
        if adata.shape[1] != len(set(adata.var_names)):
            raise ValueError('Adata var_names are not unique')

        if copy:
            adata = adata.copy()
            
        ### Inference -----------
                
        latent = torch.Tensor(self.get_latent_representation(adata))
            
        ### Generative ----------
        
        categorical_covariate_keys = self.adata.uns["covariates_dict"]["categorical"]
        categorical_covariate_embed_keys = self.adata.uns["covariates_dict"]["categorical_embed"]
        
        # latent variable switching
        for column in translate_dict.keys():
            if column not in list(adata.obs.columns):
                raise KeyError("Dict key from translate_dict not found in adata.obs.")
            adata.obs[column] = translate_dict[column]
        
        covariates, covariates_embed, orders_dict, cov_dict = prepare_metadata(
            meta_data=adata.obs,
            cov_cat_keys=categorical_covariate_keys,
            cov_cat_embed_keys=categorical_covariate_embed_keys,
            orders=self.adata.uns["covariate_orders"]
        )
            
        tensors = {
            "covariates": torch.Tensor(covariates.values).to(device), 
            "covariates_embed": torch.Tensor(covariates_embed.values).to(device)
        }

        cov = self.module._get_cov(tensors=tensors)
        predicted_cells = self.module.generative(z = latent, cov = cov)["px"].cpu().detach().numpy()
        
        # make output pretty
        
        adata.uns['covariate_orders'] = orders_dict
        adata.uns['covariates_dict'] = cov_dict
        if categorical_covariate_keys is not None:
            if 'covariates' in adata.obsm:
                del adata.obsm['covariates']
            adata.obsm['covariates'] = covariates
        if categorical_covariate_embed_keys is not None:
            if 'covariates_embed' in adata.obsm:
                del adata.obsm['covariates_embed']
            adata.obsm['covariates_embed'] = covariates_embed
        
        predicted_adata = AnnData(
            X=predicted_cells,
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            uns=adata.uns.copy(),
            obsm=adata.obsm.copy(),
        )
        
        return predicted_adata

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        categorical_covariate_keys: Optional[List[str]] = None,
        categorical_covariate_embed_keys: Optional[List[str]] = None,
        covariate_orders: Optional[Dict] = None,
        copy: bool = True,
        layer: Optional[str] = None,
        validation_ind_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Sets up the AnnData object for subsequent analysis.

        Parameters:
        ----------
        cls : class
            The class to which this classmethod belongs.
        adata : AnnData
            The annotated data matrix.
        categorical_covariate_keys : Optional[List[str]], default=None
            List of keys for categorical covariates.
        categorical_covariate_embed_keys : Optional[List[str]], default=None
            List of keys for categorical covariates to be embedded.
        covariate_orders : Optional[Dict], default=None
            Dictionary specifying the order of covariates.
        copy : bool, default=True
            Whether to return a copy of the original adata object.
        layer : Optional[str], default=None
            Specifies which layer of the adata object to consider.
        **kwargs : Additional keyword arguments.

        Returns:
        -------
        AnnData
            The modified or copied AnnData object.

        Raises:
        ------
        ValueError
            If var_names in adata are not unique.
        """

        # Make sure var names are unique
        if adata.shape[1] != len(set(adata.var_names)):
            raise ValueError('Adata var_names are not unique')

        if copy:
            adata = adata.copy()
        
        if covariate_orders is None:
            covariate_orders = {}

        covariates, covariates_embed, orders_dict, cov_dict = prepare_metadata(
            meta_data=adata.obs,
            cov_cat_keys=categorical_covariate_keys,
            cov_cat_embed_keys=categorical_covariate_embed_keys,
            orders=covariate_orders
        )

        adata.uns['covariate_orders'] = orders_dict
        adata.uns['covariates_dict'] = cov_dict
        if categorical_covariate_keys is not None:
            if 'covariates' in adata.obsm:
                del adata.obsm['covariates']
            adata.obsm['covariates'] = covariates
        if categorical_covariate_embed_keys is not None:
            if 'covariates_embed' in adata.obsm:
                del adata.obsm['covariates_embed']
            adata.obsm['covariates_embed'] = covariates_embed

        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=False)
        ]

        if categorical_covariate_keys is not None:
            anndata_fields.append(ObsmField('covariates', 'covariates'))
        if categorical_covariate_embed_keys is not None:
            anndata_fields.append(ObsmField('covariates_embed', 'covariates_embed'))

        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

        return adata
