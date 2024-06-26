from __future__ import annotations
import chex
from dataclasses import field
from typing import Dict, Iterable
from scvi._types import LossRecord, Tensor
import pandas as pd
from typing import Optional, List, Union

def prepare_metadata(meta_data: pd.DataFrame,
                     cov_cat_keys: Optional[list] = None,
                     cov_cat_embed_keys: Optional[list] = None,
                     cov_cont_keys: Optional[list] = None,
                     orders=None):
    """

    :param meta_data: Dataframe containing species and covariate info, e.g. from non-registered adata.obs
    :param cov_cat_keys: List of categorical covariates column names.
    :param cov_cat_embed_keys: List of categorical covariates column names to be encoded via embedding
    rather than one-hot encoding.
    :param cov_cont_keys: List of continuous covariates column names.
    :param orders: Defined orders for species or categorical covariates. Dict with keys being
    'species' or categorical covariates names and values being lists of categories. May contain more/less
    categories than data.
    :return: covariate data, dict with order of categories per covariate, dict with keys categorical and continuous
    specifying lists of covariates
    """
    if cov_cat_keys is None:
        cov_cat_keys = []
    if cov_cat_embed_keys is None:
        cov_cat_embed_keys = []
    if cov_cont_keys is None:
        cov_cont_keys = []

    def order_categories(values: pd.Series, categories: Union[List, None] = None):
        if categories is None:
            categories = pd.Categorical(values).categories.values
        else:
            missing = set(values.unique()) - set(categories)
            if len(missing) > 0:
                raise ValueError(f'Some values of {values.name} are not in the specified categories order: {missing}')
        return list(categories)

    def dummies_categories(values: pd.Series, categories: Union[List, None] = None):
        """
        Make dummies of categorical covariates. Use specified order of categories.
        :param values: Categories for each observation.
        :param categories: Order of categories to use.
        :return: dummies, categories. Dummies - one-hot encoding of categories in same order as categories.
        """
        categories = order_categories(values=values, categories=categories)

        # Get dummies
        # Ensure ordering
        values = pd.Series(pd.Categorical(values=values, categories=categories, ordered=True),
                           index=values.index, name=values.name)
        # This is problematic if many covariates
        dummies = pd.get_dummies(values, prefix=values.name)

        return dummies, categories

    # Covariate encoding
    # Save order of covariates and categories
    cov_dict = {'categorical': cov_cat_keys, 'categorical_embed': cov_cat_embed_keys, 'continuous': cov_cont_keys}
    # One-hot encoding of categorical covariates
    orders_dict = {}

    if len(cov_cat_keys) > 0 or len(cov_cont_keys) > 0:
        cov_cat_data = []
        for cov_cat_key in cov_cat_keys:
            cat_dummies, cat_order = dummies_categories(
                values=meta_data[cov_cat_key], categories=orders.get(cov_cat_key, None))
            cov_cat_data.append(cat_dummies)
            orders_dict[cov_cat_key] = cat_order
        # Prepare single cov array for all covariates
        cov_data_parsed = pd.concat(cov_cat_data + [meta_data[cov_cont_keys]], axis=1)
    else:
        cov_data_parsed = None

    if len(cov_cat_embed_keys) > 0:
        cov_embed_data = []
        for cov_cat_embed_key in cov_cat_embed_keys:
            cat_order = order_categories(values=meta_data[cov_cat_embed_key],
                                         categories=orders.get(cov_cat_embed_key, None))
            cat_map = dict(zip(cat_order, range(len(cat_order))))
            cov_embed_data.append(meta_data[cov_cat_embed_key].map(cat_map))
            orders_dict[cov_cat_embed_key] = cat_order
        cov_embed_data = pd.concat(cov_embed_data, axis=1)
    else:
        cov_embed_data = None

    return cov_data_parsed, cov_embed_data, orders_dict, cov_dict

from itertools import combinations

def check_adatas_var_index(*adatas):
    """Check if the variable indices of all provided AnnData objects are the same and in the same order."""
    for i, j in combinations(range(len(adatas)), 2):
        if not all(adatas[i].var.index == adatas[j].var.index):
            raise ValueError(f"The variable indices of the AnnData objects at positions {i} and {j} do not match!")
    print("Everything ok!")
    

from sklearn.metrics import r2_score

def compute_r2_score(preds, ground_truth):
    # Convert to densdataset adata.X is sparse
    ground_truth = ground_truth.X.toarray()
    ground_truth = ground_truth.mean(axis = 0)
    preds = preds.X.mean(axis = 0)
    # Compute R2 score
    r2 = r2_score(ground_truth, preds)
    return r2

import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc

def compare_de(X: np.ndarray, Y: np.ndarray, C: np.ndarray, shared_top: int = 100, **kwargs) -> dict:
    """Compare DEG across real and simulated perturbations.

    Computes DEG for real and simulated perturbations vs. control and calculates
    metrics to evaluate similarity of the results.

    Args:
        X: Real perturbed data.
        Y: Simulated perturbed data.
        C: Control data
        shared_top: The number of top DEG to compute the proportion of their intersection.
        **kwargs: arguments for `scanpy.tl.rank_genes_groups`.
    """
    
    n_vars = X.shape[1]
    assert n_vars == Y.shape[1] == C.shape[1]

    shared_top = min(shared_top, n_vars)
    vars_ranks = np.arange(1, n_vars + 1)

    adatas_xy = {}
    adatas_xy["x"] = ad.AnnData(X, obs={"label": "comp"})
    adatas_xy["y"] = ad.AnnData(Y, obs={"label": "comp"})
    adata_c = ad.AnnData(C, obs={"label": "ctrl"})

    results = pd.DataFrame(index=adata_c.var_names)
    top_names = []
    for group in ("x", "y"):
        adata_joint = ad.concat((adatas_xy[group], adata_c), index_unique="-")

        sc.tl.rank_genes_groups(adata_joint, groupby="label", reference="ctrl", key_added="de", **kwargs)

        srt_idx = np.argsort(adata_joint.uns["de"]["names"]["comp"])
        results[f"scores_{group}"] = adata_joint.uns["de"]["scores"]["comp"][srt_idx]
        results[f"pvals_adj_{group}"] = adata_joint.uns["de"]["pvals_adj"]["comp"][srt_idx]
        # needed to avoid checking rankby_abs
        results[f"ranks_{group}"] = vars_ranks[srt_idx]

        top_names.append(adata_joint.uns["de"]["names"]["comp"][:shared_top])

    metrics = {}
    metrics["shared_top_genes"] = len(set(top_names[0]).intersection(top_names[1])) / shared_top
    metrics["scores_corr"] = results["scores_x"].corr(results["scores_y"], method="pearson")
    metrics["pvals_adj_corr"] = results["pvals_adj_x"].corr(results["pvals_adj_y"], method="pearson")
    metrics["scores_ranks_corr"] = results["ranks_x"].corr(results["ranks_y"], method="spearman")

    return metrics

def compare_logfold(X: np.ndarray, Y: np.ndarray, C: np.ndarray, **kwargs) -> dict:
    """Compare DEG across real and simulated perturbations.

    Computes DEG for real and simulated perturbations vs. control and calculates
    metrics to evaluate similarity of the results.

    Args:
        X: Real perturbed data.
        Y: Simulated perturbed data.
        C: Control data
        shared_top: The number of top DEG to compute the proportion of their intersection.
        **kwargs: arguments for `scanpy.tl.rank_genes_groups`.
    """
    n_vars = X.shape[1]
    assert n_vars == Y.shape[1] == C.shape[1]

    prop_of_genes_set_to_0 = np.mean(Y < 0)
    Y[Y < 0] = 0

    adatas_xy = {}
    adatas_xy["x"] = ad.AnnData(X, obs={"label": "comp"})
    adatas_xy["y"] = ad.AnnData(Y, obs={"label": "comp"})
    adata_c = ad.AnnData(C, obs={"label": "ctrl"})

    results = pd.DataFrame(index=adata_c.var_names)
    top_names = []
    for group in ("x", "y"):
        adata_joint = ad.concat((adatas_xy[group], adata_c), index_unique="-")

        sc.tl.rank_genes_groups(adata_joint, groupby="label", reference="ctrl", key_added="de", **kwargs)
        results[f"logfold_{group}"] = [elm[0] for elm in adata_joint.uns["de"]["logfoldchanges"].tolist()]

    metics = {}
    metics["logfold_corr"] = results["logfold_x"].corr(results["logfold_y"], method="pearson")
    metics["prop_of_genes_set_to_0"] = prop_of_genes_set_to_0

    return metics