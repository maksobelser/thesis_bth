import scanpy as sc
import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

preds_dir = "/d/hpc/projects/FRI/DL/mo6643/MSC/cross_species_prediction_save/experiment_161123/"

def parse_dir_name(dir_name):
    pattern = r'transVAE_train_(.+)_hid(\d+)_lat(\d+)_lr([0-9.]+)_cov(\d+)_ep(\d+)_ly(\d+)_dr([0-9.]+)_kl([0-9.]+)_wd([0-9.]+)_s(\d+)'
    match = re.match(pattern, dir_name)

    if match:
        return {
            'addl_dataset': match.group(1) + '.h5ad',
            'hidden_layers': int(match.group(2)),
            'latent_dim': int(match.group(3)),
            'learning_rate': float(match.group(4)),
            'cov_embed_dim': int(match.group(5)),
            'max_epochs': int(match.group(6)),
            'layers': int(match.group(7)),
            'dropout_rate': float(match.group(8)),
            'kl_divergence_weight': float(match.group(9)),
            'weight_decay': float(match.group(10)),
            'seed': int(match.group(11))
        }
    else:
        return ValueError("dir does not match pattern.")
    
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

def process_and_update_config(config_dict, dataset_name, preds, ground_truth, ground_truth_h, cell_types):
    # Compute R2 score for the entire dataset and update config_dict
    config_dict[f"r2_{dataset_name}"] = compute_r2_score(preds=preds, ground_truth=ground_truth)
    
    # Compare logfold for the entire dataset and update config_dict
    overall_results = compare_de(X=ground_truth.X.todense(), Y=preds.X, C=ground_truth_h.X.todense(), shared_top = 1000, method = 't-test')
    for name, value in overall_results.items():
        config_dict[f"{name}_{dataset_name}"] = value

    # Process data for each cell type and update config_dict
    for cell_type in cell_types:
        # Filter data for the specific cell type
        ct_preds = preds[preds.obs.cell_type == cell_type]
        ct_ground_truth = ground_truth[ground_truth.obs.cell_type == cell_type]
        ct_ground_truth_h = ground_truth_h[ground_truth_h.obs.cell_type == cell_type]

        # Compute R2 score for the cell type
        ct_r2_score = compute_r2_score(preds=ct_preds, ground_truth=ct_ground_truth)

        # Compare logfold for the cell type
        ct_results = compare_de(X=ct_ground_truth.X.todense(), Y=ct_preds.X, C=ct_ground_truth_h.X.todense(), shared_top = 1000, method = 't-test')
        for name, value in ct_results.items():
            ct_key = f"{name}_{dataset_name}_{cell_type.replace(' ', '_')}"
            config_dict[ct_key] = value

    return config_dict

configs = os.listdir(preds_dir)

adata_to_predict = sc.read_h5ad("/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/data_to_predict/wang_to_predict_baseline3000hvg.h5ad")
adata_to_predict = adata_to_predict[adata_to_predict.obs.disease == "T2D"]
adata_to_predict.obs["cell_type"] = adata_to_predict.obs.cell_type.replace({"alpha": "pancreatic A cell",
                                                                            "beta": "type B pancreatic cell",
                                                                            "delta": "pancreatic D cell"})
adata_to_predict.obs.organism = "Mus musculus"

gt_dbdb = sc.read_h5ad("/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/ground_truth/full_datasets/dbdb_ground_truth_cleanCT.h5ad")
gt_dbdb_genes = sc.read_h5ad("/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/ground_truth/dbdb_ground_truth_baseline3000hvg.h5ad")
gt_dbdb_genes = gt_dbdb_genes.var.index.tolist()
gt_dbdb = gt_dbdb[:,gt_dbdb.var.index.isin(gt_dbdb_genes)]
gt_dbdb_h = gt_dbdb[gt_dbdb.obs.disease == "healthy"]
gt_dbdb = gt_dbdb[gt_dbdb.obs.disease == "T2D"]

gt_mSTZ = sc.read_h5ad("/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/ground_truth/full_datasets/mSTZ_ground_truth_cleanCT.h5ad")
gt_mSTZ_genes = sc.read_h5ad("/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/ground_truth/dbdb_ground_truth_baseline3000hvg.h5ad")
gt_mSTZ_genes = gt_mSTZ_genes.var.index.tolist()
gt_mSTZ = gt_mSTZ[:,gt_mSTZ.var.index.isin(gt_mSTZ_genes)]
gt_mSTZ_h = gt_mSTZ[gt_mSTZ.obs.disease == "healthy"]
gt_mSTZ = gt_mSTZ[gt_mSTZ.obs.disease == "T2D"]

cell_types = gt_mSTZ.obs.cell_type.unique().tolist()

df = []
total_configs = len(configs)
for index, config in enumerate(configs):
    print(f"Processing {index + 1}/{total_configs}: {config}")

    # parse the config
    config_dict = parse_dir_name(config)
    
    ## dbdb
    # get the pred dbdb OOD in adata structure
    X = np.load(os.path.join(preds_dir, config, 'preds_dbdb_OOD.npy'))
    preds_dbdb = adata_to_predict
    preds_dbdb.X = X
    preds_dbdb.obs.dataset = "db/db"
    preds_dbdb = preds_dbdb[~(preds_dbdb.obs.cell_type == "gamma")]
    config_dict = process_and_update_config(config_dict, "dbdb", preds_dbdb, gt_dbdb, gt_dbdb_h, cell_types)
    
    ## mSTZ
    X = np.load(os.path.join(preds_dir, config, 'preds_mSTZ_OOD.npy'))
    preds_mSTZ = adata_to_predict
    preds_mSTZ.X = X
    preds_mSTZ.obs.dataset = "mSTZ"
    preds_mSTZ = preds_mSTZ[~(preds_mSTZ.obs.cell_type == "gamma")]
    config_dict = process_and_update_config(config_dict, "mSTZ", preds_mSTZ, gt_mSTZ, gt_mSTZ_h, cell_types)
    
    df.append(config_dict)

pd.DataFrame(df).to_csv("/d/hpc/projects/FRI/DL/mo6643/MSC/cross_species_prediction/transVAE_seml/results/results161123_CT.csv")