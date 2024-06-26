import logging
from sacred import Experiment
import seml
import os
from transVAE.model.transVAE import transVAE
from transVAE.module._utils import check_adatas_var_index, compute_r2_score, compare_de, compare_logfold
import scanpy as sc
import scvi
import numpy as np

ex = Experiment()
seml.setup_logger(ex)

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite)) 

@ex.automain
def run(addl_dataset_train: str,
        dataset_to_predict: str,
        dataset_ground_truth1: str,
        dataset_ground_truth2: str,
        seed: int,
        n_hidden: int, 
        n_latent: int, 
        learning_rate: float, 
        cov_embed_dims: int,
        max_epochs: int, 
        n_layers: int, 
        dropout_rate: float,
        kl_weight: float,
        save_models: bool,
        save_folder_name: str,
        weight_decay: float,
       ):    

    if addl_dataset_train == "train_adata_baseline_3000hvg.h5ad":
        adata_train = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/train_data/train_adata_baseline_3000hvg.h5ad")
        logging.info(f'Received the following configuration for datasets train: {addl_dataset_train}, shape: {adata_train.shape}')
    else:
        adata_baseline = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/train_data/train_adata_baseline_3000hvg.h5ad")
        adata_extra = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/train_data/{addl_dataset_train}")
        adata_train = sc.concat([adata_baseline, adata_extra], join = "inner", axis = 0)
        logging.info(f'Received the following configuration for datasets train: {addl_dataset_train}, shape: {adata_train.shape}, baseline_shape: {adata_baseline.shape}')

    adata_train = transVAE.setup_anndata(adata_train, categorical_covariate_embed_keys=["dataset"], categorical_covariate_keys=["organism"], copy = True)
    adata_to_predict = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/data_to_predict/{dataset_to_predict}")
    dbdb_ground_truth = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/ground_truth/{dataset_ground_truth1}")
    mSTZ_ground_truth = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/ground_truth/{dataset_ground_truth2}")
    
    scvi.settings.seed = seed
    
    logging.info(f'n_hidden: {n_hidden}, n_latent: {n_latent}, learning_rate: {learning_rate}, cov_embed_dims: {cov_embed_dims}, weight_decay {weight_decay}')
    logging.info(f'max_epochs: {max_epochs}, n_layers: {n_layers}, dropout_rate: {dropout_rate}, kl_weight: {kl_weight}')
    
    check_adatas_var_index(adata_train, adata_to_predict, dbdb_ground_truth, mSTZ_ground_truth)
    
    # make the model
    model = transVAE(adata_train, 
                     n_hidden=n_hidden, 
                     n_latent=n_latent, 
                     n_layers=n_layers, 
                     dropout_rate=dropout_rate, 
                     cov_embed_dims = cov_embed_dims, 
                     kl_weight = kl_weight)

    # train the model
    model.train(batch_size=4096, max_epochs = max_epochs, train_size = 0.9, enable_progress_bar = False, 
                early_stopping = True, early_stopping_monitor = 'reconstruction_loss_validation', early_stopping_min_delta = 0.1, early_stopping_patience = 100,
                plan_kwargs = {"lr": learning_rate, 
                               "weight_decay":weight_decay,
                               "reduce_lr_on_plateau":True,
                               "lr_factor":0.5,
                               "lr_patience":50,
                               "lr_scheduler_metric":"reconstruction_loss_validation"})

    adata_to_predict_h = adata_to_predict[adata_to_predict.obs.disease == "healthy"]
    adata_to_predict_OOD = adata_to_predict[adata_to_predict.obs.disease == "T2D"]

    preds_dbdb_h = model.translate(adata_to_predict_h, translate_dict={"dataset":"db/db", "organism": "Mus musculus"})
    preds_dbdb_OOD = model.translate(adata_to_predict_OOD, translate_dict={"dataset":"db/db", "organism": "Mus musculus"})

    dbdb_ground_truth_h = dbdb_ground_truth[dbdb_ground_truth.obs.disease == "healthy"]
    dbdb_ground_truth_OOD = dbdb_ground_truth[dbdb_ground_truth.obs.disease == "T2D"]
    
    preds_mSTZ_h = model.translate(adata_to_predict_h, translate_dict={"dataset":"mSTZ", "organism": "Mus musculus"})
    preds_mSTZ_OOD = model.translate(adata_to_predict_OOD, translate_dict={"dataset":"mSTZ", "organism": "Mus musculus"})

    mSTZ_ground_truth_h = mSTZ_ground_truth[mSTZ_ground_truth.obs.disease == "healthy"]
    mSTZ_ground_truth_OOD = mSTZ_ground_truth[mSTZ_ground_truth.obs.disease == "T2D"]

    results = model.history

    results["r2_dbdb_healthy"] = compute_r2_score(preds_dbdb_h, dbdb_ground_truth_h)
    results["r2_dbdb_OOD"] = compute_r2_score(preds_dbdb_OOD, dbdb_ground_truth_OOD)
    results["r2_mSTZ_healthy"] = compute_r2_score(preds_mSTZ_h, mSTZ_ground_truth_h)
    results["r2_mSTZ_OOD"] = compute_r2_score(preds_mSTZ_OOD, mSTZ_ground_truth_OOD)
    
    results_de_dbdb = compare_de(X=dbdb_ground_truth_OOD.X.todense(), Y=preds_dbdb_OOD.X, C=dbdb_ground_truth_h.X.todense(), shared_top = 1000)
    results_de_mSTZ = compare_de(X=mSTZ_ground_truth_OOD.X.todense(), Y=preds_mSTZ_OOD.X, C=mSTZ_ground_truth_h.X.todense(), shared_top = 1000)
    
    for name, value in results_de_dbdb.items():
        results[name + "_dbdb"] = value
    for name, value in results_de_mSTZ.items():
        results[name + "_mSTZ"] = value
        
    results_logfold_dbdb = compare_logfold(X=dbdb_ground_truth_OOD.X.todense(), Y=preds_dbdb_OOD.X, C=dbdb_ground_truth_h.X.todense())
    results_logfold_mSTZ = compare_logfold(X=mSTZ_ground_truth_OOD.X.todense(), Y=preds_mSTZ_OOD.X, C=mSTZ_ground_truth_h.X.todense())
    
    for name, value in results_logfold_dbdb.items():
        results[name + "_dbdb"] = value
    for name, value in results_logfold_mSTZ.items():
        results[name + "_mSTZ"] = value
    
    if save_models:
        save_dir = f"/d/hpc/projects/FRI/DL/mo6643/MSC/cross_species_prediction_save/{save_folder_name}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dir = os.path.join(save_dir, f"transVAE_train_{addl_dataset_train}_hid{n_hidden}_lat{n_latent}_lr{learning_rate}_cov{cov_embed_dims}_ep{max_epochs}_ly{n_layers}_dr{dropout_rate}_kl{kl_weight}_wd{weight_decay}_s{seed}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        np.save(arr=preds_dbdb_h.X, file=os.path.join(save_dir,"preds_dbdb_h.npy"))
        np.save(arr=preds_dbdb_OOD.X, file=os.path.join(save_dir,"preds_dbdb_OOD.npy"))
        np.save(arr=preds_mSTZ_h.X, file=os.path.join(save_dir,"preds_mSTZ_h.npy"))
        np.save(arr=preds_mSTZ_OOD.X, file=os.path.join(save_dir,"preds_mSTZ_OOD.npy"))
         
    return results
