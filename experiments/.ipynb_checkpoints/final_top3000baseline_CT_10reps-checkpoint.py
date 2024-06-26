import logging
from sacred import Experiment
import seml
import os
from transVAE.model.transVAE import transVAE
from transVAE.module._utils import check_adatas_var_index, compare_de, compare_logfold, compute_r2_score
import scanpy as sc
import scvi
import numpy as np
import gc

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
        adata_train = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/train_data/top3000_just_baseline/train_adata_baseline_3000hvg.h5ad")
        logging.info(f'Received the following configuration for datasets train: {addl_dataset_train}, shape: {adata_train.shape}')
    else:
        adata_baseline = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/train_data/top3000_just_baseline/train_adata_baseline_3000hvg.h5ad")
        adata_extra = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/train_data/top3000_just_baseline/{addl_dataset_train}")
        adata_train = sc.concat([adata_baseline, adata_extra], join = "inner", axis = 0)
        logging.info(f'Received the following configuration for datasets train: {addl_dataset_train}, shape: {adata_train.shape}, baseline_shape: {adata_baseline.shape}')
        del adata_baseline
        del adata_extra
        gc.collect()

    adata_train = transVAE.setup_anndata(adata_train, categorical_covariate_embed_keys=["dataset"], categorical_covariate_keys=["organism"], copy = True)
    adata_to_predict = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/data_to_predict/{dataset_to_predict}")
    dbdb_ground_truth = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/ground_truth/{dataset_ground_truth1}")
    mSTZ_ground_truth = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/ground_truth/{dataset_ground_truth2}")

    scvi.settings.seed = seed

    logging.info(f'n_hidden: {n_hidden}, n_latent: {n_latent}, learning_rate: {learning_rate}, cov_embed_dims: {cov_embed_dims}, weight_decay {weight_decay}')
    logging.info(f'max_epochs: {max_epochs}, n_layers: {n_layers}, dropout_rate: {dropout_rate}, kl_weight: {kl_weight}')

    check_adatas_var_index(adata_train, adata_to_predict, dbdb_ground_truth, mSTZ_ground_truth)

    adata_to_predict_h = adata_to_predict[adata_to_predict.obs.disease == "normal"]
    adata_to_predict_OOD = adata_to_predict[adata_to_predict.obs.disease == "T2D"]
    del adata_to_predict
    gc.collect()

    dbdb_ground_truth_h = dbdb_ground_truth[dbdb_ground_truth.obs.disease == "normal"]
    dbdb_ground_truth_OOD = dbdb_ground_truth[dbdb_ground_truth.obs.disease == "T2D"]
    del dbdb_ground_truth
    gc.collect()

    mSTZ_ground_truth_h = mSTZ_ground_truth[mSTZ_ground_truth.obs.disease == "normal"]
    mSTZ_ground_truth_OOD = mSTZ_ground_truth[mSTZ_ground_truth.obs.disease == "T2D"]
    del mSTZ_ground_truth
    gc.collect()

    validation_adatas_dict = {"adata_to_predict":adata_to_predict_OOD,
                              "ground_truths": {"dbdb": dbdb_ground_truth_OOD,
                                                "mSTZ": mSTZ_ground_truth_OOD},
                              "translate_dicts": {"dbdb":{"dataset":"db/db", "organism": "Mus musculus"},
                                                  "mSTZ":{"dataset":"mSTZ", "organism": "Mus musculus"}}}

    # make the model
    model = transVAE(adata_train, 
                     n_hidden=n_hidden, 
                     n_latent=n_latent, 
                     n_layers=n_layers, 
                     dropout_rate=dropout_rate, 
                     cov_embed_dims = cov_embed_dims, 
                     kl_weight = kl_weight,
                     validation_adatas_dict = validation_adatas_dict)


    # train the model
    model.train(batch_size=4096, max_epochs = max_epochs, train_size = 0.99, enable_progress_bar = True,
                early_stopping = True, early_stopping_monitor = 'mean_r2_validation_eval', early_stopping_mode = "max", 
                early_stopping_min_delta = 0.01, early_stopping_patience = 70,
                plan_kwargs = {"lr":learning_rate,
                               "weight_decay":weight_decay,
                               "reduce_lr_on_plateau":True,
                               "lr_factor":0.5,
                               "lr_patience":50,
                               "lr_scheduler_metric":"reconstruction_loss_validation"})

    results = model.history

    # dbdb
    preds_dbdb_h = model.translate(adata_to_predict_h, translate_dict={"dataset":"db/db", "organism": "Mus musculus"})
    preds_dbdb_OOD = model.translate(adata_to_predict_OOD, translate_dict={"dataset":"db/db", "organism": "Mus musculus"})

    # mSTZ
    preds_mSTZ_h = model.translate(adata_to_predict_h, translate_dict={"dataset":"mSTZ", "organism": "Mus musculus"})
    preds_mSTZ_OOD = model.translate(adata_to_predict_OOD, translate_dict={"dataset":"mSTZ", "organism": "Mus musculus"})

    # dbdb
    results["r2_dbdb_healthy"] = compute_r2_score(preds_dbdb_h, dbdb_ground_truth_h)
    results["r2_dbdb_OOD"] = model.history["dbdb_r2_validation_eval"].iloc[-1].item()
    # mSTZ
    results["r2_mSTZ_healthy"] = compute_r2_score(preds_mSTZ_h, mSTZ_ground_truth_h)
    results["r2_mSTZ_OOD"] = model.history["mSTZ_r2_validation_eval"].iloc[-1].item()

    results_de_dbdb = compare_de(X=dbdb_ground_truth_OOD.X.toarray(), Y=preds_dbdb_OOD.X, C=dbdb_ground_truth_h.X.toarray(), shared_top = 1000, method = "t-test")
    results_de_mSTZ = compare_de(X=mSTZ_ground_truth_OOD.X.toarray(), Y=preds_mSTZ_OOD.X, C=mSTZ_ground_truth_h.X.toarray(), shared_top = 1000, method = "t-test")

    for name, value in results_de_dbdb.items():
        results[name + "_dbdb"] = value
    for name, value in results_de_mSTZ.items():
        results[name + "_mSTZ"] = value

    for ct in dbdb_ground_truth_OOD.obs.cell_type.unique():

        # healthy splits
        preds_dbdb_h_tmp = preds_dbdb_h[preds_dbdb_h.obs.cell_type == ct]
        dbdb_ground_truth_h_tmp = dbdb_ground_truth_h[dbdb_ground_truth_h.obs.cell_type == ct]
        preds_mSTZ_h_tmp = preds_mSTZ_h[preds_mSTZ_h.obs.cell_type == ct]
        mSTZ_ground_truth_h_tmp = mSTZ_ground_truth_h[mSTZ_ground_truth_h.obs.cell_type == ct]

        # OOD splits
        preds_dbdb_OOD_tmp = preds_dbdb_h[preds_dbdb_h.obs.cell_type == ct]
        dbdb_ground_truth_OOD_tmp = dbdb_ground_truth_h[dbdb_ground_truth_h.obs.cell_type == ct]
        preds_mSTZ_OOD_tmp = preds_mSTZ_h[preds_mSTZ_h.obs.cell_type == ct]
        mSTZ_ground_truth_OOD_tmp = mSTZ_ground_truth_h[mSTZ_ground_truth_h.obs.cell_type == ct]

        ct = ct.replace(" ", "_")

        results[ct + "_" + "r2_dbdb_healthy"] = compute_r2_score(preds_dbdb_h_tmp, dbdb_ground_truth_h_tmp)
        results[ct + "_" + "r2_dbdb_OOD"] = compute_r2_score(preds_mSTZ_OOD_tmp, mSTZ_ground_truth_OOD_tmp)
        # mSTZ
        results[ct + "_" + "r2_mSTZ_healthy"] = compute_r2_score(preds_mSTZ_h_tmp, mSTZ_ground_truth_h_tmp)
        results[ct + "_" + "r2_mSTZ_OOD"] = compute_r2_score(preds_mSTZ_OOD_tmp, mSTZ_ground_truth_OOD_tmp)

        results_de_dbdb = compare_de(X=dbdb_ground_truth_OOD_tmp.X.toarray(), Y=preds_dbdb_OOD_tmp.X, C=dbdb_ground_truth_h_tmp.X.toarray(), shared_top = 1000, method = "t-test")
        results_de_mSTZ = compare_de(X=mSTZ_ground_truth_OOD_tmp.X.toarray(), Y=preds_mSTZ_OOD_tmp.X, C=mSTZ_ground_truth_h_tmp.X.toarray(), shared_top = 1000, method = "t-test")

        for name, value in results_de_dbdb.items():
            results[ct + "_" + name + "_dbdb" ] = value
        for name, value in results_de_mSTZ.items():
            results[ct + "_" + name + "_mSTZ_"] = value

    if save_models:
        save_dir = f"/d/hpc/projects/FRI/DL/mo6643/MSC/cross_species_prediction_save/{save_folder_name}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dir = os.path.join(save_dir, f"transVAE_train_{addl_dataset_train}_hid{n_hidden}_lat{n_latent}_lr{learning_rate}_cov{cov_embed_dims}_ep{max_epochs}_ly{n_layers}_dr{dropout_rate}_kl{kl_weight}_wd{weight_decay}_s{seed}")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        #np.save(arr=preds_dbdb_h.X, file=os.path.join(save_dir,"preds_dbdb_h.npy"))
        np.save(arr=preds_dbdb_OOD.X, file=os.path.join(save_dir,"preds_dbdb_OOD.npy"))
        #np.save(arr=preds_mSTZ_h.X, file=os.path.join(save_dir,"preds_mSTZ_h.npy"))
        np.save(arr=preds_mSTZ_OOD.X, file=os.path.join(save_dir,"preds_mSTZ_OOD.npy"))
         
    return results
