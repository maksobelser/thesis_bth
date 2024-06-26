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
import pandas as pd

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
def run(dataset_to_predict: str,
        dataset_ground_truth1: str,
        dataset_ground_truth2: str,
        addl_dataset_train: str,
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
        shared_top: int,
        embed_ct: bool,
       ):
    
    genes_intersection = pd.read_table("/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/make_big_dataset/genes_shared_across_datasets.txt")
    genes_intersection = genes_intersection.genes_shared_across_datasets.tolist()
    categorical_covariate_embed_keys = ["dataset"]
    # if embed variables
    if embed_ct:
        categorical_covariate_embed_keys.append("cell_type")
    
    logging.info(f"categorical_covariate_embed_keys values are: {categorical_covariate_embed_keys}")

    if addl_dataset_train == "combination":
        datasets = ['train_adata_baseline.h5ad','extra_mouse_Embryonic.h5ad','extra_human_chem.h5ad','embedding_top100_mouse.h5ad','random_mouse_seed_42.h5ad',
                  'embedding_top70_human.h5ad', 'extra_human_preT2D.h5ad','random_human_seed_42.h5ad','extra_human_neonatal.h5ad','extra_mouse_chem.h5ad',
                  'extra_mouse_T1D.h5ad','extra_mouse_young.h5ad']

        genes_intersection = pd.read_table("/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/make_big_dataset/genes_shared_across_datasets.txt")
        genes_intersection = genes_intersection.genes_shared_across_datasets.tolist()

        adatas = []
        for extra in datasets:
            adata = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/train_data/full_datasets/{extra}")
            adata = adata[:,adata.var.index.isin(genes_intersection)]
            adatas.append(adata)

        adata_train = sc.concat(adatas, join = "inner", axis = 0)
        adata_train = transVAE.setup_anndata(adata_train, categorical_covariate_embed_keys=categorical_covariate_embed_keys, categorical_covariate_keys=["organism"], copy = True)
        
        logging.info(f'Received the following configuration for datasets train: {addl_dataset_train}, shape: {adata_train.shape}')
    
    elif addl_dataset_train == "train_adata_baseline_3000hvg.h5ad":
        # just the baseline data
        adata_train = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/train_data/full_datasets/train_adata_baseline.h5ad")
        logging.info(f'Received the following configuration for datasets train: {addl_dataset_train}, shape: {adata_train.shape}')
        adata_train = adata_train[:,adata_train.var.index.isin(genes_intersection)]
        adata_train = transVAE.setup_anndata(adata_train, categorical_covariate_embed_keys=categorical_covariate_embed_keys, categorical_covariate_keys=["organism"], copy = True)
            
    else:
        adata_baseline = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/train_data/full_datasets/train_adata_baseline.h5ad")
        adata_extra = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/train_data/full_datasets/{addl_dataset_train}")
        adata_train = sc.concat([adata_baseline, adata_extra], join = "inner", axis = 0)
        logging.info(f'Received the following configuration for datasets train: {addl_dataset_train}, shape: {adata_train.shape}, baseline_shape: {adata_baseline.shape}')
        del adata_baseline
        del adata_extra
        gc.collect()
        ## making random splits
        adata_train = adata_train[:,adata_train.var.index.isin(genes_intersection)]
        adata_train = transVAE.setup_anndata(adata_train, categorical_covariate_embed_keys=categorical_covariate_embed_keys, categorical_covariate_keys=["organism"], copy = True)

    # reading val data and data to translate
    adata_to_predict = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/data_to_predict/{dataset_to_predict}")
    dbdb_ground_truth = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/ground_truth/full_datasets/{dataset_ground_truth1}")
    mSTZ_ground_truth = sc.read_h5ad(f"/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/ground_truth/full_datasets/{dataset_ground_truth2}")

    adata_to_predict = adata_to_predict[:,adata_to_predict.var.index.isin(genes_intersection)]
    adata_to_predict = adata_to_predict[:,adata_to_predict.var.sort_index().index]
    dbdb_ground_truth = dbdb_ground_truth[:,dbdb_ground_truth.var.index.isin(genes_intersection)]
    mSTZ_ground_truth = mSTZ_ground_truth[:,mSTZ_ground_truth.var.index.isin(genes_intersection)]

    scvi.settings.seed = seed

    logging.info(f'n_hidden: {n_hidden}, n_latent: {n_latent}, learning_rate: {learning_rate}, cov_embed_dims: {cov_embed_dims}, weight_decay {weight_decay}')
    logging.info(f'max_epochs: {max_epochs}, n_layers: {n_layers}, dropout_rate: {dropout_rate}, kl_weight: {kl_weight}')

    check_adatas_var_index(adata_train, adata_to_predict, dbdb_ground_truth, mSTZ_ground_truth)

    # check this data (full_datasets)

    adata_to_predict_h = adata_to_predict[adata_to_predict.obs.disease == "healthy"]
    adata_to_predict_OOD = adata_to_predict[adata_to_predict.obs.disease == "T2D"]
    del adata_to_predict
    gc.collect()

    dbdb_ground_truth_h = dbdb_ground_truth[dbdb_ground_truth.obs.disease == "healthy"]
    dbdb_ground_truth_OOD = dbdb_ground_truth[dbdb_ground_truth.obs.disease == "T2D"]
    cts_to_consider = dbdb_ground_truth.obs.cell_type.unique()
    del dbdb_ground_truth
    gc.collect()

    mSTZ_ground_truth_h = mSTZ_ground_truth[mSTZ_ground_truth.obs.disease == "healthy"]
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
    model.train(batch_size=4096, max_epochs = max_epochs, train_size = 0.99, enable_progress_bar = False,
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

    results_de_dbdb = compare_de(X=dbdb_ground_truth_OOD.X.toarray(), Y=preds_dbdb_OOD.X, C=dbdb_ground_truth_h.X.toarray(), shared_top = shared_top, method = "t-test")
    results_de_mSTZ = compare_de(X=mSTZ_ground_truth_OOD.X.toarray(), Y=preds_mSTZ_OOD.X, C=mSTZ_ground_truth_h.X.toarray(), shared_top = shared_top, method = "t-test")

    for name, value in results_de_dbdb.items():
        results[name + "_dbdb"] = value
    for name, value in results_de_mSTZ.items():
        results[name + "_mSTZ"] = value

    for ct in cts_to_consider:
        # healthy splits
        preds_dbdb_h_tmp = preds_dbdb_h[preds_dbdb_h.obs.cell_type == ct]
        dbdb_ground_truth_h_tmp = dbdb_ground_truth_h[dbdb_ground_truth_h.obs.cell_type == ct]
        preds_mSTZ_h_tmp = preds_mSTZ_h[preds_mSTZ_h.obs.cell_type == ct]
        mSTZ_ground_truth_h_tmp = mSTZ_ground_truth_h[mSTZ_ground_truth_h.obs.cell_type == ct]

        # OOD splits
        preds_dbdb_OOD_tmp = preds_dbdb_OOD[preds_dbdb_OOD.obs.cell_type == ct]
        dbdb_ground_truth_OOD_tmp = dbdb_ground_truth_OOD[dbdb_ground_truth_OOD.obs.cell_type == ct]
        preds_mSTZ_OOD_tmp = preds_mSTZ_OOD[preds_mSTZ_OOD.obs.cell_type == ct]
        mSTZ_ground_truth_OOD_tmp = mSTZ_ground_truth_OOD[mSTZ_ground_truth_OOD.obs.cell_type == ct]

        ct = ct.replace(" ", "_")
        # dbdb
        results[ct + "_" + "r2_dbdb_healthy"] = compute_r2_score(preds_dbdb_h_tmp, dbdb_ground_truth_h_tmp)
        results[ct + "_" + "r2_dbdb_OOD"] = compute_r2_score(preds_mSTZ_OOD_tmp, mSTZ_ground_truth_OOD_tmp)
        # mSTZ
        results[ct + "_" + "r2_mSTZ_healthy"] = compute_r2_score(preds_mSTZ_h_tmp, mSTZ_ground_truth_h_tmp)
        results[ct + "_" + "r2_mSTZ_OOD"] = compute_r2_score(preds_mSTZ_OOD_tmp, mSTZ_ground_truth_OOD_tmp)
        
        
        results_de_dbdb = compare_de(X=dbdb_ground_truth_OOD_tmp.X.toarray(), Y=preds_dbdb_OOD_tmp.X, C=dbdb_ground_truth_h_tmp.X.toarray(), shared_top = shared_top, method = "t-test")
        results_de_mSTZ = compare_de(X=mSTZ_ground_truth_OOD_tmp.X.toarray(), Y=preds_mSTZ_OOD_tmp.X, C=mSTZ_ground_truth_h_tmp.X.toarray(), shared_top = shared_top, method = "t-test")

        for name, value in results_de_dbdb.items():
            results[ct + "_" + name + "_dbdb" ] = value
        for name, value in results_de_mSTZ.items():
            results[ct + "_" + name + "_mSTZ"] = value
            
    results["n_cells_train"] = adata_train.n_obs
    
    # free memory
    del dbdb_ground_truth_OOD_tmp, preds_dbdb_OOD_tmp, dbdb_ground_truth_h_tmp
    del mSTZ_ground_truth_OOD_tmp, preds_mSTZ_OOD_tmp, mSTZ_ground_truth_h_tmp
    del adata_train, model
    gc.collect()
    
    try:
        # GSEA performance -> should be done on just on B cell from pancereas
        # load the biomart m2h translation
        m2h = pd.read_csv("./final_experiments/mouse2human_translations_for_GSEA_on-the-fly.csv")

        ct = "type B pancreatic cell"
        # GT
        dbdb_ground_truth_h_Bcells = dbdb_ground_truth_h[dbdb_ground_truth_h.obs.cell_type == ct]
        mSTZ_ground_truth_h_Bcells = mSTZ_ground_truth_h[mSTZ_ground_truth_h.obs.cell_type == ct]
        # preds
        preds_dbdb_OOD_Bcells = preds_dbdb_OOD[preds_dbdb_OOD.obs.cell_type == ct]
        preds_mSTZ_OOD_Bcells = preds_mSTZ_OOD[preds_mSTZ_OOD.obs.cell_type == ct]

        # dbdb GT -> clean for GSEA
        new_var = dbdb_ground_truth_h_Bcells.var.merge(m2h, left_index=True, right_on="ensembl_gene_id", how = "left")
        new_var.index = new_var["hsapiens_homolog_associated_gene_name"]
        dbdb_ground_truth_h_Bcells.var = new_var

        # mSTZ GT -> clean for GSEA
        new_var = mSTZ_ground_truth_h_Bcells.var.merge(m2h, left_index=True, right_on="ensembl_gene_id", how = "left")
        new_var.index = new_var["hsapiens_homolog_associated_gene_name"]
        mSTZ_ground_truth_h_Bcells.var = new_var

        # dbdb preds -> clean for GSEA
        new_var = preds_dbdb_OOD_Bcells.var.merge(m2h, left_index=True, right_on="ensembl_gene_id", how = "left")
        new_var.index = new_var["hsapiens_homolog_associated_gene_name"]
        preds_dbdb_OOD_Bcells.var = new_var

        # mSTZ preds -> clean for GSEA
        new_var = preds_mSTZ_OOD_Bcells.var.merge(m2h, left_index=True, right_on="ensembl_gene_id", how = "left")
        new_var.index = new_var["hsapiens_homolog_associated_gene_name"]
        preds_mSTZ_OOD_Bcells.var = new_var

        # GSEA dbdb
        dbdb = sc.concat([dbdb_ground_truth_h_Bcells, preds_dbdb_OOD_Bcells])
        mSTZ = sc.concat([mSTZ_ground_truth_h_Bcells, preds_mSTZ_OOD_Bcells])

        res_KEGG =  gp.gsea(data=dbdb.to_df().T, # row -> genes, column-> samples
                            gene_sets="KEGG_2019_Mouse",
                            cls=dbdb.obs.disease,
                            permutation_num=1000,
                            permutation_type='phenotype',
                            outdir=None,
                            method='s2n', # signal_to_noise
                            threads=5,
                            verbose=True)

        GSEA_results_dbdb = res_KEGG.res2d
        del res_KEGG
        gc.collect()

        res_KEGG =  gp.gsea(data=mSTZ.to_df().T, # row -> genes, column-> samples
                            gene_sets="KEGG_2019_Mouse",
                            cls=mSTZ.obs.disease,
                            permutation_num=1000,
                            permutation_type='phenotype',
                            outdir=None,
                            method='s2n', # signal_to_noise
                            threads=5,
                            verbose=True)

        GSEA_results_mSTZ = res_KEGG.res2d
        del res_KEGG
        gc.collect()

        # get GT
        gt_pathways = pd.read_csv("gt_top_100_pathways.csv")
        gt_top100pathways_dbdb = set(gt_pathways.dbdb_top100_pathways.tolist())
        gt_top100pathways_mSTZ = set(gt_pathways.mSTZ_top100_pathways.tolist())

        # compare
        GSEA_results_dbdb_pathways = set(GSEA_results_dbdb.Term[1:100].tolist())
        GSEA_results_mSTZ_pathways = set(GSEA_results_mSTZ.Term[1:100].tolist())

        results["GSEA_top_100_matching_dbdb"] = len(gt_top100pathways_dbdb.intersection(GSEA_results_dbdb_pathways))
        results["GSEA_top_100_matching_mSTZ"] = len(gt_top100pathways_mSTZ.intersection(GSEA_results_mSTZ_pathways))
        
        # save GSEA results
        GSEA_results_dbdb.to_csv(os.path.join(save_dir,"GSEA_results_dbdb.csv"))
        GSEA_results_mSTZ.to_csv(os.path.join(save_dir,"GSEA_results_mSTZ.csv"))
    
    except:
        logging.info('An error occurred while performing GSEA.')
        results["GSEA_top_100_matching_dbdb"] = None
        results["GSEA_top_100_matching_mSTZ"] = None

    if save_models:
        try:
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
        except:
            logging.info('An error occurred while saving model predictions.')

    return results