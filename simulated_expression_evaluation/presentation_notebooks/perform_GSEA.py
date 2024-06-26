import scanpy as sc
import gseapy as gp
import pandas as pd
from gseapy import Biomart
import os
from tqdm import tqdm
import pickle
import os
import numpy as np
import pandas as pd

print("Preparing data")

configs_dbdb_df = pd.read_csv("configs_dbdb_df.csv")
configs_mSTZ_df = pd.read_csv("configs_mSTZ_df.csv")

# getting preds data dbdb

top_preds_dbdb = {}
for index, row in configs_dbdb_df.iterrows():
    save_folder_name = "datasets_for_GSEA"  
    # Construct the directory path for the saved models
    save_dir = f"/d/hpc/projects/FRI/DL/mo6643/MSC/cross_species_prediction_save/{save_folder_name}"
    save_folder = f"transVAE_train_{row['config.addl_dataset_train']}_hid{row['config.n_hidden']}_lat{int(row['config.n_latent'])}_lr{row['config.learning_rate']}_cov{int(row['config.cov_embed_dims'])}_ep{int(row['config.max_epochs'])}_ly{row['config.n_layers']}_dr{row['config.dropout_rate']}_kl{row['config.kl_weight']}_wd{row['config.weight_decay']}_s{int(row['config.seed'])}"
    dir_path = os.path.join(save_dir, save_folder)
    
    addl_dataset = row["config.addl_dataset_train"].split(".")[0]
    
    # Load the prediction file if it exists
    pred_file = os.path.join(dir_path, "preds_dbdb_OOD.npy")
    if os.path.exists(pred_file):
        preds = np.load(pred_file)
        top_preds_dbdb[addl_dataset] = preds
    else:
        print(f"Prediction file not found for configuration: {save_folder}")
        
missing = "transVAE_train_combination_hid1000_lat512_lr0.0001_cov10_ep1500_ly8_dr0.0001_kl0.005_wd0.3_s42"
save_folder_name = "full_genes_big_data_w_and_wo_ct_encoding_10_reps"  
# Construct the directory path for the saved models
save_dir = f"/d/hpc/projects/FRI/DL/mo6643/MSC/cross_species_prediction_save/{save_folder_name}"
dir_path = os.path.join(save_dir, missing)
# Load the prediction file if it exists
pred_file = os.path.join(dir_path, "preds_dbdb_OOD.npy")
if os.path.exists(pred_file):
    preds = np.load(pred_file)
    top_preds_dbdb["combination"] = preds
    
# getting preds data mSTZ

top_preds_mSTZ = {}
for index, row in configs_mSTZ_df.iterrows():
    save_folder_name = "datasets_for_GSEA"  
    # Construct the directory path for the saved models
    save_dir = f"/d/hpc/projects/FRI/DL/mo6643/MSC/cross_species_prediction_save/{save_folder_name}"
    save_folder = f"transVAE_train_{row['config.addl_dataset_train']}_hid{row['config.n_hidden']}_lat{int(row['config.n_latent'])}_lr{row['config.learning_rate']}_cov{int(row['config.cov_embed_dims'])}_ep{int(row['config.max_epochs'])}_ly{row['config.n_layers']}_dr{row['config.dropout_rate']}_kl{row['config.kl_weight']}_wd{row['config.weight_decay']}_s{int(row['config.seed'])}"
    dir_path = os.path.join(save_dir, save_folder)
    
    addl_dataset = row["config.addl_dataset_train"].split(".")[0]
    
    # Load the prediction file if it exists
    pred_file = os.path.join(dir_path, "preds_dbdb_OOD.npy")
    if os.path.exists(pred_file):
        preds = np.load(pred_file)
        top_preds_mSTZ[addl_dataset] = preds
    else:
        missing = save_folder
        print(f"Prediction file not found for configuration: {save_folder}")
        
save_folder_name = "full_genes_big_data_w_and_wo_ct_encoding_10_reps"  
# Construct the directory path for the saved models
save_dir = f"/d/hpc/projects/FRI/DL/mo6643/MSC/cross_species_prediction_save/{save_folder_name}"
dir_path = os.path.join(save_dir, missing)
# Load the prediction file if it exists
pred_file = os.path.join(dir_path, "preds_dbdb_OOD.npy")
if os.path.exists(pred_file):
    preds = np.load(pred_file)
    top_preds_mSTZ["combination"] = preds
    
# getting GT files

gt_dbdb = sc.read_h5ad("/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/ground_truth/full_datasets/dbdb_ground_truth_cleanCT.h5ad")
gt_mSTZ = sc.read_h5ad("/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/ground_truth/full_datasets/mSTZ_ground_truth_cleanCT.h5ad")

genes_considered = pd.read_table("/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/make_big_dataset/genes_shared_across_datasets.txt")

gt_dbdb = gt_dbdb[:,gt_dbdb.var.index.isin(genes_considered.genes_shared_across_datasets.tolist())]
gt_mSTZ = gt_mSTZ[:,gt_mSTZ.var.index.isin(genes_considered.genes_shared_across_datasets.tolist())]

# dbdb -> clean for GSEA

print("Cleaning dbdb")

gt_dbdb.obs['stim'] = pd.Categorical(gt_dbdb.obs['disease'], categories=["T2D", "healthy"], ordered=True)
indices = gt_dbdb.obs.sort_values(['cell_type', 'stim']).index
gt_dbdb = gt_dbdb[indices,:]

bm = Biomart()
m2h = bm.query(dataset='mmusculus_gene_ensembl',
               attributes=['ensembl_gene_id','external_gene_name',
                           'hsapiens_homolog_ensembl_gene',
                           'hsapiens_homolog_associated_gene_name'])

new_var = gt_dbdb.var.merge(m2h, left_index=True, right_on="ensembl_gene_id", how = "left")
new_var.index = new_var["hsapiens_homolog_associated_gene_name"]
gt_dbdb.var = new_var

# mSTZ -> clean for GSEA

gt_mSTZ.obs['stim'] = pd.Categorical(gt_mSTZ.obs['disease'], categories=["T2D", "healthy"], ordered=True)
indices = gt_mSTZ.obs.sort_values(['cell_type', 'stim']).index
gt_mSTZ = gt_mSTZ[indices,:]

new_var = gt_mSTZ.var.merge(m2h, left_index=True, right_on="ensembl_gene_id", how = "left")
new_var.index = new_var["hsapiens_homolog_associated_gene_name"]
gt_mSTZ.var = new_var

# prepare Wang

print("Preparing Wang")

wang = sc.read_h5ad("/d/hpc/projects/FRI/DL/mo6643/MSC/data/data_update_slack/data_splits/data_splits_train_merge/data_to_predict/wang_to_predict_cleanCT.h5ad")
wang = wang[:,wang.var.index.isin(genes_considered.genes_shared_across_datasets.tolist())]

wang_OOD = wang[wang.obs.disease == "T2D"]
new_var = wang_OOD.var.merge(m2h, left_index=True, right_on="ensembl_gene_id", how = "left")
new_var.index = new_var["hsapiens_homolog_associated_gene_name"]
wang_OOD.var = new_var

gt_dbdb_h = gt_dbdb[gt_dbdb.obs.disease == "healthy"]
gt_dbdb_h.raw = None
gt_dbdb_OOD= gt_dbdb[gt_dbdb.obs.disease == "T2D"]

print("Calculating dbdb")

for dataset, X in tqdm(top_preds_dbdb.items()):
    tmp_dbdb_OOD = wang_OOD
    tmp_dbdb_OOD.X = X
    dbdb = sc.concat([gt_dbdb_h, tmp_dbdb_OOD])
    dbdb.obs['stim'] = pd.Categorical(dbdb.obs['disease'], categories=["T2D", "healthy"], ordered=True)
    indices =dbdb.obs.sort_values(['cell_type', 'stim']).index
    dbdb = dbdb[indices,:]
    dbdb = dbdb[dbdb.obs.cell_type == "type B pancreatic cell"]
    res_KEGG =  gp.gsea(data=dbdb.to_df().T, # row -> genes, column-> samples
                        gene_sets="KEGG_2019_Mouse",
                        cls=dbdb.obs.stim,
                        permutation_num=1000,
                        permutation_type='phenotype',
                        outdir=None,
                        method='s2n', # signal_to_noise
                        threads=128,
                        verbose=True)
    tmp_df = res_KEGG.res2d
    tmp_df["dataset_train"] = dataset
    tmp_df.to_csv(f"./GSEA_results_ct/dbdb/{dataset}_enrichment_df.csv")
    # Open a file in binary write mode
    with open(f'./GSEA_results_ct/dbdb/dbdb_{dataset}_gsea_objects.pkl', 'wb') as file:
        pickle.dump(res_KEGG, file)
        
    
print("____________________________ DONE WITH dbdb ____________________________")

gt_mSTZ_h = gt_mSTZ[gt_mSTZ.obs.disease == "healthy"]
gt_mSTZ_h.raw = None
gt_mSTZ_OOD= gt_mSTZ[gt_mSTZ.obs.disease == "T2D"]

for dataset, X in tqdm(top_preds_mSTZ.items()):
    tmp_dbdb_OOD = wang_OOD
    tmp_dbdb_OOD.X = X
    dbdb = sc.concat([gt_dbdb_h, tmp_dbdb_OOD])
    dbdb.obs['stim'] = pd.Categorical(dbdb.obs['disease'], categories=["T2D", "healthy"], ordered=True)
    indices =dbdb.obs.sort_values(['cell_type', 'stim']).index
    dbdb = dbdb[indices,:]
    dbdb = dbdb[dbdb.obs.cell_type == "type B pancreatic cell"]
    res_KEGG =  gp.gsea(data=dbdb.to_df().T, # row -> genes, column-> samples
                        gene_sets="KEGG_2019_Mouse",
                        cls=dbdb.obs.stim,
                        permutation_num=1000,
                        permutation_type='phenotype',
                        outdir=None,
                        method='s2n', # signal_to_noise
                        threads=128,
                        verbose=True)
    tmp_df = res_KEGG.res2d
    tmp_df["dataset_train"] = dataset
    tmp_df.to_csv(f"./GSEA_results_ct/mSTZ/{dataset}_enrichment_df.csv")
    # Open a file in binary write mode
    with open(f'./GSEA_results_ct/mSTZ/mSTZ_{dataset}_gsea_objects.pkl', 'wb') as file:
        pickle.dump(res_KEGG, file)
    
print("____________________________ DONE WITH mSTZ ____________________________")
