import time
import numpy as np
from bioservices import KEGG
import pandas as pd
import time

print("reading KEGG data")
df = pd.read_csv("KEGG_intermediate.csv")
df["associated_pathways"] = np.NaN

print("Initializing KEGG")
k = KEGG()
k.organism = "mmu"

# Initialize counters
genes_without_pathways = 0
total_genes = len(df["NCBI Entrez ID"])
all_pathways = set()
warnings_count = 0

# Create a tqdm iterator object
processed_genes = 1
print("Starting to process")
# Iterate over genes
for gene in df["NCBI Entrez ID"]:
    
    data = k.get(f"mmu:{gene}")
    dict_data = k.parse(data)

    if "PATHWAY" in dict_data.keys():
        pathways = list(dict_data["PATHWAY"].values())
        index = df[df["NCBI Entrez ID"] == gene].index
        if not index.empty:
            df.at[index[0], "associated_pathways"] = pathways
        else:
            print(f"Gene {gene} not found in DataFrame.")
        all_pathways.update(pathways)
    else:
        genes_without_pathways += 1

    percent_without_pathways = (genes_without_pathways / processed_genes) * 100
    print(f"Processed gene {processed_genes} and {percent_without_pathways:.2f}% genes without pathways")
    processed_genes += 1
    time.sleep(3)
    
df.to_csv("KEGG_final.csv", index = False)