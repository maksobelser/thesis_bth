This is the main code repository for my Mester's thesis for the study of Biotechnology on University of Ljubljana. It contains the code used for the model construction and all of the code for data manipulation, model optimisation and different visualisations.

### Experiment design

The experimnet tries to establish if adding specific single-cell RNA-seq data to a cVAE model will improve the out-of-distribution (OOD) gene expression prediction. We use mouse and human type 2 diabetes and healthy data to compare different settings.

![The design of the experiment](fig/experiment_design.pdf)

### Dir structure

- **data**: contains all of the code used for the dataset creation, data manipulation and some additional figures made to check the integrity of the data.
- **experiments**: contains all of the `yaml` and `py` files used for experiment execution using `seml`
- **fig**: figures directory
- **optimisation_evaluation_notebooks**: Notebooks used for the evaluation of experiments and comparisons between different runs.
- **results**: `csv` files of all ther performed experiments with all of the calculated metrics, data used and hyper-paramter settings.
- **simulated_expression_evaluation**: *post hoc* evaluations using the predicted (simulated) expression, e.g. comparisons using GSEA.
- **transVAE** the model directory, the model was constructed using `scvi-tools`