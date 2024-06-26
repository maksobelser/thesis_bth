import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_plot_jitter_means(df, focused_columns, ylim = None):
    """
    Plots jittered points with averages for specified columns. 
    It ranks the datasets depending on the mean value and selects 
    best performing configuration in terms of the number of layers.

    Parameters:
    - df (DataFrame): Input DataFrame containing the data
    - focused_columns (list): List of columns to focus analysis on
    """
    
    def plot_jittered_points_subplot(df, groups_mean, highest_mean_df, focused_col, ax, ylim: list = None):
        # Sorting original dataframe based on the mean values of the groups
        sorted_groups = highest_mean_df.sort_values(by=focused_col, ascending = False)

        sorted_df = df.set_index(['config.addl_dataset_train', 'config.n_layers']).loc[sorted_groups.set_index(['config.addl_dataset_train', 'config.n_layers']).index]
        # Plotting jittered points
        sns.stripplot(x=focused_col, y="config.addl_dataset_train", data=sorted_df, jitter=True, ax=ax, hue = "dataset_group")

        # Highlighting the highest mean group
        for label, row in highest_mean_df.iterrows():
            ax.scatter(row[focused_col], row['config.addl_dataset_train'], color='black', s=30, zorder=5)

        ax.set_title(f'{focused_col}: Jittered Points with Averages')
        ax.set_xlabel('Value')
        ax.set_ylabel('Dataset')
        ax.tick_params(axis='x', rotation=45)
        if ylim:
            ax.axis(xmin=ylim[0],xmax=ylim[1])
    
    # Group by the specified columns
    groups = df.groupby(["config.addl_dataset_train", "config.n_layers"])

    # Calculate mean, standard deviation for focused columns
    groups_mean = groups[focused_columns].mean()

    # Resetting index for better readability
    groups_mean.reset_index(inplace=True)

    # Group by 'config.addl_dataset_train'
    grouped_summary = groups_mean.groupby('config.addl_dataset_train')

    # Creating dataframes for the highest mean for each focused column
    highest_mean_dfs = [
        grouped_summary.apply(lambda x: x.nlargest(1, col)).reset_index(drop=True)
        for col in focused_columns
    ]
    
    embedding_based_human = ['embedding_top30_human.h5ad','embedding_top50_human.h5ad','embedding_top70_human.h5ad']
    embedding_based_mouse = ['embedding_top100_mouse.h5ad',
     'embedding_top30_mouse.h5ad',
     'embedding_top50_mouse.h5ad',
     'embedding_top70_mouse.h5ad']
    baseline = ['train_adata_baseline.h5ad']
    extra_human = ['extra_human_chem.h5ad',
     'extra_human_neonatal.h5ad',
     'extra_human_preT2D.h5ad',]
    extra_mouse = ['extra_mouse_Embryonic.h5ad',
     'extra_mouse_T1D.h5ad',
     'extra_mouse_aged.h5ad',
     'extra_mouse_chem.h5ad',
     'extra_mouse_young.h5ad',]
    random =  ['random_human_seed_42.h5ad',
     'random_human_seed_43.h5ad',
     'random_mouse_seed_42.h5ad',
     'random_mouse_seed_43.h5ad']
        
    df["dataset_group"] = None
    df.loc[df["config.addl_dataset_train"].isin(embedding_based_human), "dataset_group"] = "Emebdding based - Human"
    df.loc[df["config.addl_dataset_train"].isin(embedding_based_mouse), "dataset_group"] = "Emebdding based - Mouse"
    df.loc[df["config.addl_dataset_train"].isin(baseline), "dataset_group"] = "Baseline"
    df.loc[df["config.addl_dataset_train"].isin(extra_human), "dataset_group"] = "Extra conditions - Human"
    df.loc[df["config.addl_dataset_train"].isin(extra_mouse), "dataset_group"] = "Extra conditions - Mouse"
    df.loc[df["config.addl_dataset_train"].isin(random), "dataset_group"] = "Random samplin of cell types"

    # Setting up the figure for side by side plots
    fig, axes = plt.subplots(nrows=1, ncols=len(focused_columns), figsize=(20, 6))

    # Plotting for each focused column
    for i, col in enumerate(focused_columns):
        plot_jittered_points_subplot(
            df, groups_mean, highest_mean_dfs[i], col, axes[i], ylim,
        )

    # Adjust layout
    plt.tight_layout()
    plt.show()
        
        
def analyze_and_plot_jitter_medians(df, focused_columns, ylim: list = None):
    """
    Plots jittered points with medians for specified columns. 
    It ranks the datasets depending on the meadian value and selects 
    best performing configuration in terms of the number of layers.

    Parameters:
    - df (DataFrame): Input DataFrame containing the data
    - focused_columns (list): List of columns to focus pltting on
    """
    
    def plot_jittered_points_subplot(df, groups_median, highest_median_df, focused_col, ax, ylim):
        # Sorting original dataframe based on the median values of the groups
        sorted_groups = highest_median_df.sort_values(by=focused_col, ascending = False)
        sorted_df = df.set_index(['config.addl_dataset_train', 'config.n_layers']).loc[sorted_groups.set_index(['config.addl_dataset_train', 'config.n_layers']).index]
        # Plotting jittered points
        sns.stripplot(x=focused_col, y="config.addl_dataset_train", data=sorted_df, jitter=True, ax=ax, hue='dataset_group')

        # Highlighting the highest median group
        for label, row in highest_median_df.iterrows():
            ax.scatter(row[focused_col], row['config.addl_dataset_train'], color='blue', s=50, zorder=5)

        ax.set_title(f'{focused_col}: Jittered Points with Medians')
        ax.set_xlabel('Value')
        ax.set_ylabel('Dataset')
        ax.tick_params(axis='x', rotation=45)
        if ylim:
            ax.axis(xmin=ylim[0],xmax=ylim[1])

    # Group by the specified columns
    groups = df.groupby(["config.addl_dataset_train", "config.n_layers"])

    # Calculate median for focused columns
    groups_median = groups[focused_columns].median()

    # Resetting index for better readability
    groups_median.reset_index(inplace=True)

    # Group by 'config.addl_dataset_train'
    grouped_summary = groups_median.groupby('config.addl_dataset_train')

    # Creating dataframes for the highest median for each focused column
    highest_median_dfs = [
        grouped_summary.apply(lambda x: x.nlargest(1, col)).reset_index(drop=True)
        for col in focused_columns
    ]

    # Setting up the figure for side by side plots
    fig, axes = plt.subplots(nrows=1, ncols=len(focused_columns), figsize=(20, 6))
     
    embedding_based_human = ['embedding_top30_human.h5ad','embedding_top50_human.h5ad','embedding_top70_human.h5ad']
    embedding_based_mouse = ['embedding_top100_mouse.h5ad',
     'embedding_top30_mouse.h5ad',
     'embedding_top50_mouse.h5ad',
     'embedding_top70_mouse.h5ad']
    baseline = ['train_adata_baseline.h5ad']
    extra_human = ['extra_human_chem.h5ad',
     'extra_human_neonatal.h5ad',
     'extra_human_preT2D.h5ad',]
    extra_mouse = ['extra_mouse_Embryonic.h5ad',
     'extra_mouse_T1D.h5ad',
     'extra_mouse_aged.h5ad',
     'extra_mouse_chem.h5ad',
     'extra_mouse_young.h5ad',]
    random =  ['random_human_seed_42.h5ad',
     'random_human_seed_43.h5ad',
     'random_mouse_seed_42.h5ad',
     'random_mouse_seed_43.h5ad']
        
    df["dataset_group"] = None
    df.loc[df["config.addl_dataset_train"].isin(embedding_based_human), "dataset_group"] = "Emebdding based - Human"
    df.loc[df["config.addl_dataset_train"].isin(embedding_based_mouse), "dataset_group"] = "Emebdding based - Mouse"
    df.loc[df["config.addl_dataset_train"].isin(baseline), "dataset_group"] = "Baseline"
    df.loc[df["config.addl_dataset_train"].isin(extra_human), "dataset_group"] = "Extra conditions - Human"
    df.loc[df["config.addl_dataset_train"].isin(extra_mouse), "dataset_group"] = "Extra conditions - Mouse"
    df.loc[df["config.addl_dataset_train"].isin(random), "dataset_group"] = "Random samplin of cell types"
    
    # Plotting for each focused column
    for i, col in enumerate(focused_columns):
        plot_jittered_points_subplot(
            df, groups_median, highest_median_dfs[i], col, axes[i], ylim,
        )
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    