#!/usr/bin/env python

import gc
import typing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from contextlib import contextmanager

@contextmanager
def plot_and_save(output_path, naming_var):
    try:
        yield
    finally:
        plt.savefig(f'{output_path}/{naming_var}.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        gc.collect()

def process_pairplots(df: pd.DataFrame, output_path: str, naming_var: str, hue_col: str = None):
    """
    Generates Seaborn pairplots for the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        output_path (str): The path to save the plots.
        naming_var (str): The variable name string to use for naming the output file.
        hue_col (str, optional): The column name string to use for stratifying the scatter plot points.

    Example:
        process_pairplots(df, 'output_path', 'naming_var', 'hue_col')

    Returns:
        None
    """
    with plot_and_save(output_path, naming_var):
        pp = sns.pairplot(df, hue=hue_col, diag_kind="kde", kind='reg', corner=True, plot_kws={'scatter_kws': {'alpha': 0.5, 's': 10}}, palette='Set1')
        # add plot title
        plt.suptitle(f'Log-Transformed Total Neoantigen Count vs Immune Features ({naming_var})', fontsize=28, fontweight='medium')

        # Iterate through the axes and set bold titles
        for i, ax in enumerate(pp.axes.flat):
            if ax is not None:
                if ax.get_xlabel() == "TotalNeo_Count":
                    ax.set_xlabel(ax.get_xlabel(), fontweight='bold', fontsize=12, color='red')
                else:
                    ax.set_xlabel(ax.get_xlabel(), fontweight='bold', fontsize=12)
                
                # Handle y-axis labels (only for the leftmost column)
                if i % pp.axes.shape[1] == 0:  # Check if it's the first column
                    if ax.get_ylabel() == "TotalNeo_Count":
                        ax.set_ylabel(ax.get_ylabel(), fontweight='bold', fontsize=12, color='red')
                    else:
                        ax.set_ylabel(ax.get_ylabel(), fontweight='bold', fontsize=12)

# define a heatmap plot function
def process_heatmaps(df: pd.DataFrame, output_path: str, naming_var: str):
    with plot_and_save(output_path, naming_var):
        corr_df = df.drop(columns='Batch').corr(method='spearman')
        # round to 2 decimal places
        corr_df = corr_df.round(2)

        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_df, dtype=bool))

        plt.figure(figsize=(12, 10))
        # Create the correlation matrix and represent it as a heatmap
        hm = sns.heatmap(corr_df, annot = True, cmap = 'coolwarm', square = True, linewidths=0.5, mask=mask, cbar_kws={"shrink": .5})

        # Get current labels
        ylabels = hm.get_yticklabels()
        xlabels = hm.get_xticklabels()

        # Hide the first y-axis label and the last x-axis label
        ylabels[0].set_visible(False)
        xlabels[-1].set_visible(False)

        # Rotate and align the tick labels
        plt.setp(xlabels, rotation=45, ha='right')

        # Change color of specific x-axis label
        for label in xlabels:
            if label.get_text() == "TotalNeo_Count":
                label.set_color('red')  # Change color to red
                label.set_fontweight('bold')

        # Removes all ticks
        hm.tick_params(left=False, bottom=False)
        
        hm.set_title(f'{naming_var}', fontsize=16, x=0.4)
        # plt.tight_layout(pad=1.1)


def subset_df_by_columns(df: pd.DataFrame, num_subsets: int, x_variable: str) -> typing.DefaultDict:
    """Subset a DataFrame into approximately equal groups of columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        num_subsets (int): The desired number of subsets.
        x_variable (str): The name of the column to use as the x-axis variable.

    Returns:
        typing.DefaultDict: A dictionary containing the subsets, where the keys are the subset indices
    """
    if df.columns[0] != 'Batch' and df.columns[1] != x_variable:
        raise ValueError(f"The first two columns of the DataFrame must be 'Batch' and the specified X variable ({x_variable}).")

    # slice the dataframe
    df_x = df.iloc[:, :2]
    df_y = df.iloc[:, 2:]
    # Get the number of columns in the remaining DataFrame
    num_col_Y = len(df_y.columns)

    # Calculate the number of columns per subset in the Y var df
    cols_per_subset, remainder = divmod(num_col_Y, num_subsets)

    # Create a list of column indices for each subset
    col_indices = []
    start = 0
    for i in range(num_subsets):
        end = start + cols_per_subset
        if i < remainder:
            end += 1
        col_indices.append(list(range(start, end)))
        start = end

    # Subset the DataFrame based on the column indices
    subsets = {i: df_y.iloc[:, indices] for i, indices in enumerate(col_indices)}

    # map a concat operation for all the dfs in the dict
    for key, value in subsets.items():
        subsets[key] = pd.concat([df_x, value], axis=1)

    return subsets

############################ START WORKFLOW #############################################
# set up variables
NUM_SS = 12
out_path_pairplot = '/home/suffian/repos/CRM-neoantigen-on-immune-scores-nb/output-data/plots/pairplots'
out_path_heatmap = '/home/suffian/repos/CRM-neoantigen-on-immune-scores-nb/output-data/plots/heatmaps'

# read in data into dF
df = pd.read_csv("/home/suffian/repos/CRM-neoantigen-on-immune-scores-nb/input-data/SA/data_updated230524_new_excludedIHC.tsv",sep="\t")

# exclude the 29 Cibersort scores, leaving only 3
dfd = df.drop(columns=['Bindea_full', 'Expanded_IFNg', 
        'C_Bcellsmemory','C_Plasmacells','C_TcellsCD8','C_TcellsCD4naive',
         'C_TcellsCD4memoryactivated','C_Tcellsfollicularhelper',
         'C_Tcellsregulatory(Tregs)','C_Tcellsgammadelta','C_NKcellsresting',
         'C_NKcellsactivated', 'C_Monocytes', 'C_MacrophagesM0',
         'C_MacrophagesM1','C_Dendriticcellsresting',
         'C_Dendriticcellsactivated', 'C_Mastcellsresting',
         'C_Mastcellsactivated','C_Eosinophils', 'C_Neutrophils', 'S_PAM100HRD'])

# subset df into just TotalNeo_Count (as X variables) and the immune scores as Y variables
dfd_x = dfd.drop(columns = ['PAM50', 'Subtype', 'HR_status',	'HER_status', 'Age', 'AgeGroup', 'Stage', 'TumorGrade', 'TumourSize', 'FusionNeo_Count', 'FusionNeo_bestScore','FusionTransscript_Count', 'Fusion_T2NeoRate', 'SNVindelNeo_Count', 'SNVindelNeo_IC50', 'SNVindelNeo_IC50Percentile'])

# let's drop all NaN for now
dfd_xc = dfd_x.dropna()

dfd_ss = dfd_xc.set_index('ID')
ss_cols = list(dfd_ss.columns)

# subset the dataset using the NUM_SS constant
len(ss_dict := subset_df_by_columns(dfd_ss, NUM_SS, 'TotalNeo_Count'))

# X variable TotalNeo_Count should be transformed due to massive outliers
# Apply log transformation
# IMPRES column is a discrete score so it does not make sense to have it log-transformed. Exclude it
ss_logtrans_dict = {}

for key, df in ss_dict.items():
    if key == 0:
        ss_logtrans_dict[key] = df[['Batch', 'IMPRES']].join(df.drop(['Batch', 'IMPRES'], axis=1).apply(np.log1p))
        # switch the position of IMPRES with TotalNeo_Count columns with each other
        col_tokeep = [col for col in ss_logtrans_dict[key].columns if col not in ['Batch', 'TotalNeo_Count', 'IMPRES']]
        new_order = ['Batch', 'TotalNeo_Count', 'IMPRES'] + col_tokeep
        ss_logtrans_dict[key] = ss_logtrans_dict[key][new_order]
    else:
        ss_logtrans_dict[key] = df[['Batch']].join(df.drop('Batch', axis=1).apply(np.log1p))

# now loop through the dictionary of the subset dfs, and plot the same pairplot for each, saving the plots to file
for key, df_ss in ss_logtrans_dict.items():
    if key < 10:
        pp_file = 'Pairplot_dataFrame-logt-allxcIMPRES-0' + str(key)
        hm_file = 'Heatmap_dataFrame-logt-allxcIMPRES-0' + str(key)
        print(pp_file, hm_file)
        process_pairplots(df_ss, out_path_pairplot, pp_file, 'Batch')
        process_heatmaps(df_ss, out_path_heatmap, hm_file)
    else:
        pp_file = 'Pairplot_dataFrame-logt-allxcIMPRES-' + str(key)
        hm_file = 'Heatmap_dataFrame-logt-allxcIMPRES-' + str(key)
        print(pp_file, hm_file)
        process_pairplots(df_ss, out_path_pairplot, pp_file, 'Batch')
        process_heatmaps(df_ss, out_path_heatmap, hm_file)




print('Done with plotting!')