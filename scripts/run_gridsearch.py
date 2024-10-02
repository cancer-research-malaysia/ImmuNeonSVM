#!/usr/bin/env python

import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt

from feature_engine.encoding import OneHotEncoder
from feature_engine.transformation import YeoJohnsonTransformer

def wrangle_raw_dataframe():
    parser = argparse.ArgumentParser(description="Run GridSearchCV on a set of X features and a set of Y labels concurrently.")
    
    parser.add_argument("file_path", required=True, help="raw input dataset file in tab-separated format (tsv).")

    args = parser.parse_args()

    # read script argument and save into a variable
    file_path = args.file_path

    try:
        if os.path.exists(file_path):
            print("Input file exists. Proceeding...")
        else:
            print("File does not exist. Aborting!")
            exit(1)
    except:
        print("Error while checking the existence of the file. Double-check your input file path.")

    #### STARTING WORKFLOW
    # read input file
    df = pd.read_csv(file_path, sep="\t")
    print(f"Input file {file_path} has been read into a dataFrame successfully.")
    print(f"The shape of the dataFrame is {df.shape}.")

    # exclude the 29 Cibersort scores, leaving only 3
    df = df.drop(columns=['Bindea_full', 'Expanded_IFNg', 
        'C_Bcellsmemory','C_Plasmacells','C_TcellsCD8','C_TcellsCD4naive',
         'C_TcellsCD4memoryactivated','C_Tcellsfollicularhelper',
         'C_Tcellsregulatory(Tregs)','C_Tcellsgammadelta','C_NKcellsresting',
         'C_NKcellsactivated', 'C_Monocytes', 'C_MacrophagesM0',
         'C_MacrophagesM1','C_Dendriticcellsresting',
         'C_Dendriticcellsactivated', 'C_Mastcellsresting',
         'C_Mastcellsactivated','C_Eosinophils', 'C_Neutrophils', 'S_PAM100HRD'])

    print(f"Removing CIBERSORT scores except for 3...")
    print(f"The shape of the current dataFrame is {df.shape}.")

    # drop fluff X features, and all NaN for now, and set col 'ID' as index
    dfd = df.drop(columns = ['Batch', 'Stage', 'PAM50', 'HR_status', 'HER_status', 'AgeGroup', 'TotalNeo_Count', 'FusionTransscript_Count', 'SNVindelNeo_IC50Percentile']).dropna().set_index('ID')
    
    print(f"Removing X features that will be excluded from the modeling...")
    print(f"The shape of the current dataFrame is {dfd.shape}.")
    dfd.head()

    # rename the column `Fusion_T2NeoRate` to `FN/FT_Ratio` and `FusionNeo_bestScore` to `FusionNeo_bestIC50`
    dfd.rename(columns={'Fusion_T2NeoRate': 'FN/FT_Ratio'}, inplace=True)
    dfd.rename(columns={'FusionNeo_bestScore': 'FusionNeo_bestIC50'}, inplace=True)

    # change data types accordingly
    print("Changing column datatypes into appropriate types...")
    dfd['Subtype'] = dfd['Subtype'].astype('category')
    dfd['Age'] = dfd['Age'].astype('int64')
    dfd['TumorGrade'] = dfd['TumorGrade'].astype('int64')
    dfd['IMPRES'] = dfd['IMPRES'].astype('int64')
    dfd['FusionNeo_Count'] = dfd['FusionNeo_Count'].astype('int64')
    dfd['SNVindelNeo_Count'] = dfd['SNVindelNeo_Count'].astype('int64')
    dfd['FN/FT_Ratio'] = dfd['FN/FT_Ratio'].astype('float64')

    # one-hot encode the 'Subtype' column
    print("One-hot encoding the 'Subtype' column...")    
    encoder = OneHotEncoder(
    variables=['Subtype'],
    drop_last=False)

    encoder.fit(dfd)
    dfd_ = encoder.transform(dfd)
    dfd_.head()

    print("One-hot encoding completed successfully. Shifting the appended categorical columns to the front of the dataframe...")

    # Specify the encoded columns to shift
    enc_cols = ['Subtype_HR+/HER2-', 'Subtype_HR+/HER2+', 'Subtype_TNBC', 'Subtype_HR-/HER2+']

    # Drop the specified columns and store them
    encoded_df = dfd_[enc_cols]
    dfenc = dfd.drop(columns=['Subtype'])

    # Specify the index where you want to reinsert the columns
    insert_index = 0  # This will insert at the first column

    # Reinsert the columns
    for i, col in enumerate(encoded_df.columns):
        dfenc.insert(insert_index + i, col, encoded_df[col])

    ######## 
    

if __name__ == "__main__":
    # main()
    pass

