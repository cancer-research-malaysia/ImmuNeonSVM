#!/usr/bin/env python

import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from feature_engine.pipeline import Pipeline
from feature_engine.encoding import OneHotEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.transformation import YeoJohnsonTransformer
from sklearn.metrics import mean_squared_error

def wrangle_raw_dataframe(file_path):
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
    X_features = ['Subtype_HR+/HER2-', 'Subtype_HR+/HER2+', 'Subtype_TNBC', 'Subtype_HR-/HER2+', 'Age', 'TumorGrade', 'TumourSize', 'FusionNeo_Count', 'FusionNeo_bestIC50', 'FN/FT_Ratio', 'SNVindelNeo_Count', 'SNVindelNeo_IC50']

    X_features_nocat = ['Age', 'TumorGrade', 'TumourSize', 'FusionNeo_Count', 'FusionNeo_bestIC50', 'FN/FT_Ratio', 'SNVindelNeo_Count', 'SNVindelNeo_IC50']

    Y_labels_all = [col for col in dfd.drop(columns=['Subtype']).columns if col not in X_features]

    print(f"Number of X features: {len(X_features)}")
    print(f"Number of all Y labels: {len(Y_labels_all)}")

    return dfenc, X_features, Y_labels_all

def split_transform_dataframe(dfenc, X_features, Y_labels_all):
    ### Data splitting
    ######### Split dataset
    # get X dataframe
    X = dfenc[X_features]

    # get Y labels
    Y = dfenc[Y_labels_all]

    # perform train-test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    ### Data transformation
    # get X variables for YJ transform
    X_vars_to_transform = ['TumourSize', 'FusionNeo_Count', 'FusionNeo_bestIC50', 'FN/FT_Ratio', 'SNVindelNeo_Count', 'SNVindelNeo_IC50']

    # select variables to scale with StandardScaler
    scale_cols_X = [col for col in X_train.columns.tolist() if col not in ['Age', 'TumorGrade', 'Subtype_HR+/HER2-', 'Subtype_HR+/HER2+', 'Subtype_TNBC', 'Subtype_HR-/HER2+']]

    ### use Pipeline to construct YJ and StandardScaler transformations
    # on X split data
    preprocess_pipeline_X = Pipeline([
        ('yeo_johnson', YeoJohnsonTransformer(variables = X_vars_to_transform)),
        ('scaler', SklearnTransformerWrapper(transformer = StandardScaler(), variables = scale_cols_X))
    ])
    # Fit the pipeline to the training data
    preprocess_pipeline_X.fit(X_train)
    # Transform the training data
    X_train_yjs = preprocess_pipeline_X.transform(X_train)
    # Transform the test data
    X_test_yjs = preprocess_pipeline_X.transform(X_test)

    # on Y split data
    # select Y variables to scale
    scale_cols_Y = Y_train.columns.tolist()
    # Create the pipeline
    preprocess_pipeline_Y = Pipeline([
        ('yeo_johnson', YeoJohnsonTransformer()),
        ('scaler', SklearnTransformerWrapper(transformer = StandardScaler(), variables = scale_cols_Y))
    ])
    # Fit the pipeline to the training data
    preprocess_pipeline_Y.fit(Y_train)
    # Transform the training data
    Y_train_yjs = preprocess_pipeline_Y.transform(Y_train)
    # Transform the test data
    Y_test_yjs = preprocess_pipeline_Y.transform(Y_test)

    return X, Y, X_train_yjs, X_test_yjs, Y_train_yjs, Y_test_yjs

def run_grid_search_CV(y_target, X, Y, X_train_yjs, Y_train_yjs, X_test_yjs, Y_test_yjs):
    # assign untransformed, raw target data
    y_train_ori = Y_train_yjs[y_target]
    y_test_ori = Y_test_yjs[y_target]

    # Define the XGBoost model
    model = xgb.XGBRegressor()

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'subsample': [0.5, 0.7, 1.0],
        'colsample_bytree': [0.5, 0.7, 1.0]
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train_yjs, Y_train_yjs)

    # Print the best parameters and the corresponding score
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", -grid_search.best_score_)

    # Get the best estimator
    best_model = grid_search.best_estimator_

    # Make predictions on the test data
    y_pred = best_model.predict(X_test_yjs)

    # Evaluate the model
    mse = mean_squared_error(Y_test_yjs, y_pred)
    print("Mean Squared Error: ", mse)

def main():
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

    ###### STARTING WORKFLOW
    dfenc, X_features, Y_labels_all = wrangle_raw_dataframe(file_path)
    split_transform_dataframe(dfenc, X_features, Y_labels_all)

if __name__ == "__main__":
    # main()
    pass

