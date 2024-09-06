## Dynamics of Neoantigens and Immune Signature Scores

This repo serves to document the data analyses both what have been done by MH, and what I will do as SA. 

### Summary of Work Logs
MH did quite a substantial amount of EDA already on the datasets, but I am having trouble figuring out where to pick up the project when they left so I have decided to note down what MH have carried out prior to my taking over.

#### Notes
1. MH noticed a batch effect (~*how?*~ *apparently the immune scores in this dataset were generated via GSVA, and it might have accentuated any underlying batch effect. MH redid the analysis with ssGSEA which ameliorated the dataset*) so they added the 'Batch' label as a separate column and did all of the analyses on the separate batches.

2. MH explored the correlation between the immune scores (IS) with the extra clinical variables (HR Status, Age, Tumor Grade etc.) Their notes are in `input-data/MH/NEW-ssGSEA/attempt1-mixedYSJ-beforeb1b2separateAnalysis`. They also plotted individual box plots comparing ISs with the neoantigen groups hued by batch, with statistical tests. They also tried lasso-regularized linear regression model but the result was not motivating.

3. MH then conducted a second round of playtesting (found at `input-data/MH/NEW-ssGSEA/attempt2-b1b2separateYSZ-separateAnalysis`) whereby they decided on Yeo-Johnson normalization and then z-score standardization on separate batch datasets. Note that the clinical variables have been ordinal, label-encoded (***will consider one-hot encoding in the future***). They then fit the scaled dataset to a random forest regressor model for feature importance assesment.  


### Plan

1. Test log transformation first as the exploratory first step. There are outliers that are still rather skew-causing (`df-01: S_Lymph_Vessels`). Additionally, there are also distributions that are rather non-normal (`df-02: S_CD8`). MH ended up using Yeo-Johnson instead of log + Box-Cox, because YJ can handle zero and negative values. I might just stick with YJ transformation on all columns except `Batch` and `IMPRES`, which is discrete, ordinal data. (~*is Z-score standardization necessary?*~ *Apparently, centered data is required for Support Vector Regressor*)

2. Run SVR and XGBoost after feature scaling and feature selection.

### Outstanding Questions

A. Is there a batch effect? 

B. How to best transform categorical variables? XGBoost works with numerics only so one-hot encoding is best. SVM(R) also needs categorical variables to be encoded. 

C. IMPRES is a discrete ordinal variable, what is the best way of encoding it?

D. It has been postulated that centering and scaling data together prior to splitting into training and testing set will leak some information from the training to the testing set, and lead to poorer model performance. Maybe I should normalize after splitting?

Note: Binning the immune scores to see if there is a thresholding effect