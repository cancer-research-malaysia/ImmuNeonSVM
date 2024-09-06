## Dynamics of Neoantigens and Immune Signature Scores

This repo serves to document the data analyses both what have been done by MH, and what I will do as SA. 

### Summary of Work Logs
MH did quite a substantial amount of EDA already on the datasets, but I am having trouble figuring out where to pick up the project when they left so I have decided to note down what MH have carried out prior to my taking over.

#### Notes
1. MH noticed a batch effect (***how?***) so they added the 'Batch' label as a separate column and did all of the analyses on the separate batches.

2. MH explored the correlation between the immune scores (IS) with the extra clinical variables (HR Status, Age, Tumor Grade etc.) Her notes are in `input-data/MH/NEW-ssGSEA/attempt1-mixedYSJ-beforeb1b2separateAnalysis`. She also plotted individual box plots comparing ISs with the neoantigen groups hued by batch, with statistical tests. She also tried lasso regression but the result was not motivating.

3. She then conducted a second round of playtesting (found at `input-data/MH/NEW-ssGSEA/attempt2-b1b2separateYSZ-separateAnalysis`) in which she decided on Yeo-Johnson normalization and then z-score standardization on separate batch datasets. Note that the clinical variables have been label-encoded (***maybe consider one-hot encoding?***). She then fit these scaled datasets into a lasso-regularized linear regression model.



### Plan

1. Test log transformation. MH ended up using Yeo-Johnson instead of log + Box-Cox. There are outliers that are still rather skew-causing (`df-01: S_Lymph_Vessels`). Additionally, there are also distributions that are rather non-normal (`df-02: S_CD8`).I might just stick with YJ transformation on all columns except `Batch` and `IMPRES` (which is a discrete, ordinal data)) (~*is Z-score standardization necessary?*~ *Centered data is required for Support Vector Regressor*)

2. Run SVR and XGBoost after feature scaling.

### Outstanding Questions

A. Is there a batch effect? 

B. How to best transform categorical variables? XGBoost works with numerics only so one-hot encoding is best. SVM(R) also needs categorical variables to be encoded. 

C. IMPRES is a discrete ordinal variable, what is the best way of encoding it?

D. It has been postulated that centering and scaling data together prior to splitting into training and testing set will leak some information from the training to the testing set, and lead to poorer model performance. Maybe I should normalize after splitting?

