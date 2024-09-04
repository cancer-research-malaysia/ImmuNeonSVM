## Dynamics of Neoantigens and Immune Signature Scores

This repo serves to document the data analyses both what have been done by MH, and what I will do as SA. 

### Summary of Work Logs
MH did quite a substantial amount of EDA already on the datasets, but I am having trouble figuring out where to pick up the project when they left so I have decided to note down what MH have carried out prior to my taking over.

List:
    1. MH noticed a batch effect (***how?***) so they split added the 'Batch' label as a separate column and did all of the analyses on the separate batches.
    2. MH explored the correlation between the immune scores (IS) with the extra clinical variables (HR Status, Age, Tumor Grade etc.) Her notes are in `input-data/MH/NEW-ssGSEA/attempt1-mixedYSJ-beforeb1b2separateAnalysis`. She also plotted individual box plots comparing ISs with the neoantigen groups hued by batch, with statistical tests. She also tried lasso regression but the result was not motivating.
    3. She then conducted a second round of playtesting (found at `input-data/MH/NEW-ssGSEA/attempt2-b1b2separateYSZ-separateAnalysis`) in which she decided on Yeo-Johnson normalization and then z-score standardization on separate batch datasets. Note that the clinical variables have been label-encoded (***maybe consider one-hot encoding?***). She then fit these scaled datasets into a lasso-regularized linear regression model.



### My Strategy

1. Test log transformation. MH ended up using Yeo-Johnson instead of log + Box-Cox. There are outliers that are still rather skew-causing (`df-01: S_Lymph_Vessels`). Additionally, there are also distributions that are rather non-normal (`df-02: S_CD8`).I might just stick with YJ transformation on all columns except `Batch`) + Z-score standardization.

2. 
