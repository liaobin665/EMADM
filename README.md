# Code repository for paper <EMADM: a highly accurate and interpretable model for detecting malicious accounts on the Ethereum blockchain>


## Code structure description:
## data : 
 The data directory contains the dataset before data pre-processing (Ethereum_transactional_dataset_including_ERC20.csv), and pre-processing results file (01_preprocessed_dataset.csv, 01_preprocessed_dataset_not_dummy.csv) , feature selection results data file (02_feature_selected_XGB.csv).  Feature_Description.txt describe the most importance features in the trainning data.
## step1_data_preprocess: data preprocessing codes.
## step2_get_performance: get or compare each models performance.
## step3_explainability: Section 5 in paper, for EMADM's explainability anlysis from model, feature and sample level.
## step4_plot_and_model_anlysis: plot or anlysis model performance, and plot the figs in the paper.
## step5_hyperparameter_tuning: hyperparameter tuning codes.
## utils_lb: some util libs for train and get the model result.


