import pandas as pd
import xgboost
from sklearn.impute import KNNImputer,SimpleImputer
import sys
sys.path.append('../utils_lb')

# from train_model_utils import get_all_classfiers, cv_bin_classifiy_performance_with_std, get_basic_classfiers
from train_model_utils import  cv_bin_classifiy_performance, get_basic_classfiers

dataset = pd.read_csv("../data/Ethereum_transactional_dataset_including_ERC20.csv", sep=",")
dataset = dataset.iloc[:, 2:]
dataset.columns = dataset.columns.str.strip()
dataset.columns = dataset.columns.str.replace(' ', '_')

null_fea_names =['Total_ERC20_tnxs', 'ERC20_total_Ether_received', 'ERC20_total_ether_sent', 'ERC20_total_Ether_sent_contract', 'ERC20_uniq_sent_addr', 'ERC20_uniq_rec_addr',
                 'ERC20_uniq_sent_addr.1', 'ERC20_uniq_rec_contract_addr', 'ERC20_avg_time_between_sent_tnx', 'ERC20_avg_time_between_rec_tnx', 'ERC20_avg_time_between_rec_2_tnx',
                 'ERC20_avg_time_between_contract_tnx', 'ERC20_min_val_rec', 'ERC20_max_val_rec', 'ERC20_avg_val_rec', 'ERC20_min_val_sent', 'ERC20_max_val_sent', 'ERC20_avg_val_sent',
                 'ERC20_min_val_sent_contract', 'ERC20_max_val_sent_contract', 'ERC20_avg_val_sent_contract', 'ERC20_uniq_sent_token_name', 'ERC20_uniq_rec_token_name']

print(dataset.isna().sum())
# dataset[null_fea_names] = KNNImputer(n_neighbors=6).fit_transform(dataset[null_fea_names])
dataset[null_fea_names] = SimpleImputer(strategy='constant').fit_transform(dataset[null_fea_names])

# dataset.dropna(how='any', axis=0, inplace=True)
print(dataset.isna().sum())

# get_dummies
data = pd.get_dummies(dataset, columns=['ERC20_most_sent_token_type', 'ERC20_most_rec_token_type'], dummy_na=True)

feature_cols = [col for col in data.columns if col not in ['Address', 'FLAG']]

X = data[feature_cols].values
y = data['FLAG'].values

cv_bin_classifiy_performance(X=X, y=y, splits_num=5, classfiers={'XGBoost':xgboost.XGBClassifier()}, random_seed=4, outputName='constant_SimpleImputer_n_neighbors=6_')