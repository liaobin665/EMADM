# 4.3节 得到不同特征集下的性能表现
import catboost
import pandas as pd
import sys

import sklearn.tree
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import xgboost
from xgboost import XGBClassifier

sys.path.append('../utils_lb')

# from train_model_utils import get_all_classfiers, cv_bin_classifiy_performance_with_std, get_basic_classfiers
from train_model_utils import cv_bin_classifiy_performance, get_all_classfiers, cv_bin_classifiy_performance_with_std

data = pd.read_csv('../data/02_feature_selected_XGB.csv')

feature_cols = [col for col in data.columns if col not in ['Address', 'FLAG']]

none_ERC20_features = ['Avg_min_between_sent_tnx', 'Avg_min_between_received_tnx', 'Time_Diff_between_first_and_last_(Mins)', 'Sent_tnx',
                       'Unique_Received_From_Addresses', 'Unique_Sent_To_Addresses',
                       'min_value_received', 'max_value_received', 'avg_val_received', 'min_val_sent',
                       'total_transactions_(including_tnx_to_create_contract', 'total_ether_received']

ERC20_features = ['Total_ERC20_tnxs', 'ERC20_total_Ether_received', 'ERC20_total_ether_sent', 'ERC20_uniq_sent_addr', 'ERC20_uniq_rec_addr',
                  'ERC20_uniq_rec_contract_addr', 'ERC20_min_val_rec', 'ERC20_max_val_rec', 'ERC20_avg_val_rec', 'ERC20_min_val_sent',
                  'ERC20_max_val_sent', 'ERC20_avg_val_sent', 'ERC20_uniq_sent_token_name', 'ERC20_uniq_rec_token_name',
                  'ERC20_most_sent_token_type_ ', 'ERC20_most_sent_token_type_BAT', 'ERC20_most_sent_token_type_Golem',
                  'ERC20_most_sent_token_type_Livepeer Token', 'ERC20_most_sent_token_type_None', 'ERC20_most_sent_token_type_Reputation',
                  'ERC20_most_sent_token_type_StatusNetwork', 'ERC20_most_sent_token_type_Tronix', 'ERC20_most_sent_token_type_blockwell.ai KYC Casper Token',
                  'ERC20_most_rec_token_type_ ', 'ERC20_most_rec_token_type_Blockwell say NOTSAFU', 'ERC20_most_rec_token_type_Free BOB Tokens - BobsRepair.com',
                  'ERC20_most_rec_token_type_Golem']

# 在这里修改特征集
X = data[feature_cols]

y = data['FLAG']

parameters = {
    'learning_rate': [0.3],
    'n_estimators': [70],
    'max_depth':[6],
    'min_child_weight':[1],
    'gamma':[0.0],
    'subsample': [0.7],
    'colsample_bytree': [0.8],
    'reg_alpha':[4e-05],
    'seed':[1024]
}

xgb = XGBClassifier(learning_rate=0.3, n_estimators=70,max_depth=6,min_child_weight=1, gamma=0.0,subsample=0.7,colsample_bytree=0.8,reg_alpha=4e-05)

cv_bin_classifiy_performance_with_std(X=X.values, y=y.values, splits_num=5, classfiers={'XGB':xgb}, random_seed=1090, outputName='total_XGB_diff_feature_set_')
