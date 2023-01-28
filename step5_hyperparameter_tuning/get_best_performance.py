# 得到所有 以 XGB 特征选择后的，模型的性能输出
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
from train_model_utils import  cv_bin_classifiy_performance, get_all_classfiers, cv_bin_classifiy_performance_with_std

data = pd.read_csv('../data/02_feature_selected_XGB.csv')

feature_cols = [col for col in data.columns if col not in ['Address', 'FLAG']]

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

cv_bin_classifiy_performance_with_std(X=X.values, y=y.values, splits_num=5, classfiers={'XGB':xgb}, random_seed=1090, outputName='XGB_tunning_')
