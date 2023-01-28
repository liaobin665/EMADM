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
from train_model_utils import  cv_bin_classifiy_performance, get_basic_classfiers

data = pd.read_csv("../data/01_preprocessed_dataset.csv")

feature_cols = [col for col in data.columns if col not in ['Address', 'FLAG']]

X = data[feature_cols]
y = data['FLAG']

# 特征选择操作
# 特征选择方法1：删除方差较小的要素（VarianceThreshold）
# from sklearn.feature_selection import VarianceThreshold
#
# selector = VarianceThreshold(threshold=0.01)
# X_sel = selector.fit_transform(X)
# print("训练数据删选前的特征维度：",X.shape)
# print("特征删选后的特征维度：",X_sel.shape)

# 特征选择方法2：单变量特征选择，基于单变量统计检验选择最佳特征（基于互信息）
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(score_func = mutual_info_classif, k=20)
X_sel = selector.fit_transform(X,y)

##  使用LR拟合的参数进行变量选择（L2范数进行特征选择） 方法三
# from sklearn.feature_selection import SelectFromModel
# LR = LogisticRegression(penalty='l2', C=1)
# LR = LR.fit(X, y)
# model = SelectFromModel(LR, prefit=True)
# X_sel = model.transform(X)

# 递归功能消除（方法3）：选定模型拟合，进行递归拟合，每次把评分低得特征去除，重复上诉循环。
# from sklearn.feature_selection import RFECV
# from sklearn.ensemble import RandomForestClassifier
#
# clf = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=4, n_jobs=-1)
# selector = RFECV(clf, step=1, cv=2)
# X_sel = selector.fit_transform(X, y)
# print(selector.support_)
# print(selector.ranking_)

# # 使用树模型选择特征 方法N
# clf = XGBClassifier()
# # clf = LGBMClassifier()
# # clf = catboost.CatBoostClassifier()
# # clf = sklearn.tree.DecisionTreeClassifier()
# clf = clf.fit(X.values, y)
#
# model = SelectFromModel(clf, prefit=True, max_features=50)
# X_sel = model.transform(X)

print('训练数据未特征筛选维度', X.shape)
print('训练数据特征筛选维度后', X_sel.shape)


# cv_bin_classifiy_performance_with_std(X.values, y.values, classfiers=get_basic_classfiers(), random_seed=0, outputName='all_cls_')

cv_bin_classifiy_performance(X=X_sel, y=y, splits_num=5, classfiers=get_basic_classfiers(), random_seed=4, outputName='mutual_info_classif—k=20_feature_select_')
