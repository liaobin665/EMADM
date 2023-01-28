"""
An example showing the plot_ks_statistic method used
by a scikit-learn classifier
"""
from __future__ import absolute_import
import pandas as pd
import sys

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer as load_data
sys.path.append("..")
import scikitplot as skplt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('../data/02_feature_selected_XGB.csv')

feature_cols = [col for col in data.columns if col not in ['Address', 'FLAG']]

X = data[feature_cols]
y = data['FLAG']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1986)

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

xgb = XGBClassifier(learning_rate=0.3, n_estimators=70, max_depth=6, min_child_weight=1, gamma=0.0,subsample=0.7,colsample_bytree=0.8,reg_alpha=4e-05)

xgb.fit(X_train, y_train)
probas = xgb.predict_proba(X_test)

skplt.metrics.plot_precision_recall(y_true=y_test, y_probas=probas)
plt.show()

