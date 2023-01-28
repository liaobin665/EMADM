import numpy as np
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,roc_curve
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
from matplotlib.pyplot import MultipleLocator
import catboost
import pandas as pd
import sys

import sklearn.tree
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import xgboost
from xgboost import XGBClassifier

data = pd.read_csv('../data/02_feature_selected_XGB.csv')

feature_cols = [col for col in data.columns if col not in ['Address', 'FLAG']]

X = data[feature_cols]
y = data['FLAG']

#绘制学习曲线，以确定模型的状况
def plot_learning_curve(estimator, title, X, y, cv=None, scoring='precision', n_jobs=1, train_sizes=np.linspace(.05, 1.0, 15)):
    plt.title(title, fontsize=12)
    plt.xlabel("Training examples", fontsize=12)
    plt.ylabel(scoring+" score", fontsize=12)
    # 设置坐标刻度字体大小
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.ylim(0.90, 1.02)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 's--', color="r", label="Training "+scoring+" score")
    plt.plot(train_sizes, test_scores_mean, '*-', color="g", label="Cross-validation "+scoring+" score")

    plt.legend(loc='lower right')
    return plt

cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=1090)

xgb = XGBClassifier(learning_rate=0.3, n_estimators=70,max_depth=6,min_child_weight=1, gamma=0.0,subsample=0.7,colsample_bytree=0.8,reg_alpha=4e-05)

# precision
# plot_learning_curve(xgb, 'Learning curve of EFTDM(evaluated by precision)',X, y, cv=cv)
# accuracy
# plot_learning_curve(xgb, 'Learning curve of EFTDM(evaluated by accuracy)',X, y, cv=cv, scoring='accuracy')
# F1
# plot_learning_curve(xgb, 'Learning curve of EFTDM(evaluated by F1)',X, y, cv=cv, scoring='f1')
# recall
plot_learning_curve(xgb, 'Learning curve of EFTDM(evaluated by recall)', X, y, cv=cv, scoring='recall')

# auc
plot_learning_curve(xgb, 'Learning curve of EFTDM(evaluated by auc)', X, y, cv=cv, scoring='auc')

plt.show()



