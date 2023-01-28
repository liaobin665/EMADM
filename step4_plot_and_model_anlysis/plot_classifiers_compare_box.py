# 对比各分类器的性能box
import numpy as np
from sklearn.svm import LinearSVC
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import ExtraTreeClassifier
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold

from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,roc_curve
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
# import shap
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('../data/02_feature_selected_XGB.csv')

feature_cols = [col for col in data.columns if col not in ['Address', 'FLAG']]
X = data[feature_cols]
y = data['FLAG']


models = []
# models.append(('LogisticRegression', LogisticRegression()))
# models.append(('LinearSVC', LinearSVC()))
# models.append(('KNeighborsClassifier', KNeighborsClassifier()))
# models.append(('BernoulliNB', BernoulliNB()))
models.append(('DecisionTree', DecisionTreeClassifier()))
models.append(('ExtraTree', ExtraTreeClassifier()))
models.append(('RandomForest', RandomForestClassifier()))
models.append(('GradientBoosting', GradientBoostingClassifier()))
models.append(('AdaBoost', AdaBoostClassifier(random_state=100)))
models.append(('HistGradientBoosting', HistGradientBoostingClassifier()))
# models.append(('RidgeClassifier', RidgeClassifier()))
# models.append(('MLPClassifier', MLPClassifier()))
models.append(("LGBM", LGBMClassifier()))
models.append(('CatBoost', CatBoostClassifier()))
models.append(('XGB', XGBClassifier()))

results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=5,random_state=1090,shuffle=True)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
    # cv_results = cross_val_score(model, X, y, cv=kfold, scoring="recall")
    # cv_results = cross_val_score(model, X, y, cv=kfold, scoring="f1")

    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
plt.style.use('fast')
plt.xlabel("Algorithm name", fontsize=14)
plt.xticks(rotation=45)
# plt.ylabel("f1",fontsize=14)
plt.ylabel("accuracy",fontsize=14)
plt.tick_params(labelsize = 12)
plt.boxplot(results,sym='r*',patch_artist=True)
ax.set_xticklabels(names)
plt.legend(fontsize=16)
plt.grid()
plt.savefig('accuracy.box.tif', dpi=500, bbox_inches='tight')
plt.show()