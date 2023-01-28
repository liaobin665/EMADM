import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn import over_sampling,under_sampling,combine

data = pd.read_csv('../data/02_feature_selected_XGB.csv')

feature_cols = [col for col in data.columns if col not in ['Address', 'FLAG']]

X = data[feature_cols].values
y = data['FLAG'].values

def get_sklearn_model(model_name,param=None):
    #朴素贝叶斯
    if model_name=='NB':
        model = MultinomialNB(alpha=0.01)
    #逻辑回归
    elif model_name=='LR':
        model = LogisticRegression(penalty='l2')
    # KNN
    elif model_name=='KNN':
        model = KNeighborsClassifier()
    #随机森林
    elif model_name=='RF':
        model = RandomForestClassifier()
    #决策树
    elif model_name=='DT':
        model = DecisionTreeClassifier()
    #向量机
    elif model_name=='SVC':
        model = SVC(kernel='rbf')
    #GBDT
    elif model_name=='GBDT':
        model = GradientBoostingClassifier()
    #XGBoost
    elif model_name=='XGB':
        model = XGBClassifier()
    #lightGBM
    elif model_name=='LGB':
        model = LGBMClassifier()
    else:
        print("wrong model name!")
        return
    if param is not None:
        model.set_params(**param)
    return model


# 绘制验证曲线
# 可以通过绘制验证曲线，可视化的了解调参的过程。
def grid_plot(classifier, cvnum, param_range, param_name, param=None):
    from sklearn.model_selection import validation_curve
    train_scores, test_scores = validation_curve(
        get_sklearn_model(classifier, param), X, y,
        param_name=param_name, param_range=param_range,
        cv=cvnum, scoring='f1', n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with " + param_name)
    plt.xlabel(param_name)
    plt.ylabel("F1 Score")
    plt.ylim(0.9, 1.02)
    plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.show()

# 默认参数
params = {
    'booster':'gbtree',
    'learning_rate': 0.3,
    'n_estimators': 100,
    'max_depth': 6,
    'min_child_weight': 1,
    'gamma': 0.0,
    'subsample': 1,
    'colsample_bytree': 1,
    # 'reg_alpha': 0
}

# learning_rate_range = [0.0001,0.001,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9,1]

# n_estimators_range = [1,2,3,4,5,10,20,30,40,50,60,70,80,90,100,120,140,160,200,250,300]
# max_depth_range = [1,2,3,4,5,6,7,8,9,10,15,20,30,40,50]
# min_child_weight_range = [0.0001,0.001,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.7,0.8,0.9,1]
# gamma_range = [0.0001,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.9,1]

# subsample_range = [0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

# colsample_bytree_range = [0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

reg_alpha_range = [0,0.001,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

grid_plot('XGB', 3, reg_alpha_range, 'reg_alpha', param=params)

plt.show()
