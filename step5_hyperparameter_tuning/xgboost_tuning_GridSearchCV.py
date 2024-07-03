import pandas as pd
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

data = pd.read_csv('../data/02_feature_selected_XGB.csv')

feature_cols = [col for col in data.columns if col not in ['Address', 'FLAG']]

X = data[feature_cols]
y = data['FLAG']

# 参数的最佳取值:{'learning_rate': 0.3, 'n_estimators': 70}
# parameters = {
#     'learning_rate': [0.29,0.3,0.31],
#     'n_estimators': [68,69,70,71],
# }

# {'learning_rate': 0.3, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 70}
# parameters = {
#     'learning_rate': [0.3],
#     'n_estimators': [70],
#     'max_depth':[6],
#     'min_child_weight':[1],
# }
# 参数的最佳取值:{'gamma': 0.0, 'learning_rate': 0.3, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 70}
# parameters = {
#     'learning_rate': [0.3],
#     'n_estimators': [70],
#     'max_depth':[6],
#     'min_child_weight':[1],
#     'gamma':[0.0],
#     'subsample': [i / 100.0 for i in range(60, 80, 5)],
#     'colsample_bytree': [i / 100.0 for i in range(80, 100, 5)],
#     'seed':[1024]
# }

# 参数的最佳取值:{'colsample_bytree': 0.8, 'gamma': 0.0, 'learning_rate': 0.3, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 70, 'seed': 1024, 'subsample': 0.7}

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

xgb = XGBClassifier()

gsearch = GridSearchCV(xgb, param_grid=parameters, scoring='f1', cv=5)
gsearch.fit(X, y)

print('参数的最佳取值:{0}'.format(gsearch.best_params_))
print('最佳模型得分:{0}'.format(gsearch.best_score_))
print(gsearch.cv_results_['mean_test_score'])
print(gsearch.cv_results_['params'])


print()
