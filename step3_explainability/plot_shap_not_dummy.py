import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score,cross_val_predict, train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler,PowerTransformer,LabelEncoder
from sklearn.metrics import accuracy_score,classification_report, recall_score,confusion_matrix, roc_auc_score, precision_score, f1_score, roc_curve, auc, plot_confusion_matrix,plot_roc_curve

import lightgbm as lgb
from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier, plot_importance
from catboost import CatBoostClassifier

import plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
from plotly.offline import iplot
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import shap
import missingno as msno

import warnings
warnings.filterwarnings("ignore")
# %matplotlib inline

accuracy= []
recall =[]
roc_auc= []
precision = []

data = pd.read_csv("../data/01_preprocessed_dataset_not_dummy.csv")
data.rename(columns ={'total_transactions_(including_tnx_to_create_contract':'total_transactions'}, inplace=True)

feature_cols = [col for col in data.columns if col not in ['Address', 'FLAG']]
X = data[feature_cols]
y = data['FLAG']
# data.describe()
for col in X.columns:
    col_type = X[col].dtype
    if col_type == 'object' or col_type.name == 'category':
        X[col] = X[col].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lgbcls = lgb.LGBMClassifier(random_state=0)

lgbcls.fit(X_train, y_train)

y_pred = lgbcls.predict(X_test)

accuracy.append(round(accuracy_score(y_test, y_pred),4))
recall.append(round(recall_score(y_test, y_pred),4))
roc_auc.append(round(roc_auc_score(y_test, y_pred),4))
precision.append(round(precision_score(y_test, y_pred),4))

model_names = ['LightGBM_random0']
result_df = pd.DataFrame({'Accuracy':accuracy,'Recall':recall, 'Roc_Auc':roc_auc, 'Precision':precision}, index=model_names)

print(result_df)

explainerlgbmc = shap.TreeExplainer(lgbcls)

shap_values_LightGBM_test = explainerlgbmc.shap_values(X_test)

shap_values_LightGBM_train = explainerlgbmc.shap_values(X_train)

# shap.summary_plot(shap_values_LightGBM_train[1], X_train, plot_type="bar", max_display=30, plot_size=(12,12))
# shap.summary_plot(shap_values_LightGBM_train[1], X_train, plot_type="dot", max_display=30, plot_size=(12,12), cmap=plt.get_cmap('cool'))
shap.summary_plot(shap_values_LightGBM_train[1], X_train, plot_type="bar", max_display=30, plot_size=(12,12), cmap=plt.get_cmap('cool'))

plt.show()

# shap.summary_plot(shap_values_LightGBM_train[1], X_train, plot_type="bar", max_display=30, plot_size=(12,12))

# shap.summary_plot(shap_values_LightGBM_train[1], X_train, plot_type="dot", max_display=30, plot_size=(12,12))

# shap.summary_plot(shap_values_LightGBM_train[1], X_train, plot_type="dot", plot_size=(12,12))

# shap.decision_plot(explainerlgbmc.expected_value[1], shap_values_LightGBM_test[1][7], X_test.iloc[[7]],link= "logit")


# waterfall
# shap.plots._waterfall.waterfall_legacy(explainerlgbmc.expected_value[1], shap_values_LightGBM_test[1][7], feature_names = X_test.columns, max_display = 30)
# shap.plots._waterfall.waterfall_legacy(explainerlgbmc.expected_value[1],shap_values_LightGBM_test[1][7], feature_names = X_test.columns, max_display = 20)
# shap.plots._waterfall.waterfall_legacy(explainerlgbmc.expected_value[1],shap_values_LightGBM_test[1][6], feature_names = X_test.columns, max_display = 20)

shap.dependence_plot("Unique_Received_From_Addresses", shap_values_LightGBM_test[1], X_test,  interaction_index=None)

# shap.dependence_plot("min_val_sent", shap_values_LightGBM_test[1], X_test,  interaction_index=None)
# shap.dependence_plot("_Total_ERC20_tnxs", shap_values_LightGBM_test[1], X_test,  interaction_index=None)
#
# shap.dependence_plot("_Total_ERC20_tnxs", shap_values_LightGBM_test[1], X_test, interaction_index='min_val_sent', cmap=plt.get_cmap('Spectral'))
# shap.dependence_plot("Unique_Received_From_Addresses", shap_values_LightGBM_test[1], X_test, interaction_index="_Total_ERC20_tnxs", cmap=plt.get_cmap('autumn'))

shap.decision_plot(explainerlgbmc.expected_value[1], shap_values_LightGBM_test[1][:100], X_test.iloc[:100], auto_size_plot=False, link= "logit")
plt.show()