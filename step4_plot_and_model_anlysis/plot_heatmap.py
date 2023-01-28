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

data = pd.read_csv("../data/01_preprocessed_dataset_not_dummy.csv")
# print(data.columns)
feature_cols = [col for col in data.columns if col not in ['Address', 'FLAG']]
X = data[feature_cols]
y = data['FLAG']

# %config InlineBackend.figure_format = 'svg'
# %config InlineBackend.figure_format = 'retina'

corrdata = data.corr()

corrdata.dropna(axis=1,how='all', inplace=True)
corrdata.dropna(axis=0,how='all', inplace=True)

print(corrdata['FLAG'].sort_values())

mask=np.triu(np.ones_like(corrdata, dtype=np.bool))
sns.set(font_scale=0.7)
sns.heatmap(corrdata, mask=mask, linewidths=0.5, cmap=sns.color_palette('RdBu_r', n_colors=128))
plt.show()
# 相关性热力图



