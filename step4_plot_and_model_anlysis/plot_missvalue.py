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

data = pd.read_csv("../data/Ethereum_transactional_dataset_including_ERC20.csv")
## 1) 查看每列的存在nan情况
print(data.isnull().sum())

# nan可视化
missing = data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()

# 可视化看下缺省值
msno.matrix(data.sample(2500))
msno.dendrogram(data.sample(2500))

plt.show()

