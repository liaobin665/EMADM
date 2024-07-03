# 这里可以将 特征选择 前后的 2D 映射 作为对比。
from self_paced_ensemble.utils._plot import plot_2Dprojection_and_cardinality

from sklearn.linear_model import LogisticRegression
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA,PCA
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE  # TSNE集成在了sklearn中

data = pd.read_csv("../data/02_feature_selected_XGB.csv")

feature_cols = [col for col in data.columns if col not in ['Address', 'FLAG']]


X = data[feature_cols]
y = data['FLAG']
# 特征标准化
scaler = StandardScaler()

X = data[feature_cols] = scaler.fit_transform(X)

# Visualize the dataset
# projection = KernelPCA(n_components=2).fit(X, y)
projection = PCA(n_components=2).fit(X)
fig = plot_2Dprojection_and_cardinality(X, y, projection=projection)
plt.show()
