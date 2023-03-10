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
import seaborn as sns
sns.set_theme(style="white")
data = pd.read_csv("../data/02_feature_selected_XGB.csv")

feature_cols = [col for col in data.columns if col not in ['Address', 'FLAG']]

X = data[feature_cols]
y = data['FLAG']
# 特征标准化
scaler = StandardScaler()

X = data[feature_cols] = scaler.fit_transform(X)

# Visualize the dataset
# projection = KernelPCA(n_components=2).fit(X, y)
projection = TSNE(n_components=3).fit_transform(X)

ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
ax.set_title('3-dimensional t-SNE satter',fontsize=11)  # 设置本图名称

# ax.scatter(projection[:, 0], projection[:, 1], projection[:, 2], c=y, cmap='rainbow', marker ='.')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
ss = ax.scatter(projection[:, 0], projection[:, 1], projection[:, 2], c=y, cmap=plt.cm.RdYlBu, marker ='.')  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色

handles, labels = ss.legend_elements(prop="colors")
labels=["normal", "fraud"]
legend = ax.legend(handles, labels, loc="best", title="", fontsize=10)


plt.axis([-26, 26, -26, 26])
ax.set_xlabel('t-SNE dim X', fontsize=10)  # 设置x坐标轴
ax.set_ylabel('t-SNE dim Y', fontsize=10)  # 设置y坐标轴
ax.set_zlabel('t-SNE dim Z', fontsize=10)  # 设置z坐标轴

# plt.legend(labels=['normal', 'fraud'], loc='best')

plt.show()

