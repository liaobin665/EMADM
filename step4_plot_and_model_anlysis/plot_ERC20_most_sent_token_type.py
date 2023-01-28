import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

dataset = pd.read_csv("../data/Ethereum_transactional_dataset_including_ERC20.csv", sep=",")
# c = Counter(dataset['FLAG'])
# print(c[1]/dataset.shape[0])
c = Counter(dataset[' ERC20 most sent token type'])

# temp = pd.value_counts(dataset[' ERC20 most sent token type']).plot()

# sns.countplot(data = c,alpha=0.8)
sns.set_style('ticks')
# sns.set(font_scale=1.2)

temp = pd.DataFrame(dataset[' ERC20 most sent token type'].value_counts())

sent_type_df = temp[temp[' ERC20 most sent token type']>10].reset_index()

g = sns.barplot(x=sent_type_df['index'], y=sent_type_df[' ERC20 most sent token type'])

g.set(yscale='log')
g.set_xticklabels(g.get_xticklabels(),rotation=90)

plt.ylabel("Count for ERC20 most sent token type (log)")
plt.show()