import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

dataorg = pd.read_csv("../data/Ethereum_transactional_dataset_including_ERC20.csv", sep=",")

data = pd.read_csv("../data/02_feature_selected_XGB.csv")

sns.displot(data,hue='ERC20_most_rec_token_type_Blockwell say NOTSAFU',x='FLAG')
plt.show()

print()