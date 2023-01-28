import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

dataorg = pd.read_csv("../data/Ethereum_transactional_dataset_including_ERC20.csv", sep=",")

# data = pd.read_csv("../data/01_preprocessed_dataset_not_dummy.csv")
data = pd.read_csv("../data/02_feature_selected_XGB.csv")

data.rename(columns ={'total_transactions_(including_tnx_to_create_contract':'total_transactions'}, inplace=True)


# print(data.columns)
feature_cols = [col for col in data.columns if col not in ['Address', 'FLAG']]
X = data[feature_cols]
y = data['FLAG']

sns.set_style('ticks')
# sns.set(font_scale=1.25)
# %matplotlib inline

fea_cols =[col for col in data.columns if col not in ['Address', 'FLAG','ERC20_most_sent_token_type', 'ERC20_most_rec_token_type']]
# sns.set_theme(style="white")



def my_plot_dist(feature_name, bin_num=60):
    g = sns.displot(data=data, x=feature_name, bins=bin_num, hue='FLAG', palette='bright')
    g.set(yscale='log')
    # g.set_xticklabels(fontsize=12)
    # g.set_yticklabels(fontsize=12)
    g.set_xlabels(fontsize=14)
    g.set_ylabels(fontsize=14)
    plt.legend(labels =['fraud (label=1)', 'normal (label=0)'], title=None, loc='best', fontsize=10)

#
# Total_ERC20_tnxs
# my_plot_dist('Total_ERC20_tnxs', bin_num=30)

# ERC20_total_ether_sent
# my_plot_dist('ERC20_total_ether_sent',bin_num=50)

# ERC20_max_val_sent
# my_plot_dist('ERC20_max_val_sent',bin_num=50)

# ERC20_uniq_rec_contract_addr
# my_plot_dist('ERC20_uniq_rec_contract_addr',bin_num=50)


# Time_Diff_between_first_and_last_(Mins)
# my_plot_dist('Time_Diff_between_first_and_last_(Mins)',bin_num=35)


# Unique_Received_From_Addresses
# my_plot_dist('Unique_Received_From_Addresses',bin_num=50)

# avg_val_received
# my_plot_dist('avg_val_received', bin_num=50)

# min_value_received
# my_plot_dist('min_value_received', bin_num=50)

# total_transactions
# my_plot_dist('total_transactions', bin_num=50)

# Avg_min_between_received_tnx
# my_plot_dist('Avg_min_between_received_tnx', bin_num=35)

# ERC20_uniq_sent_token_name
# my_plot_dist('ERC20_uniq_sent_token_name', bin_num=35)

# total_transactions
# my_plot_dist('total_transactions', bin_num=35)

# Sent_tnx
# my_plot_dist('Sent_tnx', bin_num=35)

# Avg_min_between_sent_tnx
my_plot_dist('Avg_min_between_sent_tnx', bin_num=35)

# ERC20_min_val_rec
# data = data[data['ERC20_min_val_rec']<0.0005]
# my_plot_dist('ERC20_min_val_rec', bin_num=25)


# ERC20_most_sent_token_type_
# my_plot_dist('ERC20_most_sent_token_type_ ', bin_num=20)
plt.show()