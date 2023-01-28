import pandas as pd
from sklearn.impute import KNNImputer


dataset = pd.read_csv("../data/Ethereum_transactional_dataset_including_ERC20.csv", sep=",")
dataset = dataset.iloc[:, 2:]
dataset.columns = dataset.columns.str.strip()
dataset.columns = dataset.columns.str.replace(' ', '_')
# 存储数据集的数值型连续特征的 统计信息
# pd.DataFrame(dataset.describe().T).to_excel("number_feature_describe.xlsx")


null_fea_names =['Total_ERC20_tnxs', 'ERC20_total_Ether_received', 'ERC20_total_ether_sent', 'ERC20_total_Ether_sent_contract', 'ERC20_uniq_sent_addr', 'ERC20_uniq_rec_addr',
                 'ERC20_uniq_sent_addr.1', 'ERC20_uniq_rec_contract_addr', 'ERC20_avg_time_between_sent_tnx', 'ERC20_avg_time_between_rec_tnx', 'ERC20_avg_time_between_rec_2_tnx',
                 'ERC20_avg_time_between_contract_tnx', 'ERC20_min_val_rec', 'ERC20_max_val_rec', 'ERC20_avg_val_rec', 'ERC20_min_val_sent', 'ERC20_max_val_sent', 'ERC20_avg_val_sent',
                 'ERC20_min_val_sent_contract', 'ERC20_max_val_sent_contract', 'ERC20_avg_val_sent_contract', 'ERC20_uniq_sent_token_name', 'ERC20_uniq_rec_token_name']

print(dataset.isna().sum())
dataset[null_fea_names] = KNNImputer(n_neighbors=3).fit_transform(dataset[null_fea_names])

# dataset.dropna(how='any', axis=0, inplace=True)
print(dataset.isna().sum())

# get_dummies
# dataset = pd.get_dummies(dataset,columns=['ERC20_most_sent_token_type', 'ERC20_most_rec_token_type'], dummy_na=True)
dataset.to_csv('../data/01_preprocessed_dataset_not_dummy.csv',index=0)

# 特征列
# dataset.to_csv('../data/01_preprocessed_dataset.csv', index=0)



print("work down!")
