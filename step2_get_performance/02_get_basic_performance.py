import pandas as pd
import sys
sys.path.append('../utils_lb')

from train_model_utils import get_all_classfiers, cv_bin_classifiy_performance_with_std, get_basic_classfiers

data = pd.read_csv("../data/01_preprocessed_dataset.csv")

feature_cols = [col for col in data.columns if col not in ['Address', 'FLAG']]

X = data[feature_cols]
y = data['FLAG']

cv_bin_classifiy_performance_with_std(X.values, y.values, classfiers=get_basic_classfiers(), random_seed=4, outputName='best_random4_')

