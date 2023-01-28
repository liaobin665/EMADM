'''
author:lb
function: 分类模型的utils, 功能包括：返回常用的分类器，将传入的数据运行，并将结果输出为excel格式，方便对比各模型之间的性能差异。
版本日期：2022-10-23
'''
import datetime
import pandas as pd
import numpy as np
'''matthews_corrcoef MCC  马修斯相关系数（Matthews correlation coefficient）'''
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split

# 分类器汇总
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn import over_sampling, under_sampling, combine
# from deepforest.cascade import CascadeForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from imbalanced_ensemble.ensemble.under_sampling import BalanceCascadeClassifier, UnderBaggingClassifier
from imbalanced_ensemble.ensemble.over_sampling import OverBoostClassifier, SMOTEBoostClassifier, KmeansSMOTEBoostClassifier, OverBaggingClassifier,SMOTEBaggingClassifier
from imbalanced_ensemble.ensemble.reweighting import AdaCostClassifier,AdaUBoostClassifier,AsymBoostClassifier
from imbalanced_ensemble.ensemble.compatible import CompatibleAdaBoostClassifier, CompatibleBaggingClassifier
from self_paced_ensemble import SelfPacedEnsembleClassifier

datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

def get_tree_classfiers(random_seed=0):
    """返回常用的树分类器"""
    classfiers = {
                  'DecisionTreeClassifier':DecisionTreeClassifier(),
                  'RandomForestClassifier': RandomForestClassifier(), 'ExtraTreesClassifier': ExtraTreesClassifier(),
                  'GradientBoostingClassifier': GradientBoostingClassifier(), 'AdaBoostClassifier': AdaBoostClassifier(),
                  'HistGradientBoostingClassifier': HistGradientBoostingClassifier(),
                  'LGBMClassifier': LGBMClassifier(), 'XGBClassifier': XGBClassifier(),
                  'CatBoostClassifier': CatBoostClassifier()
                  }
    return classfiers

def get_simple_classfiers(random_seed=0):
    """返回常用的简单分类器"""
    classfiers = {'LogisticRegression': LogisticRegression(), 'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),
                  'SGDClassifier':SGDClassifier(),'LinearSVC':LinearSVC(),'SVC':SVC(),'KNeighborsClassifier':KNeighborsClassifier(),
                  'GaussianNB':GaussianNB(),'BernoulliNB':BernoulliNB(),
                  'DecisionTreeClassifier':DecisionTreeClassifier(),'ExtraTreeClassifier':ExtraTreeClassifier(),
                  'MLPClassifier':MLPClassifier()
                  }
    return classfiers

def get_basic_classfiers(random_seed=0):
    """返回常用的分类器，在simple的基础上，添加了常用的集成模型"""
    classfiers = {'LogisticRegression': LogisticRegression(), 'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),
                  'SGDClassifier':SGDClassifier(),'LinearSVC':LinearSVC(),'SVC':SVC(),'KNeighborsClassifier':KNeighborsClassifier(),
                  'GaussianNB':GaussianNB(),'BernoulliNB':BernoulliNB(),
                  'DecisionTreeClassifier':DecisionTreeClassifier(),'ExtraTreeClassifier':ExtraTreeClassifier(),
                  'MLPClassifier':MLPClassifier(),'RandomForestClassifier':RandomForestClassifier(),'ExtraTreesClassifier':ExtraTreesClassifier(),
                  'GradientBoostingClassifier':GradientBoostingClassifier(),'AdaBoostClassifier':AdaBoostClassifier(),
                  'HistGradientBoostingClassifier':HistGradientBoostingClassifier(),
                  'RidgeClassifier': RidgeClassifier(), 'LGBMClassifier': LGBMClassifier(), 'XGBClassifier': XGBClassifier(),
                  'CatBoostClassifier': CatBoostClassifier()
                  }
    return classfiers

def get_imblearn_classfiers(random_seed=0):
    """返回所有不均衡集成模型的分类器"""
    classfiers = { # imbalanced-ensemble classifier
                  # Resampling-based: Under-sampling + Ensemble
                  'SelfPacedEnsembleClassifier': SelfPacedEnsembleClassifier(base_estimator=DecisionTreeClassifier()),
                  'BalanceCascadeClassifier': BalanceCascadeClassifier(random_state=random_seed),
                  'BalancedRandomForestClassifier': BalancedRandomForestClassifier(random_state=random_seed),
                  'EasyEnsembleClassifier': EasyEnsembleClassifier(random_state=random_seed),
                  'RUSBoostClassifier': RUSBoostClassifier(random_state=random_seed),
                  'UnderBaggingClassifier':UnderBaggingClassifier(random_state=random_seed),
                  # Resampling-based: Over-sampling + Ensemble
                  'OverBoostClassifier': OverBoostClassifier(random_state=random_seed),
                  'SMOTEBoostClassifier': SMOTEBoostClassifier(random_state=random_seed),
                  # 'KmeansSMOTEBoostClassifier': KmeansSMOTEBoostClassifier(random_state=random_seed),
                  'OverBaggingClassifier': OverBaggingClassifier(random_state=random_seed),
                  'SMOTEBaggingClassifier': SMOTEBaggingClassifier(random_state=random_seed),
                  'BalancedBaggingClassifier': BalancedBaggingClassifier(random_state=random_seed),
                  # Reweighting-based: Cost-sensitive Learning
                  'AdaCostClassifier': AdaCostClassifier(random_state=random_seed),
                  'AdaUBoostClassifier': AdaUBoostClassifier(random_state=random_seed),
                  'AsymBoostClassifier': AsymBoostClassifier(random_state=random_seed),
                  # Compatible
                  'CompatibleAdaBoostClassifier': CompatibleAdaBoostClassifier(random_state=random_seed),
                  'CompatibleBaggingClassifier': CompatibleBaggingClassifier(random_state=random_seed),
                  }

def get_all_classfiers(random_seed=0):
    """返回所有的分类器"""
    classfiers = {'LogisticRegression': LogisticRegression(), 'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),
                  'SGDClassifier':SGDClassifier(),'LinearSVC':LinearSVC(),'SVC':SVC(),'KNeighborsClassifier':KNeighborsClassifier(),
                  'GaussianNB':GaussianNB(),'BernoulliNB':BernoulliNB(),
                  'DecisionTreeClassifier':DecisionTreeClassifier(),'ExtraTreeClassifier':ExtraTreeClassifier(),
                  'MLPClassifier':MLPClassifier(),'RandomForestClassifier':RandomForestClassifier(),'ExtraTreesClassifier':ExtraTreesClassifier(),
                  'GradientBoostingClassifier':GradientBoostingClassifier(),'AdaBoostClassifier':AdaBoostClassifier(),
                  'HistGradientBoostingClassifier':HistGradientBoostingClassifier(),
                  'RidgeClassifier': RidgeClassifier(), 'LGBMClassifier': LGBMClassifier(), 'XGBClassifier': XGBClassifier(),
                  'CatBoostClassifier': CatBoostClassifier(),
                  # imbalanced-ensemble classifier
                  # Resampling-based: Under-sampling + Ensemble
                  'SelfPacedEnsembleClassifier': SelfPacedEnsembleClassifier(base_estimator=DecisionTreeClassifier()),
                  'BalanceCascadeClassifier': BalanceCascadeClassifier(random_state=random_seed),
                  'BalancedRandomForestClassifier': BalancedRandomForestClassifier(random_state=random_seed),
                  'EasyEnsembleClassifier': EasyEnsembleClassifier(random_state=random_seed),
                  'RUSBoostClassifier': RUSBoostClassifier(random_state=random_seed),
                  'UnderBaggingClassifier':UnderBaggingClassifier(random_state=random_seed),
                  # Resampling-based: Over-sampling + Ensemble
                  'OverBoostClassifier': OverBoostClassifier(random_state=random_seed),
                  'SMOTEBoostClassifier': SMOTEBoostClassifier(random_state=random_seed),
                  # 'KmeansSMOTEBoostClassifier': KmeansSMOTEBoostClassifier(random_state=random_seed),
                  'OverBaggingClassifier': OverBaggingClassifier(random_state=random_seed),
                  'SMOTEBaggingClassifier': SMOTEBaggingClassifier(random_state=random_seed),
                  'BalancedBaggingClassifier': BalancedBaggingClassifier(random_state=random_seed),
                  # Reweighting-based: Cost-sensitive Learning
                  'AdaCostClassifier': AdaCostClassifier(random_state=random_seed),
                  'AdaUBoostClassifier': AdaUBoostClassifier(random_state=random_seed),
                  'AsymBoostClassifier': AsymBoostClassifier(random_state=random_seed),
                  # Compatible
                  'CompatibleAdaBoostClassifier': CompatibleAdaBoostClassifier(random_state=random_seed),
                  'CompatibleBaggingClassifier': CompatibleBaggingClassifier(random_state=random_seed),
                  }
    return classfiers


def bin_classifiy_performance(X, y, test_size=0.2, classfiers=get_simple_classfiers(), random_seed=0, outputName = ''):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, stratify=y)
    result_pd = pd.DataFrame()
    cls_nameList = []
    accuracys = []
    precisions = []
    recalls = []
    F1s = []
    AUCs = []
    MMCs = []

    for cls_name, cls in classfiers.items():
        print("start training:", cls_name)
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)
        cls_nameList.append(cls_name)
        accuracys.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        F1s.append(f1_score(y_test, y_pred))
        AUCs.append(roc_auc_score(y_test, y_pred))
        MMCs.append(matthews_corrcoef(y_test, y_pred))

    result_pd['classfier_name'] = cls_nameList
    result_pd['accuracy'] = accuracys
    result_pd['precision'] = precisions
    result_pd['recall'] = recalls
    result_pd['F1'] = F1s
    result_pd['AUC'] = AUCs
    result_pd['MMC'] = MMCs

    result_pd.to_excel(outputName + datetime_str + '.xlsx', index=True)
    print("work done!")

# resample_method 需要将指定的不平衡处理（采样）方法传入。默认为 combine.SMOTEENN
def bin_classifiy_performance_with_resample(X, y, test_size=0.2, classfiers=get_simple_classfiers(), resample_method = combine.SMOTEENN, random_seed=0, outputName = ''):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, stratify=y)
    # 样本采样，不平衡数据处理，数据增强
    X_resampled, y_resampled = resample_method(random_state=random_seed).fit_resample(X_train, y_train)
    result_pd = pd.DataFrame()
    cls_nameList = []
    accuracys = []
    precisions = []
    recalls = []
    F1s = []
    AUCs = []
    MMCs = []

    for cls_name, cls in classfiers.items():
        print("start training:", cls_name)
        cls.fit(X_resampled, y_resampled)
        y_pred = cls.predict(X_test)
        cls_nameList.append(cls_name)
        accuracys.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        F1s.append(f1_score(y_test, y_pred))
        AUCs.append(roc_auc_score(y_test, y_pred))
        MMCs.append(matthews_corrcoef(y_test, y_pred))

    result_pd['classfier_name'] = cls_nameList
    result_pd['accuracy'] = accuracys
    result_pd['precision'] = precisions
    result_pd['recall'] = recalls
    result_pd['F1'] = F1s
    result_pd['AUC'] = AUCs
    result_pd['MMC'] = MMCs

    result_pd.to_excel(outputName + datetime_str + '.xlsx', index=True)
    print("work done!")

"X y input数据；splits_num：cv的数量；classfiers：模型k-v；random_seed随机数；outputName 输出文件名"
def cv_bin_classifiy_performance(X, y, splits_num =5, classfiers = get_simple_classfiers(), random_seed=0, outputName = ''):
    """将传入的数据，分类器，进行训练，然后将性能输出到指定名称的excel文件中"""
    # 从下面开始，可以定制自己性能的输出形式
    result_pd = pd.DataFrame()
    cls_nameList = []
    accuracys = []
    precisions = []
    recalls = []
    F1s = []
    AUCs = []
    MMCs = []

    # StratifiedKFold切分数据(label均分)
    skf = StratifiedKFold(n_splits=splits_num, random_state=random_seed, shuffle=True)

    for cls_name, cls in classfiers.items():
        print("start training:", cls_name)
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_auc = 0.0
        total_mmc = 0.0
        for k, (train_index, test_index) in enumerate(skf.split(X=X, y=y)):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

            # 这里可以加入不平衡处理的代码
            # X_resampled, y_resampled = combine.SMOTEENN(random_state=random_seed).fit_resample(X_train, y_train)
            cls.fit(X_train, y_train)
            y_pred = cls.predict(X_test)
            total_accuracy += accuracy_score(y_test, y_pred)
            total_precision += precision_score(y_test, y_pred)
            total_recall += recall_score(y_test, y_pred)
            total_f1 += f1_score(y_test, y_pred)
            total_auc += roc_auc_score(y_test, y_pred)
            total_mmc += matthews_corrcoef(y_test, y_pred)

        cls_nameList.append(cls_name)
        accuracys.append(total_accuracy / splits_num)
        precisions.append(total_precision / splits_num)
        recalls.append(total_recall / splits_num)
        F1s.append(total_f1 / splits_num)
        AUCs.append(total_auc / splits_num)
        MMCs.append(total_mmc / splits_num)

    result_pd['classfier_name'] = cls_nameList
    result_pd['avg_accuracy'] = accuracys
    result_pd['avg_precision'] = precisions
    result_pd['avg_recall'] = recalls
    result_pd['avg_F1'] = F1s
    result_pd['avg_AUC'] = AUCs
    result_pd['avg_MMC'] = MMCs

    result_pd.to_excel(outputName + datetime_str + '.xlsx', index=True)
    print("work done!")

# resample_method 需要将指定的不平衡处理（采样）方法传入。默认为 combine.SMOTEENN
def cv_bin_classifiy_performance_with_resample(X, y, splits_num=5,  classfiers = get_simple_classfiers(), resample_method = combine.SMOTEENN, random_seed=0, outputName = ''):
    """将传入的数据，分类器，进行训练，然后将性能输出到指定名称的excel文件中"""
    # 从下面开始，可以定制自己性能的输出形式
    result_pd = pd.DataFrame()
    cls_nameList = []
    accuracys = []
    precisions = []
    recalls = []
    F1s = []
    AUCs = []
    MMCs = []

    # StratifiedKFold切分数据(label均分)
    skf = StratifiedKFold(n_splits=splits_num, random_state=random_seed, shuffle=True)

    for cls_name, cls in classfiers.items():
        print("start training:", cls_name)
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_auc = 0.0
        total_mmc = 0.0
        for k, (train_index, test_index) in enumerate(skf.split(X=X, y=y)):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            # 样本采样，不平衡数据处理，数据增强
            X_resampled, y_resampled = resample_method(random_state=random_seed).fit_resample(X_train, y_train)
            cls.fit(X_resampled, y_resampled)
            y_pred = cls.predict(X_test)
            total_accuracy += accuracy_score(y_test, y_pred)
            total_precision += precision_score(y_test, y_pred)
            total_recall += recall_score(y_test, y_pred)
            total_f1 += f1_score(y_test, y_pred)
            total_auc += roc_auc_score(y_test, y_pred)
            total_mmc += matthews_corrcoef(y_test, y_pred)

        cls_nameList.append(cls_name)
        accuracys.append(total_accuracy / splits_num)
        precisions.append(total_precision / splits_num)
        recalls.append(total_recall / splits_num)
        F1s.append(total_f1 / splits_num)
        AUCs.append(total_auc / splits_num)
        MMCs.append(total_mmc / splits_num)

    result_pd['classfier_name'] = cls_nameList
    result_pd['avg_accuracy'] = accuracys
    result_pd['avg_precision'] = precisions
    result_pd['avg_recall'] = recalls
    result_pd['avg_F1'] = F1s
    result_pd['avg_AUC'] = AUCs
    result_pd['avg_MMC'] = MMCs

    result_pd.to_excel(outputName + datetime_str + '.xlsx', index=True)
    print("work done!")

"X y input数据；splits_num：cv的数量；classfiers：模型k-v；random_seed随机数；outputName 输出文件名"
def cv_bin_classifiy_performance_with_std(X, y, splits_num =5, classfiers = get_simple_classfiers(), random_seed=0, outputName = ''):
    """将传入的数据，分类器，进行训练，然后将性能输出到指定名称的excel文件中"""
    # 从下面开始，可以定制自己性能的输出形式
    result_pd = pd.DataFrame()
    cls_nameList, accuracys, acc_std, precisions, precision_std, recalls, recall_std, F1s, f1_std, AUCs, auc_std, MMCs, mmc_std \
        = [], [], [], [], [], [], [], [], [], [], [], [], []
    # StratifiedKFold切分数据(label均分)
    skf = StratifiedKFold(n_splits=splits_num, random_state=random_seed, shuffle=True)

    for cls_name, cls in classfiers.items():
        print("start training:", cls_name)
        total_accuracy = 0.0
        accuracy_list = []
        total_precision = 0.0
        precision_list = []
        total_recall = 0.0
        recall_list = []
        total_f1 = 0.0
        f1_list = []
        total_auc = 0.0
        auc_list = []
        total_mmc = 0.0
        mmc_list = []

        for k, (train_index, test_index) in enumerate(skf.split(X=X, y=y)):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            # 这里可以加入不平衡处理的代码
            # X_resampled, y_resampled = combine.SMOTEENN(random_state=random_seed).fit_resample(X_train, y_train)
            cls.fit(X_train, y_train)
            y_pred = cls.predict(X_test)

            total_accuracy += accuracy_score(y_test, y_pred)
            accuracy_list.append(accuracy_score(y_test, y_pred))
            total_precision += precision_score(y_test, y_pred)
            precision_list.append(recall_score(y_test, y_pred))
            total_recall += recall_score(y_test, y_pred)
            recall_list.append(recall_score(y_test, y_pred))
            total_f1 += f1_score(y_test, y_pred)
            f1_list.append(f1_score(y_test, y_pred))
            total_auc += roc_auc_score(y_test, y_pred)
            auc_list.append(roc_auc_score(y_test, y_pred))
            total_mmc += matthews_corrcoef(y_test, y_pred)
            mmc_list.append(matthews_corrcoef(y_test, y_pred))

        cls_nameList.append(cls_name)
        accuracys.append(round(total_accuracy / splits_num, 4))
        acc_std.append(round(np.std(auc_list), 4))
        precisions.append(round(total_precision / splits_num, 4))
        precision_std.append(round(np.std(precision_list), 4))
        recalls.append(round(total_recall / splits_num, 4))
        recall_std.append(round(np.std(recall_list), 4))
        F1s.append(round(total_f1 / splits_num, 4))
        f1_std.append(round(np.std(f1_list), 4))
        AUCs.append(round(total_auc / splits_num, 4))
        auc_std.append(round(np.std(auc_list), 4))
        MMCs.append(round(total_mmc / splits_num, 4))
        mmc_std.append(round(np.std(mmc_list), 4))

    result_pd['classfier_name'] = cls_nameList
    result_pd['avg_accuracy'] = accuracys
    result_pd['std_accuracy_std'] = acc_std
    result_pd['avg_precision'] = precisions
    result_pd['std_precision'] = precision_std
    result_pd['avg_recall'] = recalls
    result_pd['std_recall'] = recall_std
    result_pd['avg_F1'] = F1s
    result_pd['std_F1'] = f1_std
    result_pd['avg_AUC'] = AUCs
    result_pd['std_AUC'] = auc_std
    result_pd['avg_MMC'] = MMCs
    result_pd['std_MMC'] = mmc_std
    result_pd.to_excel(outputName+datetime_str+'.xlsx', index=True)
    print("work done!")

" X y input数据；splits_num：cv的数量；classfiers：模型k-v；random_seed随机数；outputName 输出文件名"
def cv_mutil_classifiy_performance(X, y, splits_num=5,  classfiers = get_simple_classfiers(), random_seed=0, outputName = ''):
    # 从下面开始，可以定制自己性能的输出形式
    result_pd = pd.DataFrame()
    cls_nameList = []
    accuracys = []
    precisions = []
    recalls = []
    F1s = []
    macro_precisions = []
    macro_recalls = []
    macro_F1s = []
    AUCs = []
    MMCs = []
    # StratifiedKFold切分数据(label均分)
    skf = StratifiedKFold(n_splits=splits_num, random_state=random_seed, shuffle=True)
    for cls_name, cls in classfiers.items():
        print("start training:", cls_name)
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_auc = 0.0
        total_mmc = 0.0
        total_macro_precision = 0.0
        total_macro_recall = 0.0
        total_macro_f1 = 0.0
        for k, (train_index, test_index) in enumerate(skf.split(X=X, y=y)):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            # 这里可以加入不平衡处理的代码
            # X_resampled, y_resampled = combine.SMOTEENN(random_state=random_seed).fit_resample(X_train, y_train)
            cls.fit(X_train, y_train)
            y_pred = cls.predict(X_test)
            total_accuracy += accuracy_score(y_test, y_pred)
            total_precision += precision_score(y_test, y_pred, average='micro')
            total_recall += recall_score(y_test, y_pred, average='micro')
            total_f1 += f1_score(y_test, y_pred, average='micro')
            # total_auc += roc_auc_score(y_test, y_pred)
            total_mmc += matthews_corrcoef(y_test, y_pred)
            total_macro_precision += precision_score(y_test, y_pred, average='macro')
            total_macro_recall += recall_score(y_test, y_pred, average='macro')
            total_macro_f1 += f1_score(y_test, y_pred, average='macro')
        cls_nameList.append(cls_name)
        accuracys.append(total_accuracy / splits_num)
        precisions.append(total_precision / splits_num)
        recalls.append(total_recall / splits_num)
        F1s.append(total_f1 / splits_num)
        macro_precisions.append(total_macro_precision / splits_num)
        macro_recalls.append(total_macro_recall / splits_num)
        macro_F1s.append(total_macro_f1 / splits_num)
        # AUCs.append(total_auc/splits_num)
        MMCs.append(total_mmc / splits_num)
    result_pd['classfier_name'] = cls_nameList
    result_pd['accuracy'] = accuracys
    result_pd['precision'] = precisions
    result_pd['recall'] = recalls
    result_pd['F1'] = F1s
    result_pd['macro_precision'] = macro_precisions
    result_pd['macro_recall'] = macro_recalls
    result_pd['macro_F1'] = macro_F1s
    # result_pd['AUC'] = AUCs
    result_pd['MMC'] = MMCs
    result_pd.to_excel(outputName+datetime_str+'.xlsx', index=True)

"X y input数据；splits_num：cv的数量；classfiers：模型k-v；random_seed随机数；outputName 输出文件名"
def cv_bin_classifiy_performance_with_std_resample(X, y, splits_num =5,  classfiers = get_simple_classfiers(), resample_method = combine.SMOTEENN, random_seed=0, outputName = ''):
    """将传入的数据，分类器，进行训练，然后将性能输出到指定名称的excel文件中"""
    # 从下面开始，可以定制自己性能的输出形式
    result_pd = pd.DataFrame()
    cls_nameList, accuracys, acc_std, precisions, precision_std, recalls, recall_std, F1s, f1_std, AUCs, auc_std, MMCs, mmc_std \
        = [], [], [], [], [], [], [], [], [], [], [], [], []
    # StratifiedKFold切分数据(label均分)
    skf = StratifiedKFold(n_splits=splits_num, random_state=random_seed, shuffle=True)

    for cls_name, cls in classfiers.items():
        print("start training:", cls_name)
        total_accuracy = 0.0
        accuracy_list = []
        total_precision = 0.0
        precision_list = []
        total_recall = 0.0
        recall_list = []
        total_f1 = 0.0
        f1_list = []
        total_auc = 0.0
        auc_list = []
        total_mmc = 0.0
        mmc_list = []

        for k, (train_index, test_index) in enumerate(skf.split(X=X, y=y)):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            # 样本采样，不平衡数据处理，数据增强
            X_resampled, y_resampled = resample_method(random_state=random_seed).fit_resample(X_train, y_train)
            cls.fit(X_resampled, y_resampled)
            y_pred = cls.predict(X_test)

            total_accuracy += accuracy_score(y_test, y_pred)
            accuracy_list.append(accuracy_score(y_test, y_pred))
            total_precision += precision_score(y_test, y_pred)
            precision_list.append(recall_score(y_test, y_pred))
            total_recall += recall_score(y_test, y_pred)
            recall_list.append(recall_score(y_test, y_pred))
            total_f1 += f1_score(y_test, y_pred)
            f1_list.append(f1_score(y_test, y_pred))
            total_auc += roc_auc_score(y_test, y_pred)
            auc_list.append(roc_auc_score(y_test, y_pred))
            total_mmc += matthews_corrcoef(y_test, y_pred)
            mmc_list.append(matthews_corrcoef(y_test, y_pred))

        cls_nameList.append(cls_name)
        accuracys.append(round(total_accuracy / splits_num, 4))
        acc_std.append(round(np.std(auc_list), 4))
        precisions.append(round(total_precision / splits_num, 4))
        precision_std.append(round(np.std(precision_list), 4))
        recalls.append(round(total_recall / splits_num, 4))
        recall_std.append(round(np.std(recall_list), 4))
        F1s.append(round(total_f1 / splits_num, 4))
        f1_std.append(round(np.std(f1_list), 4))
        AUCs.append(round(total_auc / splits_num, 4))
        auc_std.append(round(np.std(auc_list), 4))
        MMCs.append(round(total_mmc / splits_num, 4))
        mmc_std.append(round(np.std(mmc_list), 4))

    result_pd['classfier_name'] = cls_nameList
    result_pd['avg_accuracy'] = accuracys
    result_pd['std_accuracy_std'] = acc_std
    result_pd['avg_precision'] = precisions
    result_pd['std_precision'] = precision_std
    result_pd['avg_recall'] = recalls
    result_pd['std_recall'] = recall_std
    result_pd['avg_F1'] = F1s
    result_pd['std_F1'] = f1_std
    result_pd['avg_AUC'] = AUCs
    result_pd['std_AUC'] = auc_std
    result_pd['avg_MMC'] = MMCs
    result_pd['std_MMC'] = mmc_std
    result_pd.to_excel(outputName+datetime_str+'.xlsx', index=True)
    print("work done!")
