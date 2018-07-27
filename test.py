# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import gc
import time
import seaborn as sns
import matplotlib.pyplot as plt
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import logging
import logging.handlers

LOG_FILE = 'tst.log'

handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=5)  # 实例化handler
fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'

formatter = logging.Formatter(fmt)  # 实例化formatter
handler.setFormatter(formatter)  # 为handler添加formatter

logger = logging.getLogger('tst')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.DEBUG)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, lgb_param, stratified=False, debug=False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=50)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=50)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(**lgb_param)

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=100, early_stopping_rounds=200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        logging.info('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
    full_auc = roc_auc_score(train_df['TARGET'], oof_preds)
    print('Full AUC score %.6f' % full_auc)
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index=False)
    return full_auc


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[
           :40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


def main(debug=False):
    df = pd.read_csv('./data/lightGBM.csv', encoding='utf-8')
    param = {
        'num_leaves': range(25, 100, 2),
        'colsample_bytree': [i / 1000 for i in range(850, 1000, 10)],
        'subsample': [i / 1000 for i in range(850, 1000, 10)],
        'max_depth': range(5, 30, 5),
        'reg_alpha': [i / 100 for i in range(4, 20, 2)],
        'reg_lambda': [i / 1000 for i in range(85, 100, 2)],
        'min_split_gain': [i / 1000 for i in range(20, 50, 5)],
        'min_child_weight': range(20, 100, 10),
    }
    lgb_param = {
        'nthread': 10,
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'num_leaves': 32,
        'colsample_bytree': 0.9497036,
        'subsample': 0.8715623,
        'max_depth': 8,
        'reg_alpha': 0.04,
        'reg_lambda': 0.073,
        'min_split_gain': 0.0222415,
        'min_child_weight': 40,
        'silent': -1,
        'verbose': -1,
    }
    fit_param = lgb_param
    full_auc = kfold_lightgbm(df, num_folds=2, lgb_param=lgb_param, stratified=False, debug=debug)
    for key, value in param.items():
        num = 0
        for i in value:
            print('{}: {}'.format(key, i))
            lgb_param[key] = i
            new_full_auc = kfold_lightgbm(df, num_folds=2, lgb_param=lgb_param, stratified=False, debug=debug)
            if new_full_auc < full_auc:
                num += 1
                continue
            else:
                fit_param[key] = i
            if num > 5:
                break
        lgb_param[key] = fit_param[key]
        logging.info('param:{}'.format(fit_param))


if __name__ == "__main__":
    submission_file_name = "submission_kernel.csv"
    with timer("Full model run"):
        main(debug=True)
