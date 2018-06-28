# coding: utf-8
# https://segmentfault.com/a/1190000009101577#articleHeader6
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import Imputer
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle
from scipy import interp
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import pickle
import time

warnings.filterwarnings('ignore')



class Train:
    def __init__(self):
        self.train_full = pd.DataFrame()
        self.test_full = pd.DataFrame()
        self.ytrain = pd.Series()
        self.dct_scores = {}
        self.mean_score = {}
        self.mean_time = {}

    def get_data(self):
        # Training data
        app_train = pd.read_csv('./Downloads/application_train.csv')
        # Testing data features
        app_test = pd.read_csv('./Downloads/application_test.csv')

        def missing_values_table(df):
            # Total missing values
            mis_val = df.isnull().sum()

            # Percentage of missing values
            mis_val_percent = 100 * df.isnull().sum() / len(df)

            # Make a table with the results
            mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

            # Rename the columns
            mis_val_table_ren_columns = mis_val_table.rename(
                columns={0: 'Missing Values', 1: '% of Total Values'})
            # Sort the table by percentage of missing descending
            mis_val_table_ren_columns = mis_val_table_ren_columns[
                mis_val_table_ren_columns.iloc[:, 1] >= 30].sort_values(
                '% of Total Values', ascending=False).round(1)
            # Print some summary information
            print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                                      "There are " + str(
                mis_val_table_ren_columns.shape[0]) +
                  " columns that have missing values.")

            # Return the dataframe with missing information
            return mis_val_table_ren_columns

        # Missing values statistics
        missing_values = missing_values_table(app_train)
        app_train = app_train.drop(list(missing_values.index), axis=1)
        train_labels = app_train['TARGET']

        # Align the training and testing data, keep only columns present in both dataframes
        app_train, app_test = app_train.align(app_test, join='inner', axis=1)
        app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)

        # Create a label encoder object
        le = LabelEncoder()
        le_count = 0

        # Iterate through the columns
        for col in app_train:
            if app_train[col].dtype == 'object':
                # If 2 or fewer unique categories
                if len(list(app_train[col].unique())) <= 2:
                    # Train on the training data
                    le.fit(app_train[col])
                    # Transform both training and testing data
                    app_train[col] = le.transform(app_train[col])
                    app_test[col] = le.transform(app_test[col])

                    # Keep track of how many columns were label encoded
                    le_count += 1

        print('%d columns were label encoded.' % le_count)
        app_train = pd.get_dummies(app_train)
        app_test = pd.get_dummies(app_test)

        app_train['TARGET'] = train_labels
        mean_pred = np.mean(app_train)
        app_train.fillna(mean_pred, inplace=True)
        mean_pred = np.mean(app_test)
        app_test.fillna(mean_pred, inplace=True)



    def process_data(self):
        # Training data
        app_train = pd.read_csv('./Downloads/application_train.csv')
        # Testing data features
        app_test = pd.read_csv('./Downloads/application_test.csv')
        # Number of unique classes in each object column
        app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)

        # Create a label encoder object
        le = LabelEncoder()
        le_count = 0

        # Iterate through the columns
        for col in app_train:
            if app_train[col].dtype == 'object':
                # If 2 or fewer unique categories
                if len(list(app_train[col].unique())) <= 2:
                    # Train on the training data
                    le.fit(app_train[col])
                    # Transform both training and testing data
                    app_train[col] = le.transform(app_train[col])
                    app_test[col] = le.transform(app_test[col])

                    # Keep track of how many columns were label encoded
                    le_count += 1

        print('%d columns were label encoded.' % le_count)

        '''
        离散特征的编码分为两种情况：

        1、离散特征的取值之间没有大小的意义，比如color：[red,blue],那么就使用one-hot编码

        2、离散特征的取值有大小的意义，比如size:[X,XL,XXL],那么就使用数值的映射{X:1,XL:2,XXL:3}

        使用get_dummies可以很方便的对离散型特征进行one-hot编码

        '''

        # one-hot encoding of categorical variables
        app_train = pd.get_dummies(app_train)
        app_test = pd.get_dummies(app_test)

        print('Training Features shape: ', app_train.shape)
        print('Testing Features shape: ', app_test.shape)

        # ### Aligning Training and Testing Data

        train_labels = app_train['TARGET']

        # Align the training and testing data, keep only columns present in both dataframes
        app_train, app_test = app_train.align(app_test, join='inner', axis=1)

        # Add the target back in
        app_train['TARGET'] = train_labels

        print('Training Features shape: ', app_train.shape)
        print('Testing Features shape: ', app_test.shape)

        anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]
        non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]
        print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
        print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
        print('There are %d anomalous days of employment' % len(anom))

        # Create an anomalous flag column
        app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

        # Replace the anomalous values with nan
        app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

        app_train['DAYS_EMPLOYED'].plot.hist(title='Days Employment Histogram')

        app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
        app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

        print(
            'There are %d anomalies in the test data out of %d entries' % (
                app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))

        correlations = app_train.corr()['TARGET'].sort_values()

        # Display correlations
        print('Most Positive Correlations:\n', correlations.tail(15))
        print('\nMost Negative Correlations:\n', correlations.head(15))

        # Find the correlation of the positive days since birth and target
        app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])

        app_train.to_csv('./data/app_train.csv', index=False)
        app_test.to_csv('./data/app_test.csv', index=False)

        # Make a new dataframe for polynomial features
        poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
        poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]

        imputer = Imputer(strategy='median')

        poly_target = poly_features['TARGET']

        poly_features = poly_features.drop(columns=['TARGET'])

        # Need to impute missing values
        poly_features = imputer.fit_transform(poly_features)
        poly_features_test = imputer.transform(poly_features_test)

        # Create the polynomial object with specified degree
        poly_transformer = PolynomialFeatures(degree=3)

        # Train the polynomial features
        poly_transformer.fit(poly_features)

        # Transform the features
        poly_features = poly_transformer.transform(poly_features)
        poly_features_test = poly_transformer.transform(poly_features_test)
        print('Polynomial Features shape: ', poly_features.shape)

        poly_features = pd.DataFrame(poly_features,
                                     columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                                 'EXT_SOURCE_3', 'DAYS_BIRTH']))

        # Add in the target
        poly_features['TARGET'] = poly_target

        # Put test features into dataframe
        poly_features_test = pd.DataFrame(poly_features_test,
                                          columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2',
                                                                                      'EXT_SOURCE_3', 'DAYS_BIRTH']))

        poly_features.to_csv('./data/poly_features.csv', index=False)
        poly_features_test.to_csv('./data/poly_features_test.csv', index=False)
        # Merge polynomial features into training dataframe
        poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
        app_train_poly = app_train.merge(poly_features, on='SK_ID_CURR', how='left')

        # Merge polnomial features into testing dataframe
        poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
        app_test_poly = app_test.merge(poly_features_test, on='SK_ID_CURR', how='left')

        # Align the dataframes
        app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join='inner', axis=1)

        app_train_poly['TARGET'] = poly_target
        # Print out the new shapes
        print('Training data with polynomial features shape: ', app_train_poly.shape)
        print('Testing data with polynomial features shape:  ', app_test_poly.shape)

        app_train_poly.to_csv('./data/app_train_poly.csv', index=False)
        app_test_poly.to_csv('./data/app_test_poly.csv', index=False)

        app_train_domain = app_train.copy()
        app_test_domain = app_test.copy()

        app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain[
            'AMT_INCOME_TOTAL']
        app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain[
            'AMT_INCOME_TOTAL']
        app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
        app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']

        app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
        app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
        app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
        app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']

        app_train_domain.to_csv('./data/app_train_domain.csv', index=False)
        app_test_domain.to_csv('./data/app_test_domain.csv', index=False)

    def kfold_plot(self, train, ytrain, model):

        #     kf = StratifiedKFold(y=ytrain, n_folds=5)
        kf = StratifiedKFold(n_splits=5)
        scores = []
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        exe_time = []

        colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue'])
        lw = 2

        i = 0
        for (train_index, test_index), color in zip(kf.split(train, ytrain), colors):
            X_train, X_test = train.iloc[train_index], train.iloc[test_index]
            y_train, y_test = ytrain.iloc[train_index], ytrain.iloc[test_index]
            begin_t = time.time()
            predictions = model(X_train, X_test, y_train)
            end_t = time.time()
            exe_time.append(round(end_t - begin_t, 3))
            scores.append(roc_auc_score(y_test.astype(float), predictions))
            fpr, tpr, thresholds = roc_curve(y_test, predictions)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
            i += 1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k', label='Luck')

        mean_tpr /= kf.get_n_splits(train, ytrain)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc='lower right')
        plt.show()

        #     print 'scores: ', scores
        print('mean scores: ', np.mean(scores))
        print('mean model process time: ', np.mean(exe_time), 's')

        return scores, np.mean(scores), np.mean(exe_time)

    def forest_model(self, X_train, X_test, y_train):
        begin_t = time.time()
        model = RandomForestClassifier(n_estimators=200, max_features=20, max_depth=8,
                                       random_state=10, n_jobs=4)
        model.fit(X_train, y_train)
        end_t = time.time()
        print('train time of forest model: ', round(end_t - begin_t, 3), 's')
        predictions = model.predict_proba(X_test)[:, 1]
        return predictions

    def gradient_model(self, X_train, X_test, y_train):
        model = GradientBoostingClassifier(n_estimators=200, random_state=7, max_depth=8, learning_rate=0.03)
        model.fit(X_train, y_train)
        predictions = model.predict_proba(X_test)[:, 1]
        return predictions

    def xgboost_model(self, X_train, X_test, y_train):
        X_train = xgb.DMatrix(X_train.values, label=y_train.values, nthread=5)
        X_test = xgb.DMatrix(X_test.values)
        params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': 1, 'seed': 7,
                  'max_depth': 8, 'eta': 0.01}
        model = xgb.train(params, X_train, 600)
        predictions = model.predict(X_test)
        return predictions

    def lightgbm_model(self, X_train, X_test, y_train):
        X_train = lgb.Dataset(X_train.values, y_train.values)
        params = {'objective': 'binary', 'metric': {'auc'}, 'learning_rate': 0.01, 'max_depth': 8, 'seed': 7}
        model = lgb.train(params, X_train, num_boost_round=600)
        predictions = model.predict(X_test)
        return predictions

    def plot_model_comp(self, title, y_label, dct_result):
        '''
        比较四个模型在交叉验证机上的roc_auc平均得分和模型训练的时间
        :param title:
        :param y_label:
        :param dct_result:
        :return:
        '''

        data_source = list(dct_result.keys())
        y_pos = np.arange(len(data_source))
        model_auc = list(dct_result.values())
        barlist = plt.bar(y_pos, model_auc, align='center', alpha=0.5)
        print(model_auc)
        max_val = max(model_auc)
        print(max_val)
        idx = model_auc.index(max_val)
        barlist[idx].set_color('r')
        plt.xticks(y_pos, data_source)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()
        print('The highest auc score is {0} of model: {1}'.format(max_val, data_source[idx]))

    def plot_time_comp(self, title, y_label, dct_result):
        data_source = list(dct_result.keys())
        y_pos = np.arange(len(data_source))
        # model_auc = [0.910, 0.912, 0.915, 0.922]
        model_auc = list(dct_result.values())
        barlist = plt.bar(y_pos, model_auc, align='center', alpha=0.5)
        # get the index of highest score
        min_val = min(model_auc)
        idx = model_auc.index(min_val)
        barlist[idx].set_color('r')
        plt.xticks(y_pos, data_source)
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()
        print('The shortest time is {0} of model: {1}'.format(min_val, data_source[idx]))

    def plot_auc_score(self, dct_scores):
        auc_forest = dct_scores['forest']
        auc_gb = dct_scores['gbm']
        auc_xgb = dct_scores['xgboost']
        auc_lgb = dct_scores['lgbm']
        print('std of forest auc score: ', np.std(auc_forest))

        print('std of gbm auc score: ', np.std(auc_gb))
        print('std of xgboost auc score: ', np.std(auc_xgb))
        print('std of lightgbm auc score: ', np.std(auc_lgb))
        data_source = ['roc-fold-1', 'roc-fold-2', 'roc-fold-3', 'roc-fold-4', 'roc-fold-5']
        y_pos = np.arange(len(data_source))
        plt.plot(y_pos, auc_forest, 'b-', label='forest')
        plt.plot(y_pos, auc_gb, 'r-', label='gbm')
        plt.plot(y_pos, auc_xgb, 'y-', label='xgboost')
        plt.plot(y_pos, auc_lgb, 'g-', label='lightgbm')
        plt.title('roc-auc score of each epoch')
        plt.xlabel('epoch')
        plt.ylabel('roc-auc score')
        plt.legend()
        plt.show()

    def choose_xgb_model(self, X_train, y_train):
        '''
        通过交叉验证针对模型选择参数组合
        :param X_train:
        :param y_train:
        :return:
        '''
        tuned_params = [{'objective': ['binary:logistic'], 'learning_rate': [0.01, 0.03, 0.05],
                         'n_estimators': [100, 150, 200], 'max_depth': [4, 6, 8]}]
        begin_t = time.time()
        clf = GridSearchCV(xgb.XGBClassifier(seed=7), tuned_params, scoring='roc_auc')
        clf.fit(X_train, y_train)
        end_t = time.time()
        print('train time: ', round(end_t - begin_t, 3), 's')
        print('current best parameters of xgboost: ', clf.best_params_)
        return clf.best_estimator_

    def choose_lgb_model(self, X_train, y_train):
        tuned_params = [{'objective': ['binary'], 'learning_rate': [0.01, 0.03, 0.05, 0.1],
                         'n_estimators': [100, 150, 200, 400], 'max_depth': [4, 6, 8, 10]}]
        begin_t = time.time()
        clf = GridSearchCV(lgb.LGBMClassifier(seed=7), tuned_params, scoring='roc_auc')
        clf.fit(X_train, y_train)
        end_t = time.time()
        print('train time: ', round(end_t - begin_t, 3), 's')
        print('current best parameters of lgb: ', clf.best_params_)
        return clf.best_estimator_

    def choose_forest_model(self, X_train, y_train):
        tuned_params = [
            {'n_estimators': [100, 150, 200, 250], 'max_features': [8, 15, 30, 60], 'max_depth': [4, 8, 10, 12]}]
        begin_t = time.time()
        clf = GridSearchCV(RandomForestClassifier(random_state=7), tuned_params, scoring='roc_auc', n_jobs=4)
        clf.fit(X_train, y_train)
        end_t = time.time()
        print('train time: ', round(end_t - begin_t, 3), 's')
        print('current best parameters: ', clf.best_params_)
        return clf.best_estimator_

    def choose_gradient_model(self, X_train, y_train):
        tuned_params = [{'n_estimators': [100, 150, 200], 'learning_rate': [0.03, 0.05, 0.07],
                         'min_samples_leaf': [8, 15, 30], 'max_depth': [4, 6, 8]}]
        begin_t = time.time()
        clf = GridSearchCV(GradientBoostingClassifier(random_state=7), tuned_params, scoring='roc_auc')
        clf.fit(X_train, y_train)
        end_t = time.time()
        print('train time: ', round(end_t - begin_t, 3), 's')
        print('current best parameters: ', clf.best_params_)
        return clf.best_estimator_

    def stacking_model(self, X_train, X_test, y_train, bst_xgb, bst_lgb):
        '''
        使用stacking集成两个综合表现最佳的模型lgb和xgb，此处元分类器使用较为简单的LR模型来在已经训练好了并且经过参数选择的模型上进一步优化预测结果
        :param X_train:
        :param X_test:
        :param y_train:
        :param bst_xgb:
        :param bst_lgb:
        :return:
        '''
        lr = linear_model.LogisticRegression(random_state=7)
        sclf = StackingClassifier(classifiers=[bst_xgb, bst_lgb], use_probas=True, average_probas=False,
                                  meta_classifier=lr)
        sclf.fit(X_train, y_train)
        predictions = sclf.predict_proba(X_test)[:, 1]
        return predictions

    def stacking_model2(self, X_train, X_test, y_train, bst_xgb, bst_forest, bst_gradient, bst_lgb):
        '''
        组合四种算法
        :param X_train: 训练集
        :param X_test: 测试集
        :param y_train: 训练标签
        :param bst_xgb: xgb最优参数
        :param bst_forest: forest最优参数
        :param bst_gradient: gradient最优参数
        :param bst_lgb: lgb最优参数
        :return: 预测结果
        '''
        lr = linear_model.LogisticRegression(random_state=7)
        sclf = StackingClassifier(classifiers=[bst_xgb, bst_forest, bst_gradient, bst_lgb], use_probas=True,
                                  average_probas=False, meta_classifier=lr)
        sclf.fit(X_train, y_train)
        predictions = sclf.predict_proba(X_test)[:, 1]
        return predictions

    def voting_model(self, X_train, X_test, y_train, bst_xgb, bst_forest, bst_gradient, bst_lgb):

        vclf = VotingClassifier(estimators=[('xgb', bst_xgb), ('rf', bst_forest), ('gbm', bst_gradient),
                                            ('lgb', bst_lgb)], voting='soft', weights=[2, 1, 1, 2])
        vclf.fit(X_train, y_train)
        predictions = vclf.predict_proba(X_test)[:, 1]
        return predictions

    def submit(self, X_train, X_test, y_train, test_ids):
        '''
        TODO 提交
        :param X_train:
        :param X_test:
        :param y_train:
        :param test_ids:
        :return:
        '''
        predictions = self.voting_model(X_train, X_test, y_train)

        sub = pd.read_csv('sampleSubmission.csv')
        result = pd.DataFrame()
        result['bidder_id'] = test_ids
        result['outcome'] = predictions
        sub = sub.merge(result, on='bidder_id', how='left')

        # Fill missing values with mean
        mean_pred = np.median(predictions)
        sub.fillna(mean_pred, inplace=True)

        sub.drop('prediction', 1, inplace=True)
        sub.to_csv('result.csv', index=False, header=['bidder_id', 'prediction'])

    def init_data(self):
        app_train_domain = pd.read_csv('./data/app_train_domain.csv')
        self.test_full = pd.read_csv('./data/app_test_domain.csv')
        self.ytrain = app_train_domain['TARGET']
        self.train_full = app_train_domain.drop(columns='TARGET')

    def main(self, model_name, model):
        self.dct_scores[model_name], self.mean_score[model_name], self.mean_time[model_name] = self.kfold_plot(
            self.train_full, self.ytrain, model)

    def run(self):
        # model_map = {
        #     'forest': self.forest_model,
        #     # 'gbm': self.gradient_model,
        #     'xgboost': self.xgboost_model,
        #     'lgbm': self.lightgbm_model
        # }
        # for (key, value) in model_map.items():
        #     self.main(key, value)
        # # 比较四个模型在交叉验证机上的roc_auc平均得分和模型训练的时间
        # print(self.mean_score)
        # self.plot_model_comp('Model Performance', 'roc-auc score', self.mean_score)
        # self.plot_time_comp('Time of Building Model', 'time(s)', self.mean_time)
        # self.plot_auc_score(self.dct_scores)
        # self.choose_forest_model(self.train_full, self.ytrain)
        SK_ID_CURR = self.test_full['SK_ID_CURR']
        self.test_full = self.test_full.drop(['SK_ID_CURR'],axis=1)
        self.train_full = self.train_full.drop(['SK_ID_CURR'], axis=1)

        bst_xgb = self.choose_xgb_model(self.train_full, self.ytrain)
        bst_forest = self.choose_forest_model(self.train_full, self.ytrain)
        bst_gradient = self.choose_gradient_model(self.train_full, self.ytrain)
        bst_lgb = self.choose_lgb_model(self.train_full, self.ytrain)


        sub = pd.DataFrame()
        sub['SK_ID_CURR'] = SK_ID_CURR
        sub['TARGET'] = self.voting_model(self.train_full, self.test_full, self.ytrain, bst_xgb, bst_forest, bst_gradient, bst_lgb)
        sub.to_csv('./data/sub.csv', index=False)


train = Train()
# train.process_data()
# train.get_data()
train.init_data()
train.run()
