# coding: utf-8
# https://segmentfault.com/a/1190000009101577#articleHeader6
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import metrics
import uuid
import pandas as pd
import numpy as np
import pickle
import time


class ML:
    def __init__(self):
        self.result = pd.DataFrame()
    # KNN Classifier
    def knn_classifier(self, train_x, train_y):
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=1, n_jobs=5)
        model.fit(train_x, train_y)
        return model

    # Logistic Regression Classifier
    def logistic_regression_classifier(self, train_x, train_y):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(penalty='l2', n_jobs=1)
        model.fit(train_x, train_y)
        return model

    # Random Forest Classifier
    def random_forest_classifier(self, train_x, train_y):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=40, n_jobs=4)
        model.fit(train_x, train_y)
        return model

    # Decision Tree Classifier
    def decision_tree_classifier(self, train_x, train_y):
        from sklearn import tree
        model = tree.DecisionTreeClassifier()
        model.fit(train_x, train_y)
        return model

    # GBDT(Gradient Boosting Decision Tree) Classifier
    def gradient_boosting_classifier(self, train_x, train_y):
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=200)
        model.fit(train_x, train_y)
        return model

    # SVM Classifier
    def svm_classifier(self, train_x, train_y):
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', probability=True)
        model.fit(train_x, train_y)
        return model

    # SVM Classifier using cross validation
    def svm_cross_validation(self, train_x, train_y):
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', probability=True)
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
        grid_search = GridSearchCV(model, param_grid, n_jobs=4, verbose=1)
        grid_search.fit(train_x, train_y)
        best_parameters = grid_search.best_estimator_.get_params()
        for para, val in best_parameters.items():
            print(para, val)
        model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
        model.fit(train_x, train_y)
        return model

    def split_data(self, train, ytrain):
        from sklearn.model_selection import train_test_split
        # x为数据集的feature熟悉，y为label.
        x_train, x_test, y_train, y_test = train_test_split(train, ytrain, test_size=0.1)
        return x_train, y_train, x_test, y_test

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

        mean_pred = np.mean(app_train)
        app_train.fillna(mean_pred, inplace=True)
        mean_pred = np.mean(app_test)
        app_test.fillna(mean_pred, inplace=True)

        app_train, app_test = app_train.align(app_test, join='inner', axis=1)

        app_train['TARGET'] = train_labels

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

        return app_train_domain, app_test_domain

    def train_data(self, app_train, app_test, train_labels):
        model_save_file = None
        model_save = {}
        classifiers = {
            # 'KNN': self.knn_classifier,
            'LR': self.logistic_regression_classifier,
            # 'RF': self.random_forest_classifier,
            # 'DT': self.decision_tree_classifier,
            # 'SVM': self.svm_classifier,
            # 'SVMCV': self.svm_cross_validation,
            # 'GBDT': self.gradient_boosting_classifier,
        }
        test_classifiers = classifiers.keys()

        print('reading training and testing data...')
        train_x, train_y, test_x, test_y = self.split_data(app_train, train_labels)
        num_train, num_feat = train_x.shape
        num_test, num_feat = test_x.shape
        is_binary_class = (len(np.unique(train_y)) == 2)
        print('******************** Data Info *********************')
        print('#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat))

        for classifier in test_classifiers:
            print('******************* %s ********************' % classifier)
            start_time = time.time()
            model = classifiers[classifier](train_x, train_y)
            print('training took %fs!' % (time.time() - start_time))

            predict = model.predict(test_x)
            if model_save_file != None:
                model_save[classifier] = model
            if is_binary_class:
                precision = metrics.precision_score(test_y, predict)
                recall = metrics.recall_score(test_y, predict)
                print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
            accuracy = metrics.accuracy_score(test_y, predict)
            print('accuracy: %.2f%%' % (100 * accuracy))
            app_test_pre = model.predict(app_test)
            self.result[uuid.uuid1().hex] = app_test_pre

        if model_save_file != None:
            pickle.dump(model_save, open(model_save_file, 'wb'))

    def process_train_data(self, app_train):
        len_1 = len(app_train[app_train['TARGET'] == 1])
        app_train_0 = app_train[app_train['TARGET'] == 0].sample(len_1*3)
        app_train_1 = app_train[app_train['TARGET'] == 1]
        new_app_train = pd.concat([app_train_0, app_train_1])
        new_app_train = new_app_train.sample(len_1*2)
        return new_app_train

    def run(self):
        num = 100
        app_train, app_test = self.get_data()
        app_train = self.process_train_data(app_train)
        train_labels = list(app_train['TARGET'])
        app_test_id = app_test['SK_ID_CURR']
        app_train = app_train.drop(['TARGET'], axis=1)
        scaler = preprocessing.StandardScaler().fit(app_train)
        app_train_scaler = scaler.transform(app_train)
        app_test_scaler = scaler.transform(app_test)
        for i in range(num):
            print(i)
            self.train_data(app_train_scaler, app_test_scaler, train_labels)

        sub = pd.DataFrame()
        sub['SK_ID_CURR'] = app_test_id
        sub['TARGET'] = self.result.apply(lambda x: np.array(x).sum()//100, axis=1)
        sub.to_csv('./data/sub.csv', index=False)


train = ML()
train.run()
