# -*- coding: utf-8 -*-
# 导入需要的库
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import os
from sklearn import cross_validation, metrics
# Suppress warnings
import warnings

warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# ## Read in Data
#
# First, we can list all the available data files. There are a total of 9 files: 1 main file for training (with target) 1 main file for testing (without the target), 1 example submission file, and 6 other files containing additional information about each loan.

# In[2]:


# List files available
print(os.listdir("../Downloads/"))

# Training data
app_train = pd.read_csv('../Downloads/application_train.csv')
print('Training data shape: ', app_train.shape)
app_train.head()

# Testing data features
app_test = pd.read_csv('../Downloads/application_test.csv')
print('Testing data shape: ', app_test.shape)
app_test.head()

app_train['TARGET'].value_counts()

app_train['TARGET'].astype(int).plot.hist();


# From this information, we see this is an [_imbalanced class problem_](http://www.chioka.in/class-imbalance-problem/). There are far more loans that were repaid on time than loans that were not repaid. Once we get into more sophisticated machine learning models, we can [weight the classes](http://xgboost.readthedocs.io/en/latest/parameter.html) by their representation in the data to reflect this imbalance.

# ## Examine Missing Values
#
# Next we can look at the number and percentage of missing values in each column.

# Function to calculate missing values by column# Funct
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
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# Missing values statistics
missing_values = missing_values_table(app_train)
missing_values.head(20)

app_train = app_train.drop(missing_values.index, axis=1)
# Number of each type of column
app_train.dtypes.value_counts()

# Let's now look at the number of unique entries in each of the `object` (categorical) columns.


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

# In[12]:

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
# app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

# 不管任何参数，都用默认的，拟合下数据看看
# rf0 = RandomForestClassifier(oob_score=True, random_state=30)
# rf0.fit(app_train, train_labels)
#
# print(rf0.oob_score_)
# y_predprob = rf0.predict_proba(app_train)[:, 1]
# print("AUC Score (Train): {}".format(metrics.roc_auc_score(train_labels, y_predprob)))
# 输出如下：0.9069073951826113  AUC Score (Train): 0.9998490132055681
# 可见袋外分数已经很高（理解为袋外数据作为验证集时的准确率，也就是模型的泛化能力），而且AUC分数也很高（AUC是指从一堆样本中随机抽一个，
# 抽到正样本的概率比抽到负样本的概率大的可能性）。相对于GBDT的默认参数输出，RF的默认参数拟合效果对本例要好一些。

# 首先对n_estimators进行网格搜索
# param_test1 = {'n_estimators': range(10, 71, 10)}
# gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,
#                                                          min_samples_leaf=20,
#                                                          max_depth=8,
#                                                          max_features='sqrt',
#                                                          random_state=10),
#                         param_grid=param_test1, scoring='roc_auc', cv=5)
# gsearch1.fit(app_train, train_labels)
# print(gsearch1.scorer_, gsearch1.best_params_, gsearch1.best_score_)

# 这样我们得到了最佳的弱学习器迭代次数，接着我们对决策树最大深度max_depth和内部节点再划分所需最小样本数
# min_samples_split进行网格搜索。
param_test2 = {'max_depth': range(3, 40, 2), 'min_samples_split': range(1000, 20001, 200)}
gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=70,
                                                         n_jobs=5,
                                                         min_samples_leaf=20,
                                                         max_features='sqrt',
                                                         oob_score=True,
                                                         random_state=10),
                        param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
gsearch2.fit(app_train, train_labels)
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_
