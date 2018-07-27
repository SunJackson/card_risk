# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

app_train = pd.read_csv('./Downloads/application_train.csv')
print('Training data shape: ', app_train.shape)

# Testing data features
app_test = pd.read_csv('./Downloads/application_test.csv')
print('Testing data shape: ', app_test.shape)


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

app_train = app_train.drop(missing_values.index, axis=1)

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
# pd.DataFrame(app_train).to_csv('app_train.csv', sep=',')
print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)

test_ID = app_test['SK_ID_CURR']
train_data = app_train.drop(['SK_ID_CURR'], axis=1).as_matrix()
test_data = app_test.drop(['SK_ID_CURR'], axis=1).as_matrix()
print(train_labels.value_counts())
model = Sequential()

model.add(Dense(input_dim=len(app_train.columns) - 1, output_dim=240))  # 添加输入层、隐藏层的连接
model.add(Activation('relu'))  # 以Relu函数为激活函数
model.add(Dense(input_dim=240, output_dim=120))  # 添加隐藏层、隐藏层的连接
model.add(Activation('relu'))  # 以Relu函数为激活函数
model.add(Dense(input_dim=20, output_dim=1)) # 添加隐藏层、隐藏层的连接
model.add(Activation('sigmoid'))  # 以sigmoid函数为激活函数
# 编译模型，损失函数为binary_crossentropy，用adam法求解
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(train_data, list(train_labels), epochs=2, batch_size=1000)  # 训练模型

# 做预测

pre = model.predict_proba(test_data)

submission = pd.DataFrame({'SK_ID_CURR': list(test_ID), 'TARGET': np.ravel(pre)})

submission.to_csv('submission.csv', sep=',', index=False)
