import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()

import os
print(os.listdir("./csv"))

application_train = pd.read_csv('./csv/application_train.csv')
POS_CASH_balance = pd.read_csv('./csv/POS_CASH_balance.csv')
bureau_balance = pd.read_csv('./csv/bureau_balance.csv')
previous_application = pd.read_csv('./csv/previous_application.csv')
installments_payments = pd.read_csv('./csv/installments_payments.csv')
credit_card_balance = pd.read_csv('./csv/credit_card_balance.csv')
bureau = pd.read_csv('./csv/bureau.csv')
application_test = pd.read_csv('./csv/application_test.csv')


print('Size of application_train data    ', application_train.shape)
print('Size of POS_CASH_balance data     ', POS_CASH_balance.shape)
print('Size of bureau_balance data       ', bureau_balance.shape)
print('Size of previous_application data ', previous_application.shape)
print('Size of installments_payments data', installments_payments.shape)
print('Size of credit_card_balance data  ', credit_card_balance.shape)
print('Size of bureau data               ', bureau.shape)

application_train.head()