# coding: utf-8
import pandas as pd
from sqlalchemy import create_engine
import os
import gc

list_name = []
path = './Downloads'


def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file_path)[1] == '.csv':
            list_name.append(file_path)
    return list_name


mysql_engine = create_engine('mysql+pymysql://root:930606@localhost:3306/card_risk')
data_csv = listdir(path, list_name)

for csv in data_csv:
    name = csv.split('/')[-1].replace('.csv', '')
    print(name)
    if name == 'HomeCredit_columns_description':
        continue
    csv_data = pd.read_csv(csv, encoding="utf-8")
    csv_data.to_sql(con=mysql_engine, name=name, index=False, chunksize=3000, if_exists='replace')
    gc.enable()
    del csv_data
    gc.collect()
