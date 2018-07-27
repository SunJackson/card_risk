import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def pca_process(df):
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    pca = PCA(n_components=0.99)
    pca_df = df.drop(columns=['TARGET'])
    del df
    gc.collect()
    pca_df = pca_df.fillna(0)

    X = train_df.drop(columns=['TARGET'])
    scaler = StandardScaler().fit(pca_df)
    scaler_df = scaler.transform(pca_df)
    pca.fit(scaler_df)
    print(pca.explained_variance_ratio_)
    print(pca.n_components_)


def run_main():
    df = pd.read_csv('./data/lightGBM.csv', encoding='utf-8')
    with timer('pca'):
        pca_process(df)


if __name__ == '__main__':
    run_main()