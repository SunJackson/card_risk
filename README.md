缺失值处理：
1. 用平均值、中值、分位数、众数、随机值等替代。效果一般，因为等于人为增加了噪声。
2. 用其他变量做预测模型来算出缺失变量。效果比方法1略好。有一个根本缺陷，如果其他变量和缺失变量无关，则预测的结果无意义。如果预测结果相当准确，则又说明这个变量是没必要加入建模的。一般情况下，介于两者之间。
3. 最精确的做法，把变量映射到高维空间。比如性别，有男、女、缺失三种情况，则映射成3个变量：是否男、是否女、是否缺失。连续型变量也可以这样处理。比如Google、百度的CTR预估模型，预处理时会把所有变量都这样处理，达到几亿维。这样做的好处是完整保留了原始数据的全部信息、不用考虑缺失值、不用考虑线性不可分之类的问题。缺点是计算量大大提升。 而且只有在样本量非常大的时候效果才好，否则会因为过于稀疏，效果很差。


XGBoost、Light GBM和CatBoost

