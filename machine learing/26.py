from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

#导入文件
df_chinese= pd.read_csv(r"C:\Users\26477\Desktop\boston\MachineLearning--\boston_house_price_Chinese.csv",encoding='gbk')
df_english= pd.read_csv(r"C:\Users\26477\Desktop\boston\MachineLearning--\boston_house_price_english.csv")

df_chinese.head()
