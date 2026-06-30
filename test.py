import pandas as pd

df = pd.read_csv("data/raw/cs-training.csv")

print(df.shape)        # quantas linhas e colunas
print(df.dtypes)       # tipos de cada coluna
print(df.isnull().sum()) # quantos valores ausentes por coluna
print(df.head())       # primeiras linhas