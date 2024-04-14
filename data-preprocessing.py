import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('house-price-data.csv')


data_cleaned = data.dropna()
for col in data.columns:
    data_cleaned.loc[:, col] = pd.to_numeric(data_cleaned.loc[:, col], errors='coerce')

data_cleaned = data_cleaned.dropna()

Q1 = data_cleaned.quantile(0.25)
Q3 = data_cleaned.quantile(0.75)
IQR = Q3 - Q1

data_no_outliers = data_cleaned[~((data_cleaned < (Q1 - 1.5 * IQR)) | (data_cleaned > (Q3 + 1.5 * IQR))).any(axis=1)]

if data_no_outliers.isnull().any().any():
        data.dropna(inplace=True)

if data_no_outliers.isnull().values.any():
    data_no_outliers.dropna(inplace=True)


scaler = MinMaxScaler()
data_normalized = pd.DataFrame(scaler.fit_transform(data_no_outliers), columns=data_no_outliers.columns)

file_path = 'transformed-alternative-data.csv'
data_normalized.to_csv(file_path, index=False)
