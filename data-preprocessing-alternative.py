

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

new_data = pd.read_csv('alternative-house-price-data.csv')

transformed_data = new_data.copy()
transformed_data['area'] = transformed_data['square_feet']
transformed_data['bedrooms'] = transformed_data['bathrooms'] * 2
transformed_data['age'] = 2024 - transformed_data['year_built']

transformed_data.drop(['square_feet', 'bathrooms', 'year_built', 'lot_size'], axis=1, inplace=True)

data = transformed_data
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

file_path = 'house-price-data-after-preprocessing.csv'
data_normalized.to_csv(file_path, index=False)
print(data_normalized.head(10))