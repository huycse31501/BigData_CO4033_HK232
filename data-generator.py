import pandas as pd
import numpy as np

np.random.seed(0)

def generate_data(num_samples):
    area = np.random.randint(1000, 5000, num_samples)
    bedrooms = np.random.randint(1, 6, num_samples)
    age = np.random.randint(0, 101, num_samples)


    base_price = 70000
    price = base_price + area * 15 + bedrooms * 8000 - age * 500
    price_noise = np.random.randn(num_samples) * 10000  
    price += price_noise.astype(int)
    
    price[price < 70000] = 70000 + np.abs(price_noise[price < 70000].astype(int))

    data = pd.DataFrame({'area': area, 'bedrooms': bedrooms, 'age': age, 'price': price})

    for col in data.columns:
        data.loc[data.sample(frac=0.05).index, col] = np.nan

    outlier_indices = data.sample(frac=0.01).index
    data.loc[outlier_indices, 'area'] = data.loc[outlier_indices, 'area'] * 1.5
    data.loc[outlier_indices, 'bedrooms'] += 2
    data.loc[outlier_indices, 'age'] = data.loc[outlier_indices, 'age'] * 1.5
    data.loc[outlier_indices, 'price'] = data.loc[outlier_indices, 'price'] * 0.5

    return data

num_samples = 500000
data = generate_data(num_samples)

file_path = 'house-price-data.csv'
data.to_csv(file_path, index=False)

file_path