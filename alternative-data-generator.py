
import pandas as pd
import numpy as np

def generate_alternative_data(num_samples):
    square_feet = np.random.randint(900, 4000, num_samples)
    bathrooms = np.random.randint(1, 4, num_samples)
    year_built = np.random.randint(1900, 2021, num_samples)
    lot_size = np.random.randint(1000, 10000, num_samples)
    

    base_price = 70000
    price = base_price + square_feet * 15 + bathrooms * 16000 - year_built * 500
    price_noise = np.random.randn(num_samples) * 10000  
    price += price_noise.astype(int)
    
    price[price < 70000] = 70000 + np.abs(price_noise[price < 70000].astype(int))
    
    new_data = pd.DataFrame({
        'square_feet': square_feet,
        'bathrooms': bathrooms,
        'year_built': year_built,
        'lot_size': lot_size,
        'price': price
    })
    
    for col in new_data.columns:
        new_data.loc[new_data.sample(frac=0.05).index, col] = np.nan
    
    outlier_indices = new_data.sample(frac=0.01).index
    new_data.loc[outlier_indices, 'square_feet'] *= 1.2
    new_data.loc[outlier_indices, 'bathrooms'] += 1
    new_data.loc[outlier_indices, 'year_built'] -= 10
    new_data.loc[outlier_indices, 'lot_size'] *= 1.1
    new_data.loc[outlier_indices, 'price'] *= 0.7
    
    return new_data

num_samples = 500000
new_data = generate_alternative_data(num_samples)

alternative_file_path = 'alternative-house-price-data.csv'
new_data.to_csv(alternative_file_path, index=False)

alternative_file_path