import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error

def train_and_save_models(data1_path, data2_path):
    data1 = pd.read_csv(data1_path)
    data2 = pd.read_csv(data2_path)
    data = pd.concat([data1, data2])

    X = data.drop('price', axis=1)
    y = data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(n_estimators=100)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f'model_{name.replace(" ", "_")}.joblib')

def get_user_input():
    area = float(input("Nhập diện tích ngôi nhà (tính bằng feet vuông): "))
    bedrooms = int(input("Nhập số lượng phòng ngủ: "))
    age = int(input("Nhập tuổi của ngôi nhà (tính bằng năm): "))
    return area, bedrooms, age

def predict_price(model_choice, area, bedrooms, age):
    models = {
        '1': 'Linear Regression',
        '2': 'Decision Tree',
        '3': 'Random Forest'
    }
    model_name = models.get(model_choice)
    if model_name is None:
        print("Lựa chọn không hợp lệ.")
        return None
    model = joblib.load(f'model_{model_name.replace(" ", "_")}.joblib')
    user_data = pd.DataFrame([[area, bedrooms, age]], columns=['area', 'bedrooms', 'age'])
    predicted_price = model.predict(user_data)
    return predicted_price[0]

def main():
    # train_and_save_models('transformed-alternative-data.csv', 'house-price-data-after-preprocessing.csv')

    print("Vui lòng nhập thông tin chi tiết của ngôi nhà:")
    area, bedrooms, age = get_user_input()

    model_choice = input("Chọn mô hình để sử dụng (1 - Hồi quy tuyến tính, 2 - Cây quyết định, 3 - Rừng ngẫu nhiên): ")
    price_prediction = predict_price(model_choice, area, bedrooms, age)
    if price_prediction is not None:
        print(f"Giá nhà ước lượng là: ${price_prediction:.2f}")

if __name__ == "__main__":
    main()