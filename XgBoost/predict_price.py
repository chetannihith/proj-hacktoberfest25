import joblib
import re

# Load trained pipeline (make sure you've run the notebook once to save this file)
loaded = joblib.load("artifacts/xgb_car_price.joblib")

# Extract actual pipeline
model = loaded["model"] if isinstance(loaded, dict) else loaded

def parse_numeric(value):
    """Extract numeric part from strings like '1248 CC', '21.5 kmpl', '74 bhp'."""
    if isinstance(value, (int, float)):
        return value
    try:
        return float(re.findall(r"[\d.]+", str(value))[0])
    except:
        return None

def get_user_inputs():
    """Ask user for car details interactively."""
    data = {}
    data["year"] = int(input("Enter Year of Manufacture (e.g., 2017): "))
    data["km_driven"] = int(input("Enter Kilometers Driven (e.g., 45000): "))
    data["fuel_type"] = input("Fuel Type (Petrol/Diesel/CNG/LPG/Electric): ").title()
    data["seller_type"] = input("Seller Type (Dealer/Individual/Trustmark Dealer): ").title()
    data["transmission_type"] = input("Transmission (Manual/Automatic): ").title()
    data["owner"] = input("Owner (First Owner/Second Owner/...): ").title()
    data["mileage"] = parse_numeric(input("Mileage (e.g., 19.7 kmpl): "))
    data["engine"] = parse_numeric(input("Engine CC (e.g., 1197 CC): "))
    data["max_power"] = parse_numeric(input("Max Power (e.g., 83 bhp): "))
    data["seats"] = int(input("Number of Seats (e.g., 5): "))
    data["model"] = input("Car Model (e.g., Maruti Swift VXI): ")
    return data

def predict_price(data):
    import pandas as pd
    df = pd.DataFrame([data])
    return model.predict(df)[0]

if __name__ == "__main__":
    car_data = get_user_inputs()
    price = predict_price(car_data)
    print("\n===================================")
    print("Predicted Selling Price: ₹ {:.2f}".format(price))
    print("===================================")
