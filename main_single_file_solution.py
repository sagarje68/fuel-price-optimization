"""
Fuel Price Optimization – Single File Implementation
Author: Your Name
Role: Agentic Python Developer

This script performs:
- Data ingestion
- Cleaning & feature engineering
- Model training & evaluation
- Optimization to recommend the best price
- Reads today_example.json and outputs recommended price
"""

import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------------
# SECTION 1 – DATA INGESTION
# --------------------------------------------------------------------------------
def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Convert date
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    return df


# --------------------------------------------------------------------------------
# SECTION 2 – FEATURE ENGINEERING
# --------------------------------------------------------------------------------
def add_features(df):
    df = df.copy()

    # Price competitiveness features
    df["avg_comp_price"] = df[["comp1_price", "comp2_price", "comp3_price"]].mean(axis=1)
    df["price_diff_comp"] = df["price"] - df["avg_comp_price"]

    # Profit per liter
    df["margin"] = df["price"] - df["cost"]

    # Time-based features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # Lag features
    df["lag_volume_1"] = df["volume"].shift(1)
    df["lag_price_1"] = df["price"].shift(1)

    df.fillna(method="bfill", inplace=True)

    return df


# --------------------------------------------------------------------------------
# SECTION 3 – MODEL TRAINING
# --------------------------------------------------------------------------------
def train_model(df):

    feature_cols = [
        "price", "cost", "comp1_price", "comp2_price", "comp3_price",
        "avg_comp_price", "price_diff_comp", "margin",
        "dayofweek", "month",
        "lag_volume_1", "lag_price_1"
    ]

    X = df[feature_cols]
    y = df["volume"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("\nMODEL EVALUATION")
    print("----------------------")
    print("MAE :", mean_absolute_error(y_test, pred))
    print("R2 Score :", r2_score(y_test, pred))

    return model, feature_cols


# --------------------------------------------------------------------------------
# SECTION 4 – OPTIMIZATION ENGINE
# --------------------------------------------------------------------------------
def optimize_price(model, today_features, feature_cols):

    # Search price range based on business rules
    last_price = today_features["price"]

    possible_prices = np.arange(last_price - 2, last_price + 2, 0.10)

    best_price = None
    best_profit = -1

    for p in possible_prices:
        f = today_features.copy()
        f["price"] = p
        f["margin"] = p - today_features["cost"]
        f["price_diff_comp"] = p - f["avg_comp_price"]
        f["lag_price_1"] = last_price

        # Convert to row
        row = pd.DataFrame([f])[feature_cols]

        predicted_volume = model.predict(row)[0]
        total_profit = predicted_volume * f["margin"]

        if total_profit > best_profit:
            best_profit = total_profit
            best_price = p

    return round(best_price, 2), round(best_profit, 2)


# --------------------------------------------------------------------------------
# SECTION 5 – DAILY PREDICTION FUNCTION
# --------------------------------------------------------------------------------
def recommend_today_price(model, feature_cols, json_path):

    with open(json_path, "r") as f:
        today = json.load(f)

    # Prepare feature row
    today_features = {
        "price": today["price"],
        "cost": today["cost"],
        "comp1_price": today["comp1_price"],
        "comp2_price": today["comp2_price"],
        "comp3_price": today["comp3_price"],
        "avg_comp_price": np.mean([
            today["comp1_price"], today["comp2_price"], today["comp3_price"]
        ]),
        "dayofweek": pd.Timestamp(today["date"]).dayofweek,
        "month": pd.Timestamp(today["date"]).month,
        "margin": today["price"] - today["cost"],
        "price_diff_comp": today["price"] - np.mean([
            today["comp1_price"], today["comp2_price"], today["comp3_price"]
        ]),
        "lag_volume_1": 15000,   # fallback / assumption
        "lag_price_1": today["price"]
    }

    best_price, best_profit = optimize_price(model, today_features, feature_cols)

    print("\nRECOMMENDATION FOR TODAY")
    print("---------------------------")
    print(f"Recommended Price : ₹{best_price}")
    print(f"Expected Profit   : ₹{best_profit}")

    return best_price


# --------------------------------------------------------------------------------
# SECTION 6 – MAIN PIPELINE EXECUTION
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data...")
    df = load_data("oil_retail_history.csv")

    print("Adding features...")
    df = add_features(df)

    print("Training model...")
    model, feature_cols = train_model(df)

    print("\nRunning today's price recommendation...")
    recommend_today_price(model, feature_cols, "today_example.json")
