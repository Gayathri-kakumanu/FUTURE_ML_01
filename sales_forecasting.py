import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

os.makedirs("images", exist_ok=True)
# ==================================
# LOAD DATASET
# ==================================

df = pd.read_csv(
    "data/Sample - Superstore (1).csv",
    encoding="latin1"
)

print("Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# ==================================
# DATA CLEANING
# ==================================

df['Order Date'] = pd.to_datetime(df['Order Date'])

# ==================================
# MONTHLY SALES AGGREGATION
# ==================================

monthly_sales = df.groupby(
    pd.Grouper(key='Order Date', freq='ME')
)['Sales'].sum().reset_index()

monthly_sales['Month_Number'] = range(len(monthly_sales))

print("\nMonthly Sales:")
print(monthly_sales.head())

# ==================================
# GRAPH 1 - MONTHLY SALES TREND
# ==================================

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales['Order Date'],
    monthly_sales['Sales'],
    marker='o'
)

plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)

plt.savefig("images/monthly_sales_trend.png")

plt.show()

# ==================================
# PREPARE DATA
# ==================================

X = monthly_sales[['Month_Number']]
y = monthly_sales['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# ==================================
# LINEAR REGRESSION
# ==================================

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_mae = mean_absolute_error(
    y_test,
    lr_predictions
)

print("\nLinear Regression MAE:", lr_mae)

# ==================================
# RANDOM FOREST
# ==================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

print("Random Forest MAE:", rf_mae)

# ==================================
# SAVE RESULTS CSV
# ==================================

results = pd.DataFrame({
    'Actual Sales': y_test.values,
    'Linear Regression Prediction': lr_predictions,
    'Random Forest Prediction': rf_predictions
})

results.to_csv(
    "forecast_results.csv",
    index=False
)

print("\nResults saved to forecast_results.csv")

# ==================================
# GRAPH 2 - MODEL COMPARISON
# ==================================

plt.figure(figsize=(12, 6))

plt.plot(
    y_test.values,
    marker='o',
    label='Actual Sales'
)

plt.plot(
    lr_predictions,
    marker='s',
    label='Linear Regression'
)

plt.plot(
    rf_predictions,
    marker='^',
    label='Random Forest'
)

plt.title("Actual vs Predicted Sales Comparison")
plt.xlabel("Test Months")
plt.ylabel("Sales")

plt.legend()
plt.grid(True)

plt.savefig("images/model_comparison.png")

plt.show()

# ==================================
# FUTURE FORECAST
# ==================================

future_months = pd.DataFrame({
    'Month_Number': range(
        len(monthly_sales),
        len(monthly_sales) + 6
    )
})

future_sales = lr_model.predict(
    future_months
)

print("\nForecast for Next 6 Months:")

for i, sale in enumerate(future_sales, start=1):
    print(f"Month {i}: {sale:.2f}")

# ==================================
# FORECAST GRAPH
# ==================================

future_dates = pd.date_range(
    start=monthly_sales['Order Date'].max(),
    periods=7,
    freq='ME'
)[1:]

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales['Order Date'],
    monthly_sales['Sales'],
    marker='o',
    label='Historical Sales'
)

plt.plot(
    future_dates,
    future_sales,
    marker='o',
    linewidth=3,
    label='Forecasted Sales'
)

plt.title("Future Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.legend()
plt.grid(True)

plt.savefig("images/future_sales_forecast.png")

plt.show()

print("\nProject Completed Successfully!")
