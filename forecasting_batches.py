'''import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# UTILITY FUNCTION
def to_2d_array(x):
    return np.asarray(x).reshape(-1, 1)

# LOAD DATA
house_raw = pd.read_csv("household_power_consumption.csv")
city_raw = pd.read_csv("C:/Users/Sasi Ahalya Nair/Downloads/archive (3)/PJM_Load_hourly.csv")

# HOUSEHOLD PREPROCESSING
house_raw.replace("?", np.nan, inplace=True)

house_raw["Datetime"] = pd.to_datetime(
    house_raw["Date"] + " " + house_raw["Time"],
    dayfirst=True,
    errors="coerce"
)

house_raw["Global_active_power"] = pd.to_numeric(
    house_raw["Global_active_power"], errors="coerce"
)

house = house_raw[["Datetime", "Global_active_power"]].dropna().copy()
house = house.rename(columns={"Global_active_power": "Energy"})
house = house.set_index("Datetime").sort_index()

house = house.resample("h").mean()
house["Energy"] = house["Energy"].interpolate()

cutoff = house.index.max() - pd.Timedelta(days=90)
house = house.loc[house.index >= cutoff].copy()

house["Energy"] = house["Energy"].astype("float32")

# CITY PREPROCESSING
city = city_raw.copy()
city["Datetime"] = pd.to_datetime(city.iloc[:, 0], errors="coerce")
city["Energy"] = pd.to_numeric(city.iloc[:, 1], errors="coerce")
city = city[["Datetime", "Energy"]].dropna().sort_values("Datetime").reset_index(drop=True)

# HOUSEHOLD MODEL (SARIMA)
print("\n===== HOUSEHOLD (SARIMA) =====")

house_series = house["Energy"]

if len(house_series) < 300:
    raise ValueError("Not enough data after filtering. Increase window size.")

house_train = house_series[:-168]
house_test = house_series[-168:]

scaler_h = MinMaxScaler(feature_range=(0.1, 1.0))
train_scaled = scaler_h.fit_transform(to_2d_array(house_train)).flatten()
train_scaled = pd.Series(train_scaled, index=house_train.index)

model = SARIMAX(
    train_scaled,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 24),
    enforce_stationarity=False,
    enforce_invertibility=False
)

result = model.fit(disp=False, low_memory=True)

forecast = result.get_forecast(steps=len(house_test))
pred_scaled = np.asarray(forecast.predicted_mean)
yhat_raw = scaler_h.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
yhat = np.maximum(yhat_raw, 0.05)

house_actual = house_test.values

print("\n--- Household Test Predictions ---")
for i in range(min(10, len(yhat))):
    print(f"Actual: {house_actual[i]:.3f} | Predicted: {yhat[i]:.3f}")

mae = mean_absolute_error(house_actual, yhat)
rmse = np.sqrt(mean_squared_error(house_actual, yhat))

print(f"\nHousehold MAE: {mae:.4f}")
print(f"Household RMSE: {rmse:.4f}")

full_scaled = scaler_h.transform(to_2d_array(house_series)).flatten()
full_scaled = pd.Series(full_scaled, index=house_series.index)

future_model = SARIMAX(
    full_scaled,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 24),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False, low_memory=True)

future_fc = future_model.get_forecast(steps=168)
future_pred_raw = scaler_h.inverse_transform(
    np.asarray(future_fc.predicted_mean).reshape(-1, 1)
).flatten()
future_pred = np.maximum(future_pred_raw, 0.05)

print("\n--- Household Future Predictions (next 10 hours) ---")
for i in range(min(10, len(future_pred))):
    print(f"Hour {i+1}: {future_pred[i]:.3f}")

# CITY MODEL (PROPHET) - 7 BATCHES
print("\n===== CITY (PROPHET - 7 BATCHES) =====")

batch_size = 5000
num_batches = 7

for batch_num in range(num_batches):
    start_idx = batch_num * batch_size
    end_idx = start_idx + batch_size

    city_batch = city.iloc[start_idx:end_idx].copy()

    if len(city_batch) < 200:
        print(f"\nBatch {batch_num + 1}: Not enough rows, skipped.")
        continue

    print(f"\n===== CITY BATCH {batch_num + 1} =====")
    print(f"Rows used: {start_idx} to {end_idx - 1}")

    city_prophet = city_batch.rename(columns={"Datetime": "ds", "Energy": "y"}).copy()

    city_train = city_prophet.iloc[:-168].copy()
    city_test = city_prophet.iloc[-168:].copy()

    scaler_c = MinMaxScaler(feature_range=(0.1, 1.0))
    city_train.loc[:, "y"] = scaler_c.fit_transform(city_train[["y"]])

    model_p = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    model_p.fit(city_train)

    # Test prediction
    future_test = model_p.make_future_dataframe(periods=168, freq="h")
    forecast_test = model_p.predict(future_test)
    forecast_test = forecast_test[["ds", "yhat"]].tail(168)

    yhat_city = scaler_c.inverse_transform(
        np.asarray(forecast_test["yhat"]).reshape(-1, 1)
    ).flatten()

    yhat_city = np.clip(yhat_city, 0, None)
    city_actual = city_test["y"].values

    print("\n--- City Test Predictions ---")
    for i in range(min(10, len(yhat_city))):
        print(f"Actual: {city_actual[i]:.3f} | Predicted: {yhat_city[i]:.3f}")

    mae_c = mean_absolute_error(city_actual, yhat_city)
    rmse_c = np.sqrt(mean_squared_error(city_actual, yhat_city))

    print(f"\nCity Batch {batch_num + 1} MAE: {mae_c:.4f}")
    print(f"City Batch {batch_num + 1} RMSE: {rmse_c:.4f}")

    # Future prediction
    city_full = city_prophet.copy()
    city_full.loc[:, "y"] = scaler_c.transform(city_full[["y"]])

    model_final = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    model_final.fit(city_full)

    future = model_final.make_future_dataframe(periods=168, freq="h")
    forecast_future = model_final.predict(future)[["ds", "yhat"]].tail(168)

    future_city = scaler_c.inverse_transform(
        np.asarray(forecast_future["yhat"]).reshape(-1, 1)
    ).flatten()

    future_city = np.clip(future_city, 0, None)

    print("\n--- City Future Predictions (next 10 hours) ---")
    for i in range(min(10, len(future_city))):
        print(f"Hour {i+1}: {future_city[i]:.3f}")'''






































import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# UTILITY FUNCTION
def to_2d_array(x):
    return np.asarray(x).reshape(-1, 1)

# LOAD DATA
house_raw = pd.read_csv("household_power_consumption.csv")
city_raw = pd.read_csv("C:/Users/Sasi Ahalya Nair/Downloads/archive (3)/PJM_Load_hourly.csv")

# HOUSEHOLD PREPROCESSING
house_raw.replace("?", np.nan, inplace=True)

house_raw["Datetime"] = pd.to_datetime(
    house_raw["Date"] + " " + house_raw["Time"],
    dayfirst=True,
    errors="coerce"
)

house_raw["Global_active_power"] = pd.to_numeric(
    house_raw["Global_active_power"], errors="coerce"
)

house = house_raw[["Datetime", "Global_active_power"]].dropna().copy()
house = house.rename(columns={"Global_active_power": "Energy"})
house = house.set_index("Datetime").sort_index()

house = house.resample("h").mean()
house["Energy"] = house["Energy"].interpolate()

cutoff = house.index.max() - pd.Timedelta(days=90)
house = house.loc[house.index >= cutoff].copy()

house["Energy"] = house["Energy"].astype("float32")

# HOUSEHOLD MODEL (SARIMA)
print("\n===== HOUSEHOLD (SARIMA) =====")

house_series = house["Energy"]

# Split
house_train = house_series[:-168]
house_test = house_series[-168:]

#  NORMALIZATION (single scaler)
scaler_h = MinMaxScaler(feature_range=(0, 1))

train_scaled = scaler_h.fit_transform(to_2d_array(house_train)).flatten()
test_scaled = scaler_h.transform(to_2d_array(house_test)).flatten()

train_scaled = pd.Series(train_scaled, index=house_train.index)

# MODEL
model = SARIMAX(
    train_scaled,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 24),
    enforce_stationarity=False,
    enforce_invertibility=False
)

result = model.fit(disp=False)

# FORECAST (TEST)
forecast = result.get_forecast(steps=len(house_test))
pred_scaled = np.asarray(forecast.predicted_mean)

# INVERSE
yhat = scaler_h.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
yhat = np.maximum(yhat, 0)

house_actual = house_test.values

# PRINT
print("\n--- Household Test Predictions ---")
for i in range(min(10, len(yhat))):
    print(f"Actual: {house_actual[i]:.3f} | Predicted: {yhat[i]:.3f}")

# METRICS
mae = mean_absolute_error(house_actual, yhat)
rmse = np.sqrt(mean_squared_error(house_actual, yhat))

print(f"\nHousehold MAE: {mae:.4f}")
print(f"Household RMSE: {rmse:.4f}")

# HOUSEHOLD FUTURE
# Use SAME scaler (no refitting)
full_scaled = scaler_h.transform(to_2d_array(house_series)).flatten()
full_scaled = pd.Series(full_scaled, index=house_series.index)

future_model = SARIMAX(
    full_scaled,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 24),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

future_fc = future_model.get_forecast(steps=168)

future_pred = scaler_h.inverse_transform(
    np.asarray(future_fc.predicted_mean).reshape(-1, 1)
).flatten()

future_pred = np.maximum(future_pred, 0)

print("\n--- Household Future Predictions (next 10 hours) ---")
for i in range(min(10, len(future_pred))):
    print(f"Hour {i+1}: {future_pred[i]:.3f}")

# CITY PREPROCESSING
city = city_raw.copy()
city["Datetime"] = pd.to_datetime(city.iloc[:, 0], errors="coerce")
city["Energy"] = pd.to_numeric(city.iloc[:, 1], errors="coerce")
city = city[["Datetime", "Energy"]].dropna().sort_values("Datetime").reset_index(drop=True)

# CITY MODEL (PROPHET)
print("\n===== CITY (PROPHET) =====")

batch_size = 5000
num_batches = 7

for batch_num in range(num_batches):

    start_idx = batch_num * batch_size
    end_idx = start_idx + batch_size

    city_batch = city.iloc[start_idx:end_idx].copy()

    if len(city_batch) < 200:
        print(f"\nBatch {batch_num + 1}: Skipped")
        continue

    print(f"\n===== CITY BATCH {batch_num + 1} =====")

    city_prophet = city_batch.rename(columns={"Datetime": "ds", "Energy": "y"}).copy()

    city_train = city_prophet.iloc[:-168].copy()
    city_test = city_prophet.iloc[-168:].copy()

    #  SINGLE SCALER PER BATCH
    scaler_c = MinMaxScaler(feature_range=(0, 1))

    city_train["y"] = scaler_c.fit_transform(city_train[["y"]])
    city_test["y"] = scaler_c.transform(city_test[["y"]])

    # MODEL
    model_p = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    model_p.fit(city_train)

    # FORECAST TEST
    future_test = model_p.make_future_dataframe(periods=168, freq="h")
    forecast_test = model_p.predict(future_test)[["ds", "yhat"]].tail(168)

    yhat_city = scaler_c.inverse_transform(
        forecast_test["yhat"].values.reshape(-1, 1)
    ).flatten()

    yhat_city = np.clip(yhat_city, 0, None)

    # INVERSE ACTUAL
    city_actual = scaler_c.inverse_transform(
        city_test["y"].values.reshape(-1, 1)
    ).flatten()

    # PRINT
    print("\n--- City Test Predictions ---")
    for i in range(min(10, len(yhat_city))):
        print(f"Actual: {city_actual[i]:.3f} | Predicted: {yhat_city[i]:.3f}")

    # METRICS
    mae_c = mean_absolute_error(city_actual, yhat_city)
    rmse_c = np.sqrt(mean_squared_error(city_actual, yhat_city))

    print(f"\nCity MAE: {mae_c:.4f}")
    print(f"City RMSE: {rmse_c:.4f}")

    # FUTURE FORECAST (USE SAME SCALER)
    city_full = city_prophet.copy()
    city_full["y"] = scaler_c.transform(city_full[["y"]])

    model_final = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    model_final.fit(city_full)

    future = model_final.make_future_dataframe(periods=168, freq="h")
    forecast_future = model_final.predict(future)[["ds", "yhat"]].tail(168)

    future_city = scaler_c.inverse_transform(
        forecast_future["yhat"].values.reshape(-1, 1)
    ).flatten()

    future_city = np.clip(future_city, 0, None)

    print("\n--- City Future Predictions (next 10 hours) ---")
    for i in range(min(10, len(future_city))):
        print(f"Hour {i+1}: {future_city[i]:.3f}")



house_results = pd.DataFrame({
    "Datetime": house_test.index,
    "Actual": house_actual,
    "Predicted": yhat
})

house_results.to_csv("household_predictions.csv", index=False)

#future forecast
future_dates = pd.date_range(
    start=house_series.index[-1] + pd.Timedelta(hours=1),
    periods=168,
    freq="h"
)

house_future_df = pd.DataFrame({
    "Datetime": future_dates,
    "Forecast": future_pred
})

house_future_df.to_csv("household_future.csv", index=False)


#for city


all_city_results = []
all_city_future = []


city_results = pd.DataFrame({
    "Datetime": forecast_test["ds"],
    "Actual": city_actual,
    "Predicted": yhat_city,
    "Batch": batch_num + 1
})

all_city_results.append(city_results)

#future forecast
city_future_df = pd.DataFrame({
    "Datetime": forecast_future["ds"],
    "Forecast": future_city,
    "Batch": batch_num + 1
})

all_city_future.append(city_future_df)

final_city_results = pd.concat(all_city_results, ignore_index=True)
final_city_future = pd.concat(all_city_future, ignore_index=True)

final_city_results.to_csv("city_all_predictions.csv", index=False)
final_city_future.to_csv("city_all_future.csv", index=False)

















































































