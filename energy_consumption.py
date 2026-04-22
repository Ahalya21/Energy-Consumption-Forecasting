import pandas as pd

# Load datasets
house_df = pd.read_csv("household_power_consumption.csv")
city_df = pd.read_csv("C:/Users/Sasi Ahalya Nair/Downloads/archive (3)/PJM_Load_hourly.csv")

# Household preprocessing for forecasting
house_df.replace('?', pd.NA, inplace=True)
house_df['Datetime'] = pd.to_datetime(house_df['Date'] + ' ' + house_df['Time'], dayfirst=True)
house_df['Global_active_power'] = pd.to_numeric(house_df['Global_active_power'], errors='coerce')

house_forecast_data = house_df[['Datetime', 'Global_active_power']].dropna()
house_forecast_data = house_forecast_data.rename(columns={'Global_active_power': 'Energy'})
house_forecast_data = house_forecast_data.set_index('Datetime').resample('h').mean().reset_index()

# City preprocessing for forecasting
city_df['Datetime'] = pd.to_datetime(city_df.iloc[:, 0])
city_df['Energy'] = pd.to_numeric(city_df.iloc[:, 1], errors='coerce')

city_forecast_data = city_df[['Datetime', 'Energy']].dropna()

# Save real forecasting datasets
house_forecast_data.to_csv("household_forecasting_dataset.csv", index=False)
city_forecast_data.to_csv("city_forecasting_dataset.csv", index=False)

print("Forecasting datasets saved successfully")
print(house_forecast_data.head())
print(city_forecast_data.head())
