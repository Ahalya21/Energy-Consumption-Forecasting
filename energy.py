#1. Define the problem
# forecast household energy consumption using historical electricity usage data and weather factors

#Load the dataset

import pandas as pd

house_df = pd.read_csv("household_power_consumption.csv")
city_df = pd.read_csv("C:/Users/Sasi Ahalya Nair/Downloads/archive (3)/PJM_Load_hourly.csv")

# print(house_df.head())
# print(house_df.info())
# print(house_df.isnull().sum())

# print(city_df.head())
# print(city_df.info())
# print(city_df.isnull())

#3
#household preprocessing
house_df.replace('?', pd.NA, inplace=True)
house_df['Datetime'] = pd.to_datetime(house_df['Date']+' '+house_df['Time'], dayfirst=True)
house_df['Global_active_power'] = pd.to_numeric(house_df['Global_active_power'], errors='coerce')

house_clean = house_df[['Datetime', 'Global_active_power']].dropna()
house_clean = house_clean.rename(columns={'Global_active_power':'Energy'})
house_hourly = house_clean.set_index('Datetime').resample('h').mean().reset_index()

#City preprocessing
city_df['Datetime'] = pd.to_datetime(city_df.iloc[:, 0])
city_df['Energy'] = pd.to_numeric(city_df.iloc[:, 1], errors='coerce')
city_clean = city_df[['Datetime', 'Energy']].dropna()

print(house_hourly.head())
print(city_clean.head())

#4 EDA
house_hourly['Datetime'] = pd.to_datetime(house_hourly['Datetime'])
city_clean['Datetime'] = pd.to_datetime(city_clean['Datetime'])

# Create time features
for df in [house_hourly, city_clean]:
    df['hour'] = df['Datetime'].dt.hour
    df['day_name'] = df['Datetime'].dt.day_name()
    df['month'] = df['Datetime'].dt.month
    df['weekday'] = df['Datetime'].dt.weekday
    df['day_order'] = df['Datetime'].dt.weekday


house_hourly.to_csv("household_hourly_cleaned.csv", index=False)
city_clean.to_csv("city_hourly_cleaned.csv", index=False)

print("Cleaned files saved successfully")
