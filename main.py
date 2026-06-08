import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("url_BlackFridaySales.csv")

print(df.head())

df.fillna(0, inplace=True)

le = LabelEncoder()

# Drop User_ID as it is an identifier
df = df.drop('User_ID', axis=1)

df['Gender'] = le.fit_transform(df['Gender'])
df['Age'] = le.fit_transform(df['Age'])
df['City_Category'] = le.fit_transform(df['City_Category'])
# Encode Product_ID and Stay_In_Current_City_Years
df['Product_ID'] = le.fit_transform(df['Product_ID'])
df['Stay_In_Current_City_Years'] = le.fit_transform(df['Stay_In_Current_City_Years'])

X = df.drop("Purchase", axis=1)
y = df["Purchase"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


error = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error:", error)
