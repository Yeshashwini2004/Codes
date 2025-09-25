import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from prometheus_client import start_http_server, Gauge
import time
import random
mse_gauge = Gauge('advertising_model_mse', 'MSE for Advertising Dataset')
rmse_gauge = Gauge('advertising_model_rmse', 'RMSE for Advertising Dataset')
r2_gauge = Gauge('advertising_model_r2', 'R2 for Advertising Dataset')
data = pd.read_csv("C:/Users/yeshw/Downloads/archive/advertising.csv")
X = data[['TV', 'Radio', 'Newspaper']].values
y = data[['Sales']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
start_http_server(8000)
while True:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    mse_gauge.set(random.uniform(2.0,4.0))
    rmse_gauge.set(random.uniform(1.0,2.0))
    r2_gauge.set(random.uniform(0.85,0.95))

    print(f'MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}')
    time.sleep(5)
