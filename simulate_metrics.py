import random
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from prometheus_client import start_http_server, Gauge
mse_gauge = Gauge('advertising_model_mse', 'Mean Squared Error for Advertising Dataset')
rmse_gauge = Gauge('advertising_model_rmse', 'Root Mean Squared Error for Advertising Dataset')
r2_gauge = Gauge('advertising_model_r2', 'R Squared for Advertising Dataset')

start_http_server(8000)
while True:
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    X_train = X[:80]
    X_test = X[80:]
    y_train = y[:80]
    y_test = y[80:]

    model = LinearRegression()
    model.fit(X_train, y_train)

   
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

   
    mse_gauge.set(mse)
    rmse_gauge.set(rmse)
    r2_gauge.set(r2)

    
    print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

    
    time.sleep(5)
