import time
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from prometheus_client import start_http_server, Gauge
from codecarbon import EmissionsTracker

mse_gauge = Gauge('advertising_model_mse', 'MSE for Advertising Dataset')
rmse_gauge = Gauge('advertising_model_rmse', 'RMSE for Advertising Dataset')
r2_gauge = Gauge('advertising_model_r2', 'R2 for Advertising Dataset')
carbon_gauge = Gauge('advertising_model_carbon_emissions', 'Carbon Emissions')

start_http_server(8000)

data = pd.read_csv("C:/Users/yeshw/Downloads/archive/advertising.csv")

X_full = data[['TV', 'Radio', 'Newspaper']].values
y_full = data['Sales'].values

cumulative_emissions = 0.0

while True:
    indices = np.random.permutation(len(X_full))
    X = X_full[indices]
    y = y_full[indices]

    X_train = X[:int(0.8 * len(X))]
    X_test = X[int(0.8 * len(X)):]
    y_train = y[:int(0.8 * len(y))]
    y_test = y[int(0.8 * len(y)):]

    tracker = EmissionsTracker(measure_power_secs=1, output_dir=".", log_level="error")
    tracker.start()

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    emissions = tracker.stop()
    cumulative_emissions += emissions

    mse_gauge.set(mse)
    rmse_gauge.set(rmse)
    r2_gauge.set(r2)
    carbon_gauge.set(cumulative_emissions)

    print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}, CO₂: {cumulative_emissions:.6f} kg")

    time.sleep(5)
