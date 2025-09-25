import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from prometheus_client import start_http_server, Gauge
from codecarbon import EmissionsTracker
mse_gauge = Gauge('advertising_model_mse', 'Mean Squared Error for Advertising Dataset')
rmse_gauge = Gauge('advertising_model_rmse', 'Root Mean Squared Error for Advertising Dataset')
r2_gauge = Gauge('advertising_model_r2', 'R Squared for Advertising Dataset')
carbon_gauge = Gauge('advertising_model_carbon_emissions', 'Carbon Emissions (kg CO₂eq)')
start_http_server(8000)

cumulative_emissions = 0.0  

while True:
    tracker = EmissionsTracker(measure_power_secs=1, output_dir=".", log_level="error")
    tracker.start()

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

    emissions = tracker.stop()
    cumulative_emissions += emissions
    mse_gauge.set(mse)
    rmse_gauge.set(rmse)
    r2_gauge.set(r2)
    carbon_gauge.set(cumulative_emissions)

    print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}, Cumulative CO₂: {cumulative_emissions:.6f} kg")

    time.sleep(5)
