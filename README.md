# Advertising Sales Prediction with Monitoring and Sustainability  

##  Project Overview  
This project explores machine learning model development and monitoring using **Linear Regression**.  
The main task was to predict product sales based on advertising budgets (TV, Radio, Newspaper).  
Beyond building the model, I also integrated **Prometheus** and **Grafana** for real-time monitoring,  
and used **CodeCarbon** to track the environmental impact of training the models.  

##  Objectives  
- Build and train a Linear Regression model on the Advertising dataset.  
- Evaluate performance using MSE, RMSE, and R².  
- Integrate Prometheus for exposing performance metrics.  
- Use Grafana for dashboard visualization.  
- Track carbon emissions using CodeCarbon.  
- Experiment with synthetic data and simulated metrics.  

##  Repository Structure 
├── code1_basic_regression.py # Basic Linear Regression with Advertising dataset
├── code2_synthetic_data.py # Regression with synthetic data
├── code3_prometheus_monitoring.py # Linear Regression + Prometheus metrics
├── code4_codecarbon.py # Linear Regression + CodeCarbon integration
├── code5_simulated_metrics.py # Randomized metrics simulation for testing
├── README.md # Project documentation


Also install:

Prometheus → [Download here](https://prometheus.io/download/)

Grafana → [Download here](https://grafana.com/grafana/download)

🚀 How to Run
1. Run the ML Model

Example for basic regression:


2. Start Prometheus

Make sure your prometheus.yml is configured like this:


3. Connect Prometheus to Grafana

Open Grafana at http://localhost:3000

Add Prometheus as a data source (http://localhost:9090)

Import a dashboard JSON or create panels for:

advertising_model_mse

advertising_model_rmse

advertising_model_r2

advertising_model_carbon_emissions

4. View Dashboard

See metrics update in real-time.

<img width="1098" height="610" alt="image" src="https://github.com/user-attachments/assets/d8dfb4d4-f985-48a1-ba97-ec9d619dde6f" />

