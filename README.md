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

Install dependencies using pip:
pip install pandas numpy matplotlib scikit-learn prometheus-client codecarbon

Also install:

Prometheus → [Download here](https://prometheus.io/download/)

Grafana → [Download here](https://grafana.com/grafana/download)

🚀 How to Run
1. Run the ML Model

Example for basic regression:

python code1_basic_regression.py

2. Start Prometheus

Make sure your prometheus.yml is configured like this:
yaml
scrape_configs:
  - job_name: "advertising_model"
    static_configs:
      - targets: ["localhost:8000"]

Run Prometheus
prometheus --config.file=prometheus.yml

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

Example Outputs

Scatter plot of Actual vs Predicted Sales
<img width="793" height="685" alt="Screenshot 2025-09-25 142807" src="https://github.com/user-attachments/assets/22c0f354-ef47-4ac0-9c79-5c48dd4b8935" />


Prometheus metrics endpoint
<img width="1099" height="610" alt="Screenshot 2025-09-26 172835" src="https://github.com/user-attachments/assets/0fb7352a-8b15-4de5-97c8-c9da9dd98759" />

Grafana dashboards for MSE, RMSE, R², CO₂ emissions
<img width="1078" height="613" alt="Screenshot 2025-09-26 172709" src="https://github.com/user-attachments/assets/2d971061-a980-4ae3-988b-4632d2a9acdf" />



