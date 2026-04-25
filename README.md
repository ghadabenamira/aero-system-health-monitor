# RUL Prediction of Turbofan Engine
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-Regression-orange"/>
  <img src="https://img.shields.io/badge/Dataset-NASA%20CMAPSS-red"/>
  <img src="https://img.shields.io/badge/Scikit--learn-Used-f7931e?logo=scikitlearn"/>
  <img src="https://img.shields.io/badge/Status-Completed-success"/>
</p>

## Overview
This project focuses on predicting the **Remaining Useful Life (RUL)** of turbofan engines using the **NASA CMAPSS FD001 dataset**. The objective is to model engine degradation over time and estimate how many operational cycles remain before failure.

The problem is formulated as a **supervised regression task**, where machine learning models learn patterns of degradation from multivariate time-series sensor data collected across engine life cycles. These learned patterns are then used to predict the RUL of unseen engines, enabling predictive maintenance and improved operational safety.

---
## Objectives
- Predict the Remaining Useful Life (RUL) of aircraft engines
- Analyze sensor-based degradation patterns over time
- Develop a data-driven predictive maintenance solution
  
---
## Dataset
### NASA CMAPSS FD001

The dataset simulates run-to-failure behavior of turbofan engines under varying operational conditions.

### Key Characteristics:
- Multiple engine units
- Time-series operational cycles
- 3 operational settings
- 21 sensor measurements
- Run-to-failure trajectories

 Source: NASA Prognostics Data Repository

---
##  Problem Formulation

- **Input:** Multivariate time-series sensor readings per engine cycle  
- **Output:** Remaining Useful Life (RUL) value per engine  
- **Type:** Supervised regression problem  

---
## Methodology

### 1. Data Preprocessing
- Cleaning and structuring time-series data
- Computing RUL labels
- Feature scaling (Normalization / Standardization)

### 2. Exploratory Data Analysis (EDA)
- Sensor trend visualization
- Correlation analysis
- Degradation behavior observation

### 3. Feature Engineering
- Selection of relevant sensors
- Smoothing noisy signals
- Time-based transformations
### 4. Modeling
Models explored include:
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting (XGBoost / LightGBM)

### 5. Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Custom scoring functions for RUL prediction

---

##  Results

- Machine learning models successfully capture degradation trends.
- Tree-based ensemble models (e.g., Random Forest / Gradient Boosting) outperform baseline models.
- Feature engineering significantly improves prediction accuracy.

---

##  Project Structure


├── data/ # NASA CMAPSS dataset
├── notebooks/ # EDA and experiments
├── src/ # Source code (preprocessing, modeling)
├── models/ # Saved trained models
├── results/ # Visualizations and evaluation outputs
└── README.md


---

##  Tech Stack

- Python 
- NumPy / Pandas
- Scikit-learn
- Matplotlib / Seaborn
- XGBoost / LightGBM
- Jupyter Notebook

---

##  Future Improvements

- Implement deep learning models (LSTM / GRU for time-series)
- Hyperparameter optimization (Optuna / GridSearch)
- Advanced feature selection techniques
- Model deployment using Flask or FastAPI
- Real-time engine health monitoring dashboard

---

##  Authors

- Ghada Ben Amira
- Hiba Chaabouni
- Ali Ayari
- Maysa Hammami

---

##  License

This project is intended for academic and educational purposes.

---

##  Acknowledgements

- NASA CMAPSS Dataset
- Course instructors and project mentors
