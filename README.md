Advanced Time Series Forecasting on the AirPassengers Dataset
SARIMAX • N-BEATS (PyTorch) • Bayesian Optimization (Optuna) • Rolling-Origin CV

This project performs a full end-to-end time series forecasting pipeline on the classic AirPassengers dataset (monthly airline passenger counts from 1949–1960).
It includes EDA, baseline models, statistical forecasting, deep learning forecasting, and Bayesian hyperparameter optimization, with professional evaluation and model comparison.

Project Overview

This repository implements an advanced forecasting framework consisting of:

Exploratory Data Analysis (EDA)

seasonal decomposition (trend, seasonal, residual)

stationarity testing (ADF unit root test)

analysis of autocorrelation (ACF) and partial autocorrelation (PACF)

summary insights on trend and seasonality

Baseline Model

A Seasonal Naive model is used as a benchmark by repeating last year's monthly values.

SARIMAX Forecasting (Classical Statistical Model)

A seasonal ARIMA model with:

exogenous regressors

monthly dummies

lag-1 and lag-12 features

time trend

SARIMAX parameters are tuned with Bayesian optimization using Optuna, applying rolling-origin cross-validation.

N-BEATS Deep Learning Forecasting (PyTorch)

A compact and clean implementation of the N-BEATS architecture:

univariate forecasting

configurable hidden size, layers, learning rate, batch size, epochs

trained on sliding windows

optimized using Optuna

produces multi-step monthly forecasts for the test period

Hyperparameter Optimization (Optuna)

Both SARIMAX and N-BEATS are optimized using:

TPE sampler (Bayesian optimization)

Rolling-origin cross-validation

RMSE as the objective metric

Final Model Comparison

Models evaluated on the 24-month test set using:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

SMAPE (Symmetric Mean Absolute Percentage Error)

A Markdown table is automatically generated summarizing:

each model

its metrics

its selected hyperparameters

Outputs Saved

The script produces:

airpassengers_forecasts.csv (actual vs forecasts)

nbeats_model.pth (final N-BEATS model weights)

a Markdown model comparison table

a concise analysis report

 Dataset Description

AirPassengers Dataset

Monthly U.S. international airline passengers (1949–1960)

Strong multiplicative yearly seasonality

Positive trend

Increasing variance

Non-stationary (ADF test confirms)

This classic dataset is widely used to evaluate time series models.

 Methodology Summary
1. Data Loading

The dataset is loaded from statsmodels or from a fallback array if unavailable.

2. Exploratory Data Analysis

Performed automatically:

decomposition

descriptive statistics

ADF test

ACF/PACF inspection

3. Train/Test Split

The last 24 months are reserved as the test set.

4. Baseline Model

Seasonal naive forecast using values from the previous year.

5. Feature Engineering for SARIMAX

Includes:

month dummies

lag-1 and lag-12 features

time trend

alignment of exogenous features with the target

6. SARIMAX Model + Hyperparameter Optimization

Parameters searched:

ARIMA: p,d,q

Seasonal: P,D,Q, with s=12

Rolling-origin CV

Objective: RMSE

7. N-BEATS Deep Learning Model

Includes:

custom PyTorch implementation

multi-layer fully connected blocks

backcast and forecast heads

training on sliding input windows

8. N-BEATS Hyperparameter Optimization

Search space includes:

hidden size

number of layers

learning rate

batch size

number of epochs

9. Final Training & Forecast

Best Hyperparameters → Train Full Model → Forecast Test Set.

10. Evaluation

For each model:

compute RMSE

compute MAE

compute SMAPE

produce comparison table

11. Saving Results

forecast CSV

N-BEATS model weights

markdown table

analysis summary

 Models Compared
Baseline: Seasonal Naive

Simple and surprisingly strong on seasonal data.

SARIMAX (Statistical Model)

Captures:

trend

seasonality

lagged effects

exogenous month indicators

Advantages:

interpretable

statistically grounded

N-BEATS (Deep Learning)

Captures:

nonlinear patterns

long-term trend interactions

seasonality without explicit feature engineering

Advantages:

highly flexible

good at long forecasting horizons

 Output Examples
1. Model Comparison Table (auto-generated)

(Actual values depend on training)

Model	RMSE	MAE	SMAPE (%)	Parameters
SeasonalNaive	...	...	...	{}
SARIMAX	...	...	...	{p,d,q,P,D,Q}
N-BEATS	...	...	...	{hidden_size, n_layers, lr, batch_size, epochs}
2. Files Produced

airpassengers_forecasts.csv — actuals and predictions

nbeats_model.pth — N-BEATS trained weights

 Analysis Report Summary

The report generated includes:

Dataset Summary

length

date range

trend and seasonality characteristics

EDA Summary

ADF p-value

decomposition insights

autocorrelation structure

Optimization Summary

SARIMAX search space

N-BEATS search space

number of trials for each model

Test-Set Performance

RMSE / MAE / SMAPE of all models

Discussion

strengths and weaknesses of each model

seasonal naive as a benchmark

SARIMAX interpretability

N-BEATS flexibility

impact of hyperparameter tuning
