import numpy as np
import pandas as pd
import yfinance as yf
import talib
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from utils import get_stock_data, add_technical_indicators, \
prepare_data, hyperparameter_tuning, train_and_evaluate


if __name__ == '__main__':
    # Execute the Pipeline
    data = get_stock_data('AAPL', '2020-01-01', '2024-01-01')
    data = add_technical_indicators(data)

    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }

    X_train, X_test, y_train, y_test = prepare_data(data)
    train_and_evaluate(X_train, X_test, y_train, y_test, models)