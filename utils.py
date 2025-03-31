import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

scaler = MinMaxScaler()

def prepare_input(df):
    # Use 9 features (exclude 'Date', maybe exclude 'Close' if you want to predict it)
    features = df[['Open', 'High', 'Low', 'Volume', 'Adj Close', 'Some_TA1', 'Some_TA2', 'Some_TA3', 'Some_TA4']]
    
    # Scale
    scaled = scaler.fit_transform(features)

    # Lookback window = 3 timesteps
    X = []
    for i in range(3, len(scaled)):
        X.append(scaled[i-3:i])
    
    return np.array(X), scaler
