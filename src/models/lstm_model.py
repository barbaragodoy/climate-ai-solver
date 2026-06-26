"""Módulo LSTM para previsão de séries temporais climáticas."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


def build_model(n_steps: int, units_1: int = 64, units_2: int = 32, dropout: float = 0.2):
    model = Sequential([
        LSTM(units_1, return_sequences=True, input_shape=(n_steps, 1)),
        BatchNormalization(),
        Dropout(dropout),
        LSTM(units_2, return_sequences=False),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mae'])
    return model


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_steps: int,
    checkpoint_path: str = 'outputs/models/lstm_best.keras',
    epochs: int = 100,
    batch_size: int = 64,
):
    model = build_model(n_steps)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


def evaluate(model, scaler: MinMaxScaler, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred_s = model.predict(X_test, verbose=0).flatten()
    y_pred = scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
    y_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    return {
        'MAE':  mean_absolute_error(y_real, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_real, y_pred)),
        'R2':   r2_score(y_real, y_pred),
        'y_pred': y_pred,
        'y_real': y_real,
    }
