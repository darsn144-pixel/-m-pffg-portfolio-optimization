"""
LSTM Model for Stock Return Prediction
Research: Integration of Artificial Intelligence with M-Polar Fermatean Fuzzy Graphs
Author: Kholood Alsager, Nada Almuairi

This module implements the LSTM model used to predict stock returns for October 2024
based on September 2024 training data.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)


class StockLSTMPredictor:
    """
    LSTM-based model for predicting stock returns
    """
    
    def __init__(self, lookback_period=30, random_state=42):
        """
        Initialize LSTM predictor
        
        Parameters:
        -----------
        lookback_period : int
            Number of previous days to use for prediction (default: 30)
        random_state : int
            Random seed for reproducibility
        """
        self.lookback_period = lookback_period
        self.random_state = random_state
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        
        # Set seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def build_model(self, input_shape, lstm_units=[64, 32], dropout_rate=0.2):
        """
        Build LSTM model architecture
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (lookback_period, n_features)
        lstm_units : list
            Number of units in each LSTM layer
        dropout_rate : float
            Dropout rate for regularization
        
        Returns:
        --------
        model : Sequential
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer with return sequences
        model.add(LSTM(
            units=lstm_units[0],
            return_sequences=True,
            input_shape=input_shape,
            activation='tanh',
            name='lstm_layer_1'
        ))
        model.add(Dropout(dropout_rate, name='dropout_1'))
        
        # Second LSTM layer
        model.add(LSTM(
            units=lstm_units[1],
            return_sequences=False,
            activation='tanh',
            name='lstm_layer_2'
        ))
        model.add(Dropout(dropout_rate, name='dropout_2'))
        
        # Output layer
        model.add(Dense(units=1, activation='linear', name='output_layer'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, data, target_column='Return'):
        """
        Prepare time series data for LSTM
        
        Parameters:
        -----------
        data : pd.DataFrame
            Stock data with features and target
        target_column : str
            Name of target column (return)
        
        Returns:
        --------
        X : np.array
            Input sequences
        y : np.array
            Target values
        """
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.values)
        
        X, y = [], []
        
        # Create sequences
        for i in range(self.lookback_period, len(scaled_data)):
            X.append(scaled_data[i-self.lookback_period:i])
            
            # Target is the return value at current time
            target_idx = data.columns.get_loc(target_column)
            y.append(scaled_data[i, target_idx])
        
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, verbose=1):
        """
        Train LSTM model
        
        Parameters:
        -----------
        X_train : np.array
            Training sequences
        y_train : np.array
            Training targets
        X_val : np.array, optional
            Validation sequences
        y_val : np.array, optional
            Validation targets
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        verbose : int
            Verbosity level
        
        Returns:
        --------
        history : History
            Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_lstm_model.h5',
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train model
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : np.array
            Input sequences
        
        Returns:
        --------
        predictions : np.array
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        
        # Inverse transform predictions to original scale
        # Create dummy array for inverse transform
        dummy = np.zeros((len(predictions), self.scaler.n_features_in_))
        dummy[:, 0] = predictions.flatten()
        predictions_original = self.scaler.inverse_transform(dummy)[:, 0]
        
        return predictions_original
    
    def plot_training_history(self, save_path='lstm_training_history.png'):
        """
        Plot training history
        
        Parameters:
        -----------
        save_path : str
            Path to save plot
        """
        if self.history is None:
            raise ValueError("No training history available.")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Model Loss During Training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE plot
        axes[1].plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].set_title('Model MAE During Training')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.close()
    
    def save_model(self, filepath='lstm_stock_predictor.h5'):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='lstm_stock_predictor.h5'):
        """Load trained model"""
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")


def load_and_prepare_stock_data(stock_symbol, start_date='2024-09-01', end_date='2024-09-30'):
    """
    Load and prepare stock data
    
    Parameters:
    -----------
    stock_symbol : str
        Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    start_date : str
        Start date for training data
    end_date : str
        End date for training data
    
    Returns:
    --------
    df : pd.DataFrame
        Prepared stock data with features
    """
    # This is a placeholder - replace with actual data loading
    # In practice, you would load from Yahoo Finance or your CSV files
    
    print(f"Loading data for {stock_symbol} from {start_date} to {end_date}")
    
    # Example structure - replace with actual data
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Simulated data structure based on Table 22 in the paper
    df = pd.DataFrame({
        'Date': dates,
        'Close': np.random.randn(len(dates)).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
        'Return': np.random.randn(len(dates)) * 0.02
    })
    
    # Calculate additional features
    df['Volatility'] = df['Return'].rolling(window=5).std()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    
    # Remove NaN values
    df = df.dropna()
    
    return df


def predict_october_returns(stocks=['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']):
    """
    Predict October 2024 returns for given stocks using LSTM
    Based on Table 24 in the research paper
    
    Parameters:
    -----------
    stocks : list
        List of stock symbols
    
    Returns:
    --------
    predictions_df : pd.DataFrame
        DataFrame with predicted returns
    """
    results = []
    
    for stock in stocks:
        print(f"\n{'='*60}")
        print(f"Training LSTM for {stock}")
        print(f"{'='*60}")
        
        # Load September 2024 data for training
        df = load_and_prepare_stock_data(stock, '2024-09-01', '2024-09-30')
        
        # Prepare features
        feature_columns = ['Close', 'Volume', 'Return', 'Volatility']
        df_features = df[feature_columns]
        
        # Initialize predictor
        predictor = StockLSTMPredictor(lookback_period=20, random_state=42)
        
        # Prepare data
        X, y = predictor.prepare_data(df_features, target_column='Return')
        
        # Split into train and validation (80-20)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        predictor.build_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            lstm_units=[64, 32],
            dropout_rate=0.2
        )
        
        print(f"\nModel Architecture:")
        predictor.model.summary()
        
        # Train model
        print(f"\nTraining model...")
        predictor.train(
            X_train, y_train,
            X_val, y_val,
            epochs=100,
            batch_size=32,
            verbose=1
        )
        
        # Plot training history
        predictor.plot_training_history(f'training_history_{stock}.png')
        
        # Make prediction for October
        # Use last sequence from September to predict October
        last_sequence = X[-1:]
        predicted_return = predictor.predict(last_sequence)[0]
        
        # Save model
        predictor.save_model(f'lstm_model_{stock}.h5')
        
        results.append({
            'Stock': stock,
            'Predicted_Return_%': round(predicted_return, 2)
        })
        
        print(f"\n{stock} - Predicted October Return: {predicted_return:.2f}%")
    
    predictions_df = pd.DataFrame(results)
    predictions_df.to_csv('lstm_predicted_returns_october2024.csv', index=False)
    
    print(f"\n{'='*60}")
    print("LSTM Predictions Complete")
    print(f"{'='*60}")
    print(predictions_df)
    
    return predictions_df


if __name__ == "__main__":
    """
    Main execution: Train LSTM and predict October 2024 returns
    Reproduces Table 24 from the research paper
    """
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║    LSTM Model for Stock Return Prediction                 ║
    ║    Research: M-Polar Fermatean Fuzzy Graphs              ║
    ║    Training Period: September 2024                        ║
    ║    Prediction Period: October 2024                        ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Model hyperparameters (as used in the research)
    HYPERPARAMETERS = {
        'lookback_period': 20,
        'lstm_units': [64, 32],
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32,
        'random_seed': 42
    }
    
    print("\nModel Hyperparameters:")
    print("-" * 60)
    for key, value in HYPERPARAMETERS.items():
        print(f"  {key:.<40} {value}")
    print("-" * 60)
    
    # Stocks to predict (from Table 22)
    stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    
    # Run predictions
    predictions = predict_october_returns(stocks)
    
    print("\n✓ All models trained and predictions saved!")
    print("\nGenerated files:")
    print("  - lstm_predicted_returns_october2024.csv")
    print("  - lstm_model_*.h5 (for each stock)")
    print("  - training_history_*.png (for each stock)")
