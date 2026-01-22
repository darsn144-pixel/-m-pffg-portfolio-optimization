"""
Data Processing Module
Research: Integration of Artificial Intelligence with M-Polar Fermatean Fuzzy Graphs
Authors: Kholood Alsager, Nada Almuairi

This module handles data collection, preprocessing, and normalization for the research.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class StockDataProcessor:
    """
    Process and prepare stock data for analysis
    """
    
    def __init__(self, stocks=['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']):
        """
        Initialize data processor
        
        Parameters:
        -----------
        stocks : list
            List of stock ticker symbols
        """
        self.stocks = stocks
        self.raw_data = {}
        self.processed_data = {}
        self.scaler = MinMaxScaler()
    
    def fetch_stock_data(self, start_date='2024-09-01', end_date='2024-10-31'):
        """
        Fetch stock data from Yahoo Finance
        
        Parameters:
        -----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        
        Returns:
        --------
        data_dict : dict
            Dictionary of dataframes for each stock
        """
        print(f"\n{'='*70}")
        print(f"Fetching Stock Data from Yahoo Finance")
        print(f"Period: {start_date} to {end_date}")
        print(f"{'='*70}\n")
        
        for stock in self.stocks:
            try:
                print(f"Downloading {stock}...", end=' ')
                
                # Download data
                ticker = yf.Ticker(stock)
                df = ticker.history(start=start_date, end=end_date)
                
                # Keep relevant columns
                df = df[['Close', 'Volume']]
                df['Stock'] = stock
                
                self.raw_data[stock] = df
                print(f"✓ Downloaded {len(df)} days")
                
            except Exception as e:
                print(f"✗ Error: {e}")
        
        print(f"\n✓ Data collection complete for {len(self.raw_data)} stocks")
        return self.raw_data
    
    def calculate_returns(self, df, method='simple'):
        """
        Calculate stock returns
        
        Parameters:
        -----------
        df : pd.DataFrame
            Stock data with 'Close' prices
        method : str
            'simple' or 'log' returns
        
        Returns:
        --------
        df : pd.DataFrame
            DataFrame with returns column added
        """
        if method == 'simple':
            df['Return'] = df['Close'].pct_change() * 100  # Percentage
        elif method == 'log':
            df['Return'] = np.log(df['Close'] / df['Close'].shift(1)) * 100
        
        return df
    
    def calculate_volatility(self, df, window=5):
        """
        Calculate rolling volatility (standard deviation of returns)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Stock data with 'Return' column
        window : int
            Rolling window size
        
        Returns:
        --------
        df : pd.DataFrame
            DataFrame with volatility column added
        """
        df['Volatility'] = df['Return'].rolling(window=window).std()
        return df
    
    def normalize_volume(self, df):
        """
        Normalize trading volume
        
        Parameters:
        -----------
        df : pd.DataFrame
            Stock data with 'Volume' column
        
        Returns:
        --------
        df : pd.DataFrame
            DataFrame with normalized volume
        """
        df['Volume_Normalized'] = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()
        return df
    
    def process_all_stocks(self, calculate_features=True):
        """
        Process all stocks: calculate returns, volatility, normalize
        
        Parameters:
        -----------
        calculate_features : bool
            Whether to calculate additional features
        
        Returns:
        --------
        processed_data : dict
            Dictionary of processed dataframes
        """
        print(f"\n{'='*70}")
        print(f"Processing Stock Data")
        print(f"{'='*70}\n")
        
        for stock, df in self.raw_data.items():
            print(f"Processing {stock}...", end=' ')
            
            # Calculate returns
            df = self.calculate_returns(df, method='simple')
            
            if calculate_features:
                # Calculate volatility
                df = self.calculate_volatility(df, window=5)
                
                # Normalize volume
                df = self.normalize_volume(df)
                
                # Calculate moving averages
                df['MA_5'] = df['Close'].rolling(window=5).mean()
                df['MA_10'] = df['Close'].rolling(window=10).mean()
            
            # Remove NaN values
            df = df.dropna()
            
            self.processed_data[stock] = df
            print(f"✓ Processed {len(df)} rows")
        
        print(f"\n✓ Processing complete for {len(self.processed_data)} stocks")
        return self.processed_data
    
    def calculate_summary_statistics(self):
        """
        Calculate summary statistics for each stock
        Reproduces Table 22 from the research paper
        
        Returns:
        --------
        summary_df : pd.DataFrame
            Summary statistics table
        """
        summary_data = []
        
        for stock, df in self.processed_data.items():
            summary = {
                'Stock': stock,
                'Closing_Price_Oct31': df['Close'].iloc[-1],
                'Avg_Return_%': df['Return'].mean(),
                'Avg_Volatility_%': df['Volatility'].mean(),
                'Volume_M': df['Volume'].mean() / 1e6
            }
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        
        print(f"\n{'='*70}")
        print(f"Stock Data Summary (October 2024)")
        print(f"{'='*70}")
        print(summary_df.to_string(index=False))
        print(f"{'='*70}\n")
        
        return summary_df
    
    def export_processed_data(self, output_dir='data/processed/'):
        """
        Export processed data to CSV files
        
        Parameters:
        -----------
        output_dir : str
            Output directory path
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export individual stock data
        for stock, df in self.processed_data.items():
            filepath = f"{output_dir}{stock}_processed.csv"
            df.to_csv(filepath)
            print(f"✓ Exported: {filepath}")
        
        # Export summary statistics
        summary = self.calculate_summary_statistics()
        summary.to_csv(f"{output_dir}summary_statistics.csv", index=False)
        print(f"✓ Exported: {output_dir}summary_statistics.csv")
        
        # Export combined data
        combined_df = pd.concat(self.processed_data.values(), keys=self.processed_data.keys())
        combined_df.to_csv(f"{output_dir}all_stocks_combined.csv")
        print(f"✓ Exported: {output_dir}all_stocks_combined.csv")


def calculate_fuzzy_values(summary_df):
    """
    Calculate membership and non-membership degrees for M-PFFG
    Reproduces Table 23 from the research paper
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Summary statistics dataframe
    
    Returns:
    --------
    fuzzy_df : pd.DataFrame
        Fuzzy values (membership and non-membership degrees)
    """
    print(f"\n{'='*70}")
    print(f"Calculating Fuzzy Values (M-Polar Fermatean)")
    print(f"{'='*70}\n")
    
    fuzzy_data = []
    
    for _, row in summary_df.iterrows():
        stock = row['Stock']
        volatility = row['Avg_Volatility_%'] / 100  # Convert to decimal
        volume = row['Volume_M']
        
        # Polarity 1: Volatility-based
        phi1 = 1 / np.sqrt(volatility) if volatility > 0 else 1.0
        phi1 = min(phi1, 1.0)  # Cap at 1.0
        psi1 = np.sqrt(1 - phi1**2)
        
        # Polarity 2: Volume-based
        volume_normalized = volume / summary_df['Volume_M'].max()
        phi2 = np.sqrt(volume_normalized)
        psi2 = np.sqrt(1 - phi2**2)
        
        # Polarity 3: Sentiment (placeholder - replace with actual sentiment)
        # For now, use a combination of return and volatility
        sentiment_score = row['Avg_Return_%'] / row['Avg_Volatility_%']
        sentiment_score = (sentiment_score + 1) / 2  # Normalize to [0, 1]
        phi3 = min(sentiment_score, 1.0)
        psi3 = np.sqrt(1 - phi3**2)
        
        fuzzy_data.append({
            'Stock': stock,
            'φ1': round(phi1, 2),
            'ψ1': round(psi1, 2),
            'φ2': round(phi2, 2),
            'ψ2': round(psi2, 2),
            'φ3': round(phi3, 2),
            'ψ3': round(psi3, 2)
        })
    
    fuzzy_df = pd.DataFrame(fuzzy_data)
    
    print("3-Polar Fuzzy Values (October 2024):")
    print(fuzzy_df.to_string(index=False))
    print(f"{'='*70}\n")
    
    return fuzzy_df


def load_example_data():
    """
    Load example data based on Table 22 from the research paper
    Use this if Yahoo Finance data is not available
    
    Returns:
    --------
    summary_df : pd.DataFrame
        Example summary statistics
    """
    data = {
        'Stock': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA'],
        'Closing_Price_Oct31': [222.56, 406.35, 186.40, 171.11, 43.75],
        'Avg_Return_%': [0.042, 0.035, 0.055, 0.032, 0.095],
        'Avg_Volatility_%': [1.25, 1.18, 1.45, 1.10, 2.10],
        'Volume_M': [58.3, 2.1, 38.9, 25.6, 96.2]
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    """
    Main execution: Fetch, process, and export stock data
    """
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║        Stock Data Processing Pipeline                     ║
    ║    Research: M-Polar Fermatean Fuzzy Graphs              ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    STOCKS = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    START_DATE = '2024-09-01'
    END_DATE = '2024-10-31'
    
    # Initialize processor
    processor = StockDataProcessor(stocks=STOCKS)
    
    # Option 1: Fetch real data from Yahoo Finance
    try:
        print("\nAttempting to fetch data from Yahoo Finance...")
        processor.fetch_stock_data(start_date=START_DATE, end_date=END_DATE)
        processor.process_all_stocks(calculate_features=True)
        summary_df = processor.calculate_summary_statistics()
        
    except Exception as e:
        print(f"\n⚠ Could not fetch data from Yahoo Finance: {e}")
        print("Using example data from research paper instead...")
        summary_df = load_example_data()
    
    # Calculate fuzzy values (Table 23)
    fuzzy_df = calculate_fuzzy_values(summary_df)
    
    # Export data
    if processor.processed_data:
        processor.export_processed_data(output_dir='../data/processed/')
    
    # Export fuzzy values
    fuzzy_df.to_csv('../data/processed/fuzzy_values_table23.csv', index=False)
    print(f"✓ Exported fuzzy values to: data/processed/fuzzy_values_table23.csv")
    
    # Export summary
    summary_df.to_csv('../data/processed/stock_summary_table22.csv', index=False)
    print(f"✓ Exported summary to: data/processed/stock_summary_table22.csv")
    
    print("\n✓ Data processing pipeline complete!")
