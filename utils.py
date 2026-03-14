import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def calculate_technical_indicators(data):
    """
    Calculate technical indicators: RSI, MACD, Bollinger Bands
    
    Parameters:
    data: DataFrame with 'Close' column
    
    Returns:
    DataFrame with added indicator columns
    """
    df = data.copy()
    
    # RSI (Relative Strength Index) - 14 period
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger Bands (20-day, 2 std dev)
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma20 + (std20 * 2)
    df['BB_Lower'] = sma20 - (std20 * 2)
    df['BB_Middle'] = sma20
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    return df


def calculate_momentum_features(data):
    """
    Calculate price momentum features: 5-day, 10-day, 21-day % changes
    
    Parameters:
    data: DataFrame with 'Close' column
    
    Returns:
    DataFrame with momentum features
    """
    df = data.copy()
    
    # Price momentum (percentage change)
    df['Mom_5d'] = df['Close'].pct_change(5)      # 5-day momentum
    df['Mom_10d'] = df['Close'].pct_change(10)    # 10-day momentum
    df['Mom_21d'] = df['Close'].pct_change(21)    # 21-day momentum
    
    return df


def calculate_volatility_features(data):
    """
    Calculate volatility features: 21-day rolling std dev
    
    Parameters:
    data: DataFrame with 'Close' column
    
    Returns:
    DataFrame with volatility features
    """
    df = data.copy()
    
    # 21-day rolling volatility
    df['Volatility_21d'] = df['Close'].pct_change().rolling(window=21).std()
    
    return df


def calculate_volume_features(data):
    """
    Calculate volume features: volume moving average ratios
    
    Parameters:
    data: DataFrame with 'Volume' column
    
    Returns:
    DataFrame with volume features
    """
    df = data.copy()
    
    # Volume moving averages
    df['Vol_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_MA_50'] = df['Volume'].rolling(window=50).mean()
    df['Vol_Ratio_20'] = df['Volume'] / df['Vol_MA_20']
    df['Vol_Ratio_50'] = df['Volume'] / df['Vol_MA_50']
    
    return df


def engineer_features(stock_data):
    """
    Complete feature engineering pipeline
    
    Parameters:
    stock_data: Dictionary with stock symbols as keys and DataFrames as values
    
    Returns:
    Dictionary with engineered features
    """
    features = {}
    
    for symbol, df in stock_data.items():
        df_processed = df.copy()
        df_processed = calculate_momentum_features(df_processed)
        df_processed = calculate_technical_indicators(df_processed)
        df_processed = calculate_volatility_features(df_processed)
        df_processed = calculate_volume_features(df_processed)
        features[symbol] = df_processed
    
    return features


def create_weekly_target(stock_data):
    """
    Create binary target based on next week's return
    
    Parameters:
    stock_data: Dictionary with stock symbols as keys and DataFrames as values
    
    Returns:
    Dictionary with targets
    """
    targets = {}
    
    for symbol, df in stock_data.items():
        df_copy = df.copy()
        # Calculate next week's return (Friday to Friday)
        # Weekly return: (Close[t+5] - Close[t]) / Close[t]
        df_copy['Weekly_Return'] = df_copy['Close'].pct_change(5).shift(-5)
        # Target: 1 if next week's return > 0, else 0
        df_copy['Target'] = (df_copy['Weekly_Return'] > 0).astype(int)
        targets[symbol] = df_copy[['Weekly_Return', 'Target']]
    
    return targets


def prepare_ml_data(features_dict, targets_dict, feature_columns):
    """
    Prepare data for machine learning
    
    Parameters:
    features_dict: Dictionary with engineered features
    targets_dict: Dictionary with targets
    feature_columns: List of feature column names to use
    
    Returns:
    X (features), y (targets), and symbol-date index
    """
    all_X = []
    all_y = []
    all_dates = []
    all_symbols = []
    
    for symbol in features_dict.keys():
        X = features_dict[symbol][feature_columns].copy()
        y = targets_dict[symbol]['Target'].copy()
        
        # Remove rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        
        X = X[mask]
        y = y[mask]
        dates = X.index[mask]
        
        all_X.append(X)
        all_y.append(y)
        all_dates.extend(dates)
        all_symbols.extend([symbol] * len(X))
    
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)
    
    return X_combined, y_combined, all_dates, all_symbols


def calculate_portfolio_metrics(portfolio_returns, risk_free_rate=0.02):
    """
    Calculate key performance metrics
    
    Parameters:
    portfolio_returns: Series of weekly returns
    risk_free_rate: Annual risk-free rate (default 2%)
    
    Returns:
    Dictionary with metrics
    """
    # Cumulative return
    cumulative_return = (1 + portfolio_returns).prod() - 1
    
    # Annualized return
    weeks_per_year = 52
    annualized_return = (1 + portfolio_returns.mean() * weeks_per_year) ** (52/len(portfolio_returns)) - 1
    
    # Annualized volatility
    annualized_volatility = portfolio_returns.std() * np.sqrt(52)
    
    # Sharpe ratio
    if annualized_volatility != 0:
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    else:
        sharpe_ratio = 0
    
    # Max drawdown
    cumulative_return_series = (1 + portfolio_returns).cumprod()
    running_max = cumulative_return_series.expanding().max()
    drawdown = (cumulative_return_series - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'Cumulative_Return': cumulative_return,
        'Annualized_Return': annualized_return,
        'Annualized_Volatility': annualized_volatility,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown
    }


def calculate_hit_rate(predictions, actuals):
    """
    Calculate hit rate (percentage of correct predictions)
    
    Parameters:
    predictions: Array of predicted labels (0 or 1)
    actuals: Array of actual labels (0 or 1)
    
    Returns:
    Hit rate as a percentage
    """
    correct = (predictions == actuals).sum()
    total = len(actuals)
    return (correct / total) * 100


def backtest_strategy(probabilities_df, actual_returns_df, transaction_cost=0.001):
    """
    Backtest the trading strategy: select top 2 stocks each week, equal weight
    
    Parameters:
    probabilities_df: DataFrame with probability predictions (dates as index, stocks as columns)
    actual_returns_df: DataFrame with actual returns (dates as index, stocks as columns)
    transaction_cost: Cost per transaction (default 0.1% = 0.001)
    
    Returns:
    DataFrame with portfolio returns and holdings
    """
    # Align indices
    common_dates = probabilities_df.index.intersection(actual_returns_df.index)
    probs = probabilities_df.loc[common_dates]
    returns = actual_returns_df.loc[common_dates]
    
    portfolio_returns = []
    portfolio_dates = []
    holdings_history = []
    
    # Iterate through each week
    for date in probs.index:
        # Rank stocks by probability
        ranked = probs.loc[date].sort_values(ascending=False)
        top_2_stocks = ranked.head(2).index.tolist()
        
        # Get returns for top 2 stocks
        if date in returns.index:
            stock_returns = returns.loc[date, top_2_stocks].values
            
            if not np.isnan(stock_returns).any():
                # Equal weight: 50% each
                weights = np.array([0.5, 0.5])
                
                # Calculate portfolio return
                portfolio_return = np.sum(weights * stock_returns)
                
                # Apply transaction costs (buy and sell)
                total_cost = 2 * transaction_cost  # buy cost + sell cost
                portfolio_return -= total_cost
                
                portfolio_returns.append(portfolio_return)
                portfolio_dates.append(date)
                holdings_history.append({
                    'Date': date,
                    'Stock_1': top_2_stocks[0],
                    'Stock_1_Prob': ranked[top_2_stocks[0]],
                    'Stock_1_Return': stock_returns[0],
                    'Stock_2': top_2_stocks[1],
                    'Stock_2_Prob': ranked[top_2_stocks[1]],
                    'Stock_2_Return': stock_returns[1],
                    'Portfolio_Return': portfolio_return
                })
    
    backtest_df = pd.DataFrame(holdings_history)
    return backtest_df


class FeatureScaler:
    """
    Scaler for features with train/test separation handling
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, X_train):
        """Fit scaler on training data"""
        self.scaler.fit(X_train)
        self.fitted = True
        return self
    
    def transform(self, X):
        """Transform data using fitted scaler"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted first")
        return self.scaler.transform(X)
    
    def fit_transform(self, X_train):
        """Fit and transform training data"""
        return self.scaler.fit_transform(X_train)
