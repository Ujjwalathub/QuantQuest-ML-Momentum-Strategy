"""
Advanced Utilities for Stock Trading Strategy Backtester
Contains additional analysis, optimization, and visualization functions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def analyze_strategy_by_market_regime(returns, market_returns, regime_threshold=0.0):
    """
    Analyze strategy performance in different market regimes (bull/bear)
    
    Parameters:
    returns: Series of strategy returns
    market_returns: Series of market (benchmark) returns
    regime_threshold: Threshold to define bull/bear market
    
    Returns:
    Dictionary with regime analysis
    """
    bull_mask = market_returns > regime_threshold
    bear_mask = market_returns <= regime_threshold
    
    bull_returns = returns[bull_mask]
    bear_returns = returns[bear_mask]
    
    return {
        'Bull_Market': {
            'Mean Return': bull_returns.mean(),
            'Std Dev': bull_returns.std(),
            'Sharpe': (bull_returns.mean() * 52 - 0.02) / (bull_returns.std() * np.sqrt(52)) 
                      if bull_returns.std() > 0 else 0,
            'Win Rate': (bull_returns > 0).sum() / len(bull_returns) if len(bull_returns) > 0 else 0,
            'Sample Count': len(bull_returns)
        },
        'Bear_Market': {
            'Mean Return': bear_returns.mean(),
            'Std Dev': bear_returns.std(),
            'Sharpe': (bear_returns.mean() * 52 - 0.02) / (bear_returns.std() * np.sqrt(52)) 
                      if bear_returns.std() > 0 else 0,
            'Win Rate': (bear_returns > 0).sum() / len(bear_returns) if len(bear_returns) > 0 else 0,
            'Sample Count': len(bear_returns)
        }
    }


def calculate_rolling_metrics(returns, window=26):
    """
    Calculate rolling performance metrics
    
    Parameters:
    returns: Series of returns
    window: Rolling window size (default 26 weeks = 6 months)
    
    Returns:
    DataFrame with rolling metrics
    """
    rolling_data = {
        'Rolling_Return': returns.rolling(window).sum(),
        'Rolling_Volatility': returns.rolling(window).std() * np.sqrt(52),
        'Rolling_Sharpe': ((returns.rolling(window).mean() * 52 - 0.02) / 
                          (returns.rolling(window).std() * np.sqrt(52))).fillna(0),
        'Rolling_Drawdown': returns.rolling(window).apply(lambda x: ((1 + x).cumprod() - 
                                                                     (1 + x).cumprod().expanding().max()) / 
                                                                     (1 + x).cumprod().expanding().max()).min()
    }
    
    return pd.DataFrame(rolling_data)


def optimize_portfolio_weights(returns_matrix, expected_returns=None, optimization_method='equal'):
    """
    Calculate optimal portfolio weights
    
    Parameters:
    returns_matrix: DataFrame of asset returns (assets as columns)
    expected_returns: Array of expected returns for each asset
    optimization_method: 'equal', 'min_variance', 'max_sharpe'
    
    Returns:
    Array of optimal weights
    """
    n_assets = returns_matrix.shape[1]
    
    if optimization_method == 'equal':
        # Equal weight
        weights = np.array([1/n_assets] * n_assets)
    
    elif optimization_method == 'min_variance':
        # Minimum variance portfolio
        cov_matrix = returns_matrix.cov()
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(n_assets)
        weights = inv_cov @ ones / (ones @ inv_cov @ ones)
    
    else:
        # Default to equal weight
        weights = np.array([1/n_assets] * n_assets)
    
    return weights / weights.sum()  # Ensure sum to 1


def detect_drawdown_periods(returns):
    """
    Identify and analyze drawdown periods
    
    Parameters:
    returns: Series of returns
    
    Returns:
    DataFrame with drawdown information
    """
    cumulative_value = (1 + returns).cumprod()
    running_max = cumulative_value.expanding().max()
    drawdown = (cumulative_value - running_max) / running_max
    
    # Identify drawdown periods
    in_drawdown = drawdown < 0
    drawdown_periods = []
    
    start = None
    max_dd = 0
    for i, (idx, is_dd) in enumerate(zip(drawdown.index, in_drawdown)):
        if is_dd and start is None:
            start = idx
            max_dd = drawdown[idx]
        elif is_dd:
            max_dd = min(max_dd, drawdown[idx])
        elif not is_dd and start is not None:
            drawdown_periods.append({
                'Start': start,
                'End': drawdown.index[i-1] if i > 0 else idx,
                'Max_Drawdown': max_dd,
                'Duration_Weeks': i if start is None else (i - drawdown.index.get_loc(start))
            })
            start = None
            max_dd = 0
    
    return pd.DataFrame(drawdown_periods)


def stress_test_strategy(returns, shock_scenarios):
    """
    Stress test strategy against various market shocks
    
    Parameters:
    returns: Series of historical returns
    shock_scenarios: Dictionary of scenarios {name: percentage_change}
    
    Returns:
    Dictionary with stressed returns and metrics
    """
    results = {}
    
    for scenario_name, shock in shock_scenarios.items():
        # Apply shock to returns
        shocked_returns = returns * (1 + shock)
        
        # Calculate metrics on shocked returns
        cumulative = (1 + shocked_returns).prod() - 1
        sharpe = (shocked_returns.mean() * 52 - 0.02) / (shocked_returns.std() * np.sqrt(52))
        
        results[scenario_name] = {
            'Cumulative_Return': cumulative,
            'Sharpe_Ratio': sharpe,
            'Hit_Rate': (shocked_returns > 0).mean(),
            'Max_Loss': shocked_returns.min()
        }
    
    return pd.DataFrame(results).T


def analyze_stock_concentration(holdings_df):
    """
    Analyze portfolio concentration and diversification
    
    Parameters:
    holdings_df: DataFrame from backtest with Stock_1 and Stock_2 columns
    
    Returns:
    Dictionary with concentration metrics
    """
    # Count selections
    all_stocks = pd.concat([holdings_df['Stock_1'], holdings_df['Stock_2']])
    stock_counts = all_stocks.value_counts()
    
    # Herfindahl index (concentration measure)
    weights = stock_counts / stock_counts.sum()
    herfindahl = (weights ** 2).sum()
    
    # Equally weighted version
    n_stocks = len(stock_counts)
    min_herfindahl = 1 / n_stocks
    
    # Concentration ratio (top 2 stocks)
    top_2_concentration = weights.head(2).sum()
    
    return {
        'Total_Unique_Stocks': len(stock_counts),
        'Herfindahl_Index': herfindahl,
        'Min_Herfindahl': min_herfindahl,
        'Concentration_Score': (herfindahl - min_herfindahl) / (1 - min_herfindahl),
        'Top_2_Concentration': top_2_concentration,
        'Stock_Counts': stock_counts.to_dict()
    }


def calculate_risk_adjusted_metrics(returns, benchmark_returns=None, risk_free_rate=0.02):
    """
    Calculate comprehensive risk-adjusted metrics
    
    Parameters:
    returns: Series of strategy returns
    benchmark_returns: Series of benchmark returns (optional)
    risk_free_rate: Risk-free rate for calculations
    
    Returns:
    Dictionary with risk-adjusted metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['Total_Return'] = (1 + returns).prod() - 1
    metrics['Annual_Return'] = (1 + returns.mean()) ** 52 - 1
    metrics['Annual_Volatility'] = returns.std() * np.sqrt(52)
    metrics['Sharpe_Ratio'] = (metrics['Annual_Return'] - risk_free_rate) / metrics['Annual_Volatility']
    
    # Additional metrics
    metrics['Sortino_Ratio'] = None
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        downside_volatility = downside_returns.std() * np.sqrt(52)
        metrics['Sortino_Ratio'] = (metrics['Annual_Return'] - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
    
    # Tail risk metrics
    metrics['Skewness'] = returns.skew()
    metrics['Kurtosis'] = returns.kurtosis()
    metrics['VaR_95'] = returns.quantile(0.05)  # Value at risk at 95% confidence
    metrics['CVaR_95'] = returns[returns <= returns.quantile(0.05)].mean()  # Conditional VaR
    
    # Benchmark comparison (if provided)
    if benchmark_returns is not None:
        metrics['Alpha'] = metrics['Annual_Return'] - (risk_free_rate + 
                                                       (benchmark_returns.mean() * 52 - risk_free_rate))
        metrics['Beta'] = returns.cov(benchmark_returns) / benchmark_returns.var()
        metrics['Correlation_with_Benchmark'] = returns.corr(benchmark_returns)
        metrics['Information_Ratio'] = (returns - benchmark_returns).mean() / (returns - benchmark_returns).std() * np.sqrt(52)
    
    return metrics


class PortfolioOptimizer:
    """Utility class for portfolio optimization tasks"""
    
    def __init__(self, returns_data):
        """
        Initialize optimizer with historical returns
        
        Parameters:
        returns_data: DataFrame of returns (dates as index, assets as columns)
        """
        self.returns = returns_data
        self.mean_returns = returns_data.mean()
        self.cov_matrix = returns_data.cov()
        self.n_assets = returns_data.shape[1]
    
    def efficient_frontier(self, n_portfolios=100, risk_free_rate=0.02):
        """
        Generate efficient frontier
        
        Parameters:
        n_portfolios: Number of random portfolios to generate
        risk_free_rate: Risk-free rate
        
        Returns:
        DataFrame with portfolio volatilities and returns
        """
        results = []
        
        for _ in range(n_portfolios):
            weights = np.random.random(self.n_assets)
            weights /= weights.sum()
            
            port_return = (weights * self.mean_returns).sum() * 52
            port_std = np.sqrt(weights @ self.cov_matrix @ weights) * np.sqrt(52)
            sharpe = (port_return - risk_free_rate) / port_std if port_std > 0 else 0
            
            results.append({
                'Return': port_return,
                'Volatility': port_std,
                'Sharpe': sharpe,
                'Weights': list(weights)
            })
        
        return pd.DataFrame(results)
    
    def best_portfolio_by_sharpe(self, n_iterations=1000, risk_free_rate=0.02):
        """Find portfolio with maximum Sharpe ratio"""
        frontier = self.efficient_frontier(n_iterations, risk_free_rate)
        best_idx = frontier['Sharpe'].idxmax()
        return frontier.loc[best_idx]


# Export for use in notebooks
__all__ = [
    'analyze_strategy_by_market_regime',
    'calculate_rolling_metrics',
    'optimize_portfolio_weights',
    'detect_drawdown_periods',
    'stress_test_strategy',
    'analyze_stock_concentration',
    'calculate_risk_adjusted_metrics',
    'PortfolioOptimizer'
]
