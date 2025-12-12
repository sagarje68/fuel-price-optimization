"""
Helper utility functions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta


def calculate_profit(price: float, cost: float, volume: float) -> float:
    """
    Calculate total profit.
    
    Args:
        price: Selling price per liter
        cost: Cost per liter
        volume: Volume sold in liters
        
    Returns:
        Total profit
    """
    return (price - cost) * volume


def calculate_profit_margin(price: float, cost: float) -> float:
    """
    Calculate profit margin percentage.
    
    Args:
        price: Selling price per liter
        cost: Cost per liter
        
    Returns:
        Profit margin percentage
    """
    if price == 0:
        return 0.0
    return ((price - cost) / price) * 100


def format_currency(amount: float) -> str:
    """
    Format amount as currency string.
    
    Args:
        amount: Dollar amount
        
    Returns:
        Formatted currency string
    """
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """
    Format value as percentage string.
    
    Args:
        value: Percentage value
        
    Returns:
        Formatted percentage string
    """
    return f"{value:.2f}%"


def get_date_range(start_date: str, end_date: str) -> List[str]:
    """
    Generate list of dates between start and end.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        List of date strings
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    dates = pd.date_range(start, end, freq='D')
    return [d.strftime('%Y-%m-%d') for d in dates]


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with statistics
    """
    arr = np.array(values)
    
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75))
    }


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def moving_average(data: List[float], window: int) -> List[float]:
    """
    Calculate moving average.
    
    Args:
        data: List of values
        window: Window size
        
    Returns:
        List of moving averages
    """
    if len(data) < window:
        return data
    
    return list(pd.Series(data).rolling(window=window, min_periods=1).mean())


def exponential_moving_average(data: List[float], span: int) -> List[float]:
    """
    Calculate exponential moving average.
    
    Args:
        data: List of values
        span: Span parameter for EMA
        
    Returns:
        List of exponential moving averages
    """
    return list(pd.Series(data).ewm(span=span, adjust=False).mean())


def is_weekend(date: str) -> bool:
    """
    Check if date is weekend.
    
    Args:
        date: Date string (YYYY-MM-DD)
        
    Returns:
        True if weekend, False otherwise
    """
    dt = pd.to_datetime(date)
    return dt.dayofweek >= 5


def get_season(date: str) -> str:
    """
    Get season for a given date.
    
    Args:
        date: Date string (YYYY-MM-DD)
        
    Returns:
        Season name
    """
    month = pd.to_datetime(date).month
    
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'


def round_to_nearest(value: float, precision: float = 0.01) -> float:
    """
    Round value to nearest precision.
    
    Args:
        value: Value to round
        precision: Rounding precision
        
    Returns:
        Rounded value
    """
    return round(value / precision) * precision


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value between min and max.
    
    Args:
        value: Value to clamp
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def normalize(values: List[float]) -> List[float]:
    """
    Normalize values to 0-1 range.
    
    Args:
        values: List of values
        
    Returns:
        Normalized values
    """
    arr = np.array(values)
    min_val = arr.min()
    max_val = arr.max()
    
    if max_val == min_val:
        return [0.5] * len(values)
    
    return list((arr - min_val) / (max_val - min_val))


if __name__ == "__main__":
    # Example usage
    print("Profit:", calculate_profit(95.0, 85.0, 14000))
    print("Margin:", format_percentage(calculate_profit_margin(95.0, 85.0)))
    print("Is weekend:", is_weekend("2024-12-31"))
    print("Season:", get_season("2024-12-31"))
