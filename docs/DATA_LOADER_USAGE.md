# Data Loader Usage Guide

This guide explains how to use the updated DataLoader with pickle format support.

## Overview

The DataLoader has been simplified to use pickle format for faster and more efficient data loading. All Excel-related code has been removed to streamline the implementation.

## Quick Start

```python
from src.data_loader import DataLoader

# Initialize the data loader
loader = DataLoader()

# Load 5-minute data
data_5min = loader.load_5min_data()
print(f"5-minute data shape: {data_5min.shape}")

# Load daily data
data_daily = loader.load_daily_data()
print(f"Daily data shape: {data_daily.shape}")

# Load stock weights
weights = loader.load_stock_weights()
print(f"Weights shape: {weights.shape}")
```

## Data Structure

### 5-Minute Data
- **Format**: MultiIndex DataFrame
- **Index**: DatetimeIndex (5-minute intervals)
- **Columns**: MultiIndex (stock_symbol, data_field)
- **Data Fields**: open_px, high_px, low_px, close_px, volume, vwap

### Daily Data
- **Format**: MultiIndex DataFrame
- **Index**: DatetimeIndex (daily)
- **Columns**: MultiIndex (stock_symbol, data_field)
- **Data Fields**: open_px, high_px, low_px, close_px, volume, vwap

### Stock Weights
- **Format**: pandas Series
- **Index**: stock symbols
- **Values**: weights (sum to 1.0)

## Performance

- **5-minute data**: ~0.6 seconds to load 71,344 rows × 600 columns
- **Daily data**: ~0.01 seconds to load 1,456 rows × 600 columns
- **Caching**: Subsequent loads are instant (from memory cache)

## File Locations

- **5-minute data**: `data/raw/Train_IntraDayData_5minute.pkl`
- **Daily data**: `data/raw/Train_DailyData.pkl`
- **Stock weights**: `data/raw/stock_weight.pkl`

## Convenience Functions

```python
from src.data_loader import load_data

# Load all data types
data = load_data("both")
print(data.keys())  # ['5min', 'daily', 'weights']

# Load specific data type
data_5min = load_data("5min")
data_daily = load_data("daily")
```

## Error Handling

The DataLoader includes robust error handling:

- **File not found**: Returns sample data for development
- **Invalid data format**: Provides clear error messages
- **Data validation**: Checks data integrity automatically

## Testing

Run the test script to verify functionality:

```bash
python3 test_data_loader.py
```

This will test:
- Data loading performance
- Data integrity validation
- Caching functionality
- Error handling

