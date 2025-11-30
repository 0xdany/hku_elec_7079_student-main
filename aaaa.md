Loading 5-minute data from: /Users/danyraihan/Documents/HKU/HKU Courses/ELEC4546/hku_elec_7079_student-main/data/raw/Train_IntraDayData_5minute.pkl
5-minute data validation completed
Loaded 5-minute data: 71344 rows, 600 columns
Date range: 2019-01-02 09:30:00 to 2024-12-31 15:00:00
======================================================================
COMPREHENSIVE PARAMETER SWEEP
======================================================================
Progress: 20/144 | Best so far: +16.39%
Progress: 40/144 | Best so far: +52.11%
Progress: 60/144 | Best so far: +52.11%
Progress: 80/144 | Best so far: +52.11%
Progress: 100/144 | Best so far: +52.11%
Progress: 120/144 | Best so far: +52.11%
Progress: 140/144 | Best so far: +52.11%

======================================================================
TOP 10 CONFIGURATIONS BY RETURN
======================================================================
1. lb= 3 reb= 9600 q=0.05 partial=0.05 | Return: +52.11% Sharpe: +0.094 DD: -13.7%
2. lb= 3 reb= 9600 q=0.05 no_partial   | Return: +43.99% Sharpe: +0.083 DD: -13.7%
3. lb= 3 reb= 9600 q=0.05 partial=0.03 | Return: +43.99% Sharpe: +0.083 DD: -13.7%
4. lb= 3 reb= 4800 q=0.05 partial=0.05 | Return: +16.39% Sharpe: +0.037 DD: -18.4%
5. lb= 6 reb= 4800 q=0.10 partial=0.05 | Return: +12.29% Sharpe: +0.036 DD: -17.5%
6. lb= 6 reb= 4800 q=0.10 no_partial   | Return: +12.06% Sharpe: +0.036 DD: -17.2%
7. lb= 3 reb= 9600 q=0.10 partial=0.05 | Return: +11.99% Sharpe: +0.036 DD: -23.6%
8. lb= 3 reb= 4800 q=0.05 no_partial   | Return: +11.98% Sharpe: +0.030 DD: -18.4%
9. lb= 3 reb= 4800 q=0.05 partial=0.03 | Return: +11.98% Sharpe: +0.030 DD: -18.4%
10. lb=12 reb= 4800 q=0.05 no_partial   | Return: +11.85% Sharpe: +0.030 DD: -23.3%

======================================================================
BEST CONFIGURATION
======================================================================
  lookback: 3
  rebalance_periods: 9600
  quantile: 0.05
  partial_threshold: 0.05
  Total Return: +52.11%