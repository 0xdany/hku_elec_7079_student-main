# HKU ELEC7079 Quantitative Strategy Development - Student Practice Project

## 🎯 Project Overview

This is the student practice project for the University of Hong Kong's ELEC4546/7079 course "Data Analysis, Signal Prediction & Strategy Development". The project aims to help students master the complete quantitative investment strategy development workflow through three progressive tasks.

## 📋 Task Structure

### 🔰 Part 1: Data Analysis & Feature Exploration (Difficulty: ⭐⭐☆☆☆)
**Estimated Time**: 8-12 hours | 📖 **Detailed Guide**: [TASK_1.md](./TASK_1.md)

**Core Tasks**:
- Task 1.1: Target Variable Engineering & Return Calculation
- Task 1.2: Market & Asset Characteristic Analysis
- Task 1.3: Cross-sectional Analysis

### 🚀 Part 2: Signal Prediction & Alpha Modeling (Difficulty: ⭐⭐⭐⭐☆)
**Estimated Time**: 15-20 hours | 📖 **Detailed Guide**: [TASK_2.md](./TASK_2.md)

**Core Tasks**:
- Task 2.1: Alpha Factor Engineering
- Task 2.2: Information Coefficient (IC) Analysis
- Task 2.3: Machine Learning Model Development

### 🏆 Part 3: Strategy Development & Performance Analysis (Difficulty: ⭐⭐⭐⭐⭐)
**Estimated Time**: 20-25 hours | 📖 **Detailed Guide**: [TASK_3.md](./TASK_3.md)

**Core Tasks**:
- Task 3.1: Strategy Construction & Backtesting
- Task 3.2: Performance Evaluation & Report Generation
- Task 3.3: Result Analysis & Improvement Recommendations

## 🚀 Quick Start

### 1. Environment Setup
```bash
pip install -r requirements.txt
python -c "import pandas, numpy, matplotlib; print('Environment ready!')"
```

### 2. Implementation Order
1. Start with Part 1 (must follow sequence)
2. Run corresponding tests after each task completion
3. Proceed to next task after passing tests

### 3. Test Verification
```bash
# Run specific tests
python -m pytest tests/test_part1/test_task1.py -v

# Run all tests  
python -m pytest tests/ -v
```

### 4. Data Setup
- Data format: pickle files under `data/raw/`
- See `docs/DATA_LOADER_USAGE.md` and `data/README.md` for details

```bash
python - <<'PY'
from src.data_loader import DataLoader
dl = DataLoader()
dl.load_5min_data(); dl.load_daily_data(); dl.load_stock_weights()
print('Data ready')
PY
```

## 🎯 Core Learning Objectives

Upon completing this project, students will master:
- Financial time series data analysis
- Alpha factor engineering techniques
- Machine learning quantitative applications
- Long-short strategy construction and backtesting
- Performance evaluation and risk management

## 📊 Data Description

### Data Structure
- **5-minute data**: 2019-2024, 100 A-share stocks, including OHLCV and VWAP
- **Daily data**: Same stock universe with daily frequency
- **Stock weights**: For benchmark index construction

### Data Fields
- `open_px`: Open price
- `high_px`: High price
- `low_px`: Low price  
- `close_px`: Close price
- `volume`: Trading volume
- `vwap`: Volume-weighted average price

See `data/README.md` for file paths and formats.

## 🎯 Assessment Criteria

### Code Quality (40%)
- ✅ Complete function implementation passing tests
- ✅ Clear code structure with comprehensive comments
- ✅ Appropriate error handling
- ✅ Modular design

### Analysis Quality (35%)
- ✅ Correct understanding of financial data characteristics
- ✅ Reasonable statistical analysis methods
- ✅ Effective visualization presentations
- ✅ In-depth result interpretations

### Innovation Capability (25%)
- ✅ Creative factor design approaches
- ✅ Effective strategy improvement solutions
- ✅ Unique analytical perspectives
- ✅ Practical business recommendations

## 📚 Learning Resources

### Required Reading
- [Pandas Time Series Documentation](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Quantitative Trading Fundamentals](https://en.wikipedia.org/wiki/Quantitative_trading)

### Project Docs
- Data loader usage: `docs/DATA_LOADER_USAGE.md`
- Part 1 guide: `TASK_1.md`
- Part 2 guide: `TASK_2.md`
- Part 3 guide: `TASK_3.md`

## 🏅 Project Outcomes

Upon completion, students will possess:
- ✅ Complete quantitative investment workflow mastery
- ✅ Solid Python financial data analysis capabilities
- ✅ Practical machine learning experience in investment
- ✅ Professional strategy development and evaluation skills
- ✅ Industry-standard code development practices

---

**Course**: ELEC4546/7079 - Data Analysis, Signal Prediction & Strategy Development  
**Semester**: Fall 2025  
**Last Updated**: June 2025

Best wishes for successful learning! 🎉
