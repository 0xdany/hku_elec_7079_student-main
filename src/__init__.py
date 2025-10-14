"""
HKU ELEC4546/7079 Quantitative Strategy Development Project

This package contains the complete implementation for the quantitative strategy
development course project, including data analysis, alpha modeling, and 
strategy backtesting components.

Package Structure:
    data_loader: Data loading and preprocessing utilities
    utils: Common utility functions
    part1_data_analysis: Data analysis and feature exploration (Tasks 1-3)
    part2_alpha_modeling: Signal prediction and alpha modeling (Tasks 4-6)  
    part3_strategy: Strategy development and performance analysis (Tasks 7-9)
    common: Shared modules for factor engineering, backtesting, and metrics

Author: ELEC4546/7079 Course
Date: December 2024
Version: 1.0.0
"""

# Import core modules
from . import data_loader
from . import utils
from . import part1_data_analysis

# Version information
__version__ = '1.0.0'
__author__ = 'ELEC4546/7079 Course'
__email__ = 'course@hku.hk'

# Package metadata
__title__ = 'HKU Quantitative Strategy Development'
__description__ = 'Comprehensive quantitative strategy development framework'
__url__ = 'https://github.com/hku-elec7079/quant-strategy-dev'

__all__ = [
    'data_loader',
    'utils', 
    'part1_data_analysis'
]


