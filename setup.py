"""Setup script for Stock&Finance application"""
from setuptools import setup, find_packages

setup(
    name="stockfinance",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.2',
        'tensorflow>=2.6.0',
        'yfinance>=0.1.63',
        'plotly>=5.1.0',
        'PyQt6>=6.2.0',
        'PyQtWebEngine>=6.2.0',
        'pandas-ta>=0.3.14b0',
        'python-dotenv>=0.19.0'
    ],
    entry_points={
        'console_scripts': [
            'stockfinance=src.main:main',
        ],
    },
    python_requires='>=3.8',
)