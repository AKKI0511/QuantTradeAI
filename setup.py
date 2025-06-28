from setuptools import setup, find_packages

setup(
    name="quanttradeai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "yfinance",
        "scikit-learn",
        "xgboost",
        "pandas-ta",
        "matplotlib",
        "seaborn",
    ],
    author="Akshat Joshi",
    author_email="joshiakshat0511@gmail.com",
    description="A machine learning framework for quantitative trading strategies",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
)
