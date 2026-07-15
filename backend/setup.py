"""
setup.py — makes the src package installable in development mode.
Run: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="aqi-forecasting-noida",
    version="1.0.0",
    description="Scalable AQI Forecasting and Explainable AI System for Noida",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],   # managed via requirements.txt
)
