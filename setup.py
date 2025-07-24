"""
Setup script for MMSDS Time Series Forecasting package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mmsds-forecasting",
    version="1.0.0",
    author="mrbestnaija",
    author_email="",
    description="Advanced time series forecasting with SSA, mSSA, and tSSA methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mrbestnaija/mmsds_proj",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "nbsphinx>=0.9.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "mmsds-demo=examples.quick_demo:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)