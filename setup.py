"""
Setup configuration for Crypto Strategy Lab
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="crypto-strategy-lab",
    version="2.0.0",
    author="Joel Saucedo",
    author_email="joel@crypto-strategy-lab.com",
    description="Multi-Exchange Trading Framework with 12 Orthogonal Strategies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joel-saucedo/Crypto-Strategy-Lab",
    packages=find_packages(include=['src', 'src.*']),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.7.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "monitoring": [
            "streamlit>=1.16.0",
            "plotly>=5.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crypto-lab=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["config/**/*.yaml", "docs/**/*.md"],
    },
)
