from setuptools import setup, find_packages

setup(
    name="air-combat-simulator",
    version="0.1.0",
    description="Physically realistic air combat simulator - F-86 Sabre vs MiG-15",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "plotly>=5.14.0",
        "pandas>=2.0.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3.9",
    ],
)
