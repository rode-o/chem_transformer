from setuptools import setup, find_packages

setup(
    name="chem_transformer",  # Name of your package
    version="0.1.0",  # Initial version of your package
    author="Rode Peters",  # Author name
    description="A package for chemical data generation and visualization",  # Brief description
    packages=find_packages(),  # Automatically find and include all packages
    python_requires=">=3.11",  # Specify Python version compatibility
    classifiers=[
        "Programming Language :: Python :: 3.11",  # Python version compatibility
    ],
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines() if line.strip()
    ],  # Dynamically read dependencies from requirements.txt
)
