from setuptools import setup, find_packages

setup(
    name="micros_based_stats",  # Replace with your package name
    version="0.0.2",  # Package version
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for microstate analysis based on statistics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/microstate_based_on_stats",  # Replace with your repo URL
    packages=find_packages(),  # Automatically find sub-packages
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version
)
