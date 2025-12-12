from setuptools import setup, find_packages

setup(
    name="micros_based_on_stats",
    version="0.0.1",
    author="Matteo Demuru",
    author_email="suforraxi@gmail.com",
    description="A package for selecting microstate classes based on statistics.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/suforraxi/microstate_based_on_stats",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
                "numpy",
                "setuptools",
                "pandas",
                "matplotlib",
                "scipy",
                "scikit-learn",
                "statsmodels",
                "mne",
                "neurokit2"
    ]
)
