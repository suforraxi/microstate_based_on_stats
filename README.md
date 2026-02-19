# Microstate Based on Stats

## Overview
The `microstate_based_on_stats` package is designed to assist researchers and data scientists in determining the optimal number of microstates (denoted as `k`) to use in k-means clustering. In k-means clustering, the parameter `k` is often chosen arbitrarily, which can lead to suboptimal results. This package provides tools and methods to make a more informed decision about the optimal number of microstates, based on statistical analysis and hypothesis testing.

## Purpose
The main purpose of this package is to help users:

1. **Determine the Optimal Number of Microstates (`k`)**: The package provides methods to evaluate and select the most suitable number of microstates for a given dataset, ensuring better alignment with the underlying data structure.

2. **Compare Conditions in Experimental Designs**: The package includes tools to analyze and compare differences between conditions in experimental designs, helping researchers validate their hypotheses and draw meaningful conclusions.

## Features
- Statistical analysis to guide the selection of the optimal number of microstates.
- Tools for comparing microstate distributions across different experimental conditions.
- Visualization utilities to aid in interpreting results.

## Installation
To install the package, use the following command:

```bash
pip install -r requirements.txt
```

## Usage
The package includes several modules to perform microstate analysis and visualization. Example scripts are provided in the `examples/` directory to demonstrate how to use the package effectively.

### Example
To run the microstate pipeline for a case-control study, use the script:

```bash
python micros_based_stats/examples/microstate_pipeline_main_casectrl.py
```

For a longitudinal study, use:

```bash
python micros_based_stats/examples/microstate_pipeline_main_long.py
```

## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
