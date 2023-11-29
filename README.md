# HNSW Approximate Nearest Neighbors Experiment
Final Project SDSC3001 Big Data: The Art and Science of Scaling

- Lecturer: Dr. YANG Yu
- City University of Hong Kong

### Authors
This project was created by:

- ABDINEGARA, Beatrice
- GUSTAF, Benedict Ronaldo
- HA, Quang Minh
- HALIM, Seivabel Jessica
- YENNOTO, Keane Dylan

## Introduction

This repository contains a series of Jupyter Notebooks and supporting Python scripts for experimenting with Hierarchical Navigable Small World (HNSW) graphs for Approximate Nearest Neighbors (ANN) search. The focus is on building, analyzing, tuning, and testing HNSW-based models to understand their performance in terms of recall and query time comparing to other algorithms for ANN.

## Installation

To set up the environment for running these experiments, follow these steps:

1. Install the required packages:
pip install -r requirements.txt

## Jupyter Notebooks

### 1. Initial Build and Test (`1-intial_build_and_test.ipynb`)

This notebook is the starting point of the experiments. It includes the initial setup, building of the HNSW graph with other algorithms including Brute Force, Annoy, and FLANN. Building time, query time, and recall are measured as performance metric.

### 2. Recall, Query Time, and K Analysis (`2-recall_query_time_and_k_analysis.ipynb`)

In this notebook, we analyze the recall and query time performance of the different algorithms. We also explore the impact of different values of `k` (number of nearest neighbors).

### 3. Tuning HNSW Parameters (`3-tuning_parameter_hnsw.ipynb`)

This notebook is dedicated to tuning the parameters of the HNSW model. It uses the `config.yml` file for parameter configurations.

### 4. Test Tuned HNSW (`4-test_tuned_hnsw.ipynb`)

After tuning the parameters, this notebook tests the performance of the tuned HNSW model, focusing on recall and query time.

## Configuration and Helpers

- *Configuration (`config.yml`):* This YAML file contains the configuration parameters used for tuning the HNSW model in `3-tuning_parameter_hnsw.ipynb`.

- *Experiment Results (`exp_result.py`):* File to store the experiment results.

- *Helpers (`helpers.py`):* This script includes helper functions used across different notebooks for various tasks like data loading, preprocessing, and functions for building and running the algorithms.
> Reference: https://github.com/erikbern/ann-benchmarks/tree/main