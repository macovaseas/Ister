# ISTER: Inverted Seasonal-Trend Decomposition Transformer

This repository contains the official implementation of the paper "ISTER: Inverted Seasonal-Trend Decomposition Transformer," under review for ICLR 2024. ISTER is a novel model designed for long-term multivariate time series forecasting, addressing challenges like computational complexity, intricate temporal pattern capture, and cyclical information consideration.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

Time series forecasting is crucial in various applications, including energy consumption, transportation, economic planning, weather prediction, and disease propagation. The Inverted Seasonal-Trend Decomposition Transformer (ISTER) is introduced to enhance long-term multivariate time series forecasting by decomposing time series into seasonal and trend components and employing a channel-independent network architecture.

## Features

- **Seasonal-Trend Decomposition:** Decomposes time series into seasonal and trend components for more accurate forecasting.
- **Channel-Independent Architecture:** Enhances performance in multivariate prediction tasks.
- **State-of-the-Art Performance:** Achieves superior accuracy and efficiency in long-term forecasting tasks across multiple datasets.

## Installation

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

## Run

To run the experiment in paper, run:

```bash
bash scripts/long_term_forecast/ETT_script/Ister_ETTh1.sh
bash scripts/long_term_forecast/ECL_script/Ister.sh
```
