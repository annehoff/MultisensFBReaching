# Reaching Simulation Code

This Python code generates simulations for a reaching task, investigating the impact of visual uncertainty and movement time on feedback responses to combined force and visual perturbations.

## Overview

This code simulates a reaching task with varying visual feedback uncertainties and movement times. The simulations focus on the influence of visual uncertainty and movement time on corrections in the lateral dimension of the movement. The results from this code are used to generate Figures 8, 8-1, and 8-2 in the associated manuscript.

## Simulations

The simulations are performed using the LQG (Linear Quadratic Gaussian) framework, including a state estimator based on a Kalman filter.

### Main Script: `main_run_plotting.py`

- Defines simulation parameters and calls LQG & simulation functions from file lqgfunctions
- Runs simulations using either additive or signal-dependent sensory & motor noise
- Generates position, error, force, slopes, and Kalman & control gain plots.

### Remaining Files

- File `lqgfunctions` contains functions implementing LQG control and running the simulations
- The code is also available as jupyter notebook called `Model_Hoffmann_and_Crevecoeur.ipynb`

## Quickstart
Run the script 'main_run_plotting.py' to obtain the basic simulations and generate figures.

```
main_run_plotting; 
```

> 🚧 Success
> 
> Our implementation has been tested on a Dell laptop with 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz 1.69 GHz and 16GB RAM, running Windows 10, Spyder version 5.

## Python libaries

    numpy, matplotlib, seaborn

For any questions or issues, please refer to the corresponding author of the submission.

## Corresponding publication

Hoffmann, A. H., & Crevecoeur, F. (2024). Dissociable effects of urgency and evidence accumulation during reaching revealed by dynamic multisensory integration. eNeuro.
https://doi.org/10.1523/ENEURO.0262-24.2024