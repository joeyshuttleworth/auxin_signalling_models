# Auxin Signalling Models

A package and scripts for generating ODE-based models of auxin gene regulatory networks.

## Description

Genetic regulatory networks (GRNs) play an important role in dictating a plant
cell's response to changes in auxin concentration. These sub-cellular mechanisms
underpin lots of plant physiology, such as plant growth. This repository
contains a python package, scripts and jupiter notebooks which generate
mathematical, ODE-based models of auxin GRNs, and compile performant code for
simulations nad statistical analysis.

## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

The following installation instructions assume an environment with Python (version >= 3.10) and pip installed. The code has been tested under a Linux (Ubuntu) environment.

### Installing

First generate a virtual environment, `.venv`:
```
python3 -m venv .venv
```

Then install the necessary packages

```
python3 -m pip --upgrade pip
python3 -m pip install -e .
```

### Running code
Tests can be run by entering
```
python3 -m pytest
```

The code in Jupyter Notebooks may be run by installing Jupyter and running:
```
jupiter-notebook
```
in the console.
