# Forward Modelling GUI

## 1. What this program is

This project is a GUI-based forward modelling tool for near-surface geophysical feasibility testing. It provides one shared scenario/template system for comparing several geophysical methods on the same geological model, target geometry, and optional topography.

The GUI is built with PySide6 and currently supports:

- ERT using pyGIMLi
- GPR using gprMax, with a built-in GUI preview/fallback for quick checks
- Seismic acoustic using Devito
- Seismic elastic using Devito
- EM 1D using the 1D FDEM reflection-coefficient implementation
- EM 2D using SimPEG as a practical 3D finite-volume approximation for 2D profile-style EM anomaly testing

The purpose is not to replace specialist modelling software. It is meant to provide a unified tool for quickly testing whether a target may be detectable by different methods under consistent assumptions.

Main features:

- Shared scenario YAML files
- Layered geological models
- Embedded targets/anomalies
- Optional topography correction
- Method-specific acquisition settings
- Background-vs-target comparison for selected methods
- Plot outputs for model, response, anomaly difference, and detectability checks

Implementation summary:

- ERT: finite-element resistivity forward modelling and inversion through pyGIMLi
- GPR: gprMax FDTD modelling, with GUI-compatible preview/fallback plots
- Seismic acoustic: acoustic finite-difference wave simulation with Devito
- Seismic elastic: isotropic elastic velocity-stress finite-difference simulation with Devito
- EM 1D: 1D FDEM layered-earth solver
- EM 2D: SimPEG FDEM approximation for spatial anomaly profiles

The EM 1D module should be treated as the accurate layered-earth reference. The EM 2D module is intended for spatial feasibility and anomaly-shape testing.

---

## 2. Installation instructions

These instructions assume Linux or WSL (for Windows) or Linux VM (for MacOS). Installation should be done inside a Python virtual environment in the project folder. 

gprMax requires a working compiler with OpenMP support. It also has a known habit of being annoying on Windows, so use WSL/Linux emulator, or join the dark side and use Linux ;)

### 2.1 Clone and enter the project

    git clone https://github.com/Darkening-Silhouette/Forward_Modelling && cd Forward_Modelling

### 2.2 Create the virtual environment and install all dependencies

    python3 -m venv venv && source venv/bin/activate && python -m pip install --upgrade pip setuptools wheel && python -m pip install -r requirements.txt && python -m pip install numpy Cython && python -m pip install --no-build-isolation -r requirements-gprmax.txt

### 2.3 Test the installation

    source venv/bin/activate && python -m py_compile main.py gui/main_window.py core/*.py solvers/*.py && python -c "import numpy, scipy, matplotlib, yaml, PySide6; import pygimli, devito, simpeg, discretize, pymatsolver, gprMax; print('All dependencies OK')"

### 2.4 Run the program

    source venv/bin/activate && python main.py

### 2.5 Normal workflow

Start the GUI, load or select a scenario YAML file (or make your own YAML file), choose a method from the dropdown, adjust the method-specific settings, click run, then inspect the output plots and log.

You may also apply topographic/elevation correction. A default .csv file is provided in the /topographic_elevation_correction folder.

