# User Guide for the Notebook `src/debug.ipynb`
> GeoclassificationMPS Project  
> Branch: `debug`

---

## 📝 Summary

This notebook is designed to **test and debug the geostatistical simulation pipeline** within the GeoclassificationMPS project. It enables configuration of simulation parameters, generation of Training Images (TI) and Conditioning Data (CD), execution of the simulation via DeeSse, and subsequent analysis and visualization of results.

---

## 🗺️ General Structure

### 1. **Initialization and Imports**
```python
from launcher import *
from interface import *
from geone.imgplot import drawImage2D
import os
os.chdir(r"C:\users\amen0052\documents\personal\geoclassificationmps")
print("Current directory:", os.getcwd())
```

### 2. **Simulation Parameters**
Key parameters to adjust as needed:
- `arg_seed`: Random seed.
- `arg_n_ti`: Number of Training Images.
- `arg_ti_pct_area`: Percentage of area used for TI.
- `arg_num_shape`: Number of shapes (for TI generation).
- `arg_aux_vars`: Auxiliary variables.
- `arg_output_dir`: Output directory.

### 3. **Advanced Configuration**
The central function `get_simulation_info_custom()` gathers all necessary parameters:
- Data CSV path
- TI generation methods available:  
  `"DependentCircles"`, `"DependentSquares"`, `"IndependentSquares"`, `"Customised"`, `"ReducedTiSg"`
- Management of auxiliary, conditional, and simulation variables
- DeeSse settings (number of realizations, threads, etc.)
- Options for saving and visualizing outputs

---

## 🛠️ Practical Usage Guide

### ⚙️ 1. Prepare Your Environment

- Ensure all required files (.csv, .npy, custom masks) are present in the expected folder (`./data/`).
- Adjust file paths as needed in the import and configuration cells.

### 📋 2. Define Your Parameters

In the relevant cell:
```python
verbose = True
arg_seed = 123
arg_n_ti = 1
arg_ti_pct_area = 90
arg_num_shape = 15
arg_aux_vars = ["grid_lmp", "grid_grv"]
arg_output_dir = "./output/tmp"
```
Modify these values according to your study case.

### 🏗️ 3. Configure the Simulation

Call the function:
```python
params, shorten, nvar, sim_var, auxTI_var, auxSG_var, condIm_var, names_var, types_var, outputVarFlag, nr, nc = get_simulation_info_custom(
    arg_seed, arg_n_ti, arg_ti_pct_area, arg_num_shape, arg_aux_vars, arg_output_dir)
```

### 🧩 4. Generate Training Images and Conditioning Data

Select the TI generation method, for example:
```python
ti_methods = params['ti_methods']
if "DependentSquares" in ti_methods:
    # Generation and visualization...
```
Other methods are available based on your requirements.

### 🖼️ 5. Visualize Masks and Training Images

Use plotting functions to display intermediate results:
```python
plot_mask(ti_frame_DS[0], masking_strategy="Dependent Squares")
drawImage2D(ti_list[0], iv=0)
save_plot("aux_var")
```

### ▶️ 6. Run the DeeSse Simulation

Prepare the simulation object:
```python
deesse_input = gn.deesseinterface.DeesseInput(...)
deesse_output = gn.deesseinterface.deesseRun(deesse_input, nthreads=nthreads, verbose=2)
```

### 💾 7. Save the Results

```python
if saveOutput:
    save_simulation(deesse_output, params, output_directory=deesse_output_folder_complete)
```

### 📊 8. Analyze Indicators

Compute and visualize entropy, divergence, etc.:
```python
ent, dist_hist, dist_topo_hamming = calculate_indicators(...)
std_ent, realizations_range1 = calculate_std_deviation(ent, 1, numberofmpsrealizations)
plot_realization(deesse_output=deesse_output)
```

---

## 🧠 Recommendations and Best Practices

- **File Paths**: Paths are currently hard-coded for the main user; consider making them relative or parameterizable for better portability.
- **Documentation**: Follow the notebook’s in-line examples and formats for preparing your data files (.csv, .npy).
- **Modularity**: Experiment with different TI generation methods to compare their impact on results.
- **Visualization**: Use plotting to verify the coherence of masks and simulation outputs.
- **Warnings**: Console warnings and errors are valuable for diagnosing data or configuration issues.

---
