# Guide II  ·  LAUNCH SIMULATIONS
This guide explain how to launch a simulation on a personal computer
## II. 1. Place the data in folder
### ⮕ The Numpy (.npy) format
- All the raw data provided to the project must be in a form of 2D numpy arrays.
- All the data files must be loaded in the Numpy format (see "[.npy](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html)" for more information about the .npy format).
- This format is used for its simplicity and its ubiquity.
### ⮕ The CSV (.csv) file
- All data files used must be meticulously recorded in a .csv file, similar to the one available in the [data](https://github.com/AxMeNi/GeoclassificationMPS/tree/main/data) folder.
- The validity of the provided information and data is verified before each simulation.

## II. 2. Provide the parameters
All the parameters related to the simulations can be changed in the INTERFACE.PY script.
NOTE: To modify the plot style and the generic textual information displayed on the plot, use the DISPLAY_FUNCTIONS.py script.
### ⮕ General parameters
- All parameters related to the data are stored in the .csv file. Therefore, the first parameter to specify is the path to this CSV file
- For the following parameters :
  - seed
  - ti_pct_area
  - ti_nshapes
  - nRandomTICDsets
  
  It is required to specify directly their value in the get_simulation_info function (e.g. set "ti_pct_area = 55" if the percentage of the data grid covered by the training image is 55 %)
- 
