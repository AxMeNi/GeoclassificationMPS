# -*- coding:utf-8 -*-
__projet__ = "some_functions.py"
__nom_fichier__ = "interface"
__author__ = "MENGELLE Axel"
__date__ = "juillet 2024"

from some_functions_bis import *
from main import *
from some_function_ter import *

# ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ INTERFACE FOR PROGRAMMING A COMBINED DEESSE AND LOOPUI SIMULATION                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

##################### ARRAY DEFINITION FOR THE SIMULATION #####################

# Simulated variables can be partially informed
simulated_var = np.array([[[varAval11 varBval11],[varAval12 varBval12], [varAval13 varBval13]]
                          [[varAval21 varBval21],[varAval22 varBval22], [varAval23 varBval23]]
                          ])

# Auxiliary variables must be fully informed
auxiliary_var = np.array([[[var1val11 var2val11],[var1val12 var2val12], [var1val13 var2val13]]
                          [[var1val21 var2val21],[var1val22 var2val22], [var1val23 var2val23]]
                          ])

names_var = np.array([[name_varA name_varB], [name_var1 name_var2]])

# Type must be set as "continuous" or "categorical".
types_var = np.array([[type_varA type_varB], [type_var1 type_var2]])



##################### LOCATIONS OF THE LOADING AND SAVING FILES #####################

# Path to the pre-processed data folder
path2data = "./data/"

# Name of the pre-processed data file
suffix = "-simple"  # "", "-simple", "-very-simple"
data_filename = "mt-isa-data" + suffix + ".pickle"

# Paths to the saving FOLDERS
path2ti = "./ti/"  # Training images path
path2cd = "./cd/"  # Conditioning data path
path2real = "./mpsReal/"  # Realizations path
path2log = "./log/"  # Logging info path
path2ind = "./ind/"  # Index data path

# Create directories if they do not exist
if not os.path.exists(path2ti):
    os.makedirs(path2ti)
if not os.path.exists(path2cd):
    os.makedirs(path2cd)
if not os.path.exists(path2real):
    os.makedirs(path2real)
if not os.path.exists(path2log):
    os.makedirs(path2log)
if not os.path.exists(path2ind):
    os.makedirs(path2ind)

# Load pre-processed data from pickle file
with open(pickledestination, 'rb') as f:
    [_, grid_geo, grid_lmp, grid_mag,
     grid_grv, grid_ext, vec_x, vec_y
     ] = pickle.load(f)

# Extract dimensions of grid_geo
ny, nx = grid_geo.shape

##################### RANDOM PARAMETERS #####################

seed = 12345

##################### TRAINING IMAGE PARAMETERS #####################

ti_pct_area = 33  # Percentage area of the TI
ti_ndisks = 1  # Number of disks in the TI
ti_realid = 1  # Realization ID for the TI
xycv = False  # Flag for cross-validation

##################### CONDITIONING DATA PARAMETERS #####################

geolcd = False  # Flag for geological constraints

# The following is not implemented and is only used to illustrate what the cd settings might look like

CDFilesProvided = True  # Flag to tell whether the program should take the already provided conditioning data or should
# create its own conditioning data

# In the case of provided files
cd_filenames = ["cdfile1", "cdfile2"]  # cd_filenames = None # Conditioning data location
cd_variables = np.array([["var1_cdfile1", "var2_cdfile1"], ["var1_cdfile2", "var2_cdfile2"]])
auxiliary_cd = ["var1"]


##################### DEESSE SIMULATION PARAMETERS #####################

nn = 12  # Number of neighboring nodes
dt = 0.1  # Distance threshold
ms = 0.25  # Maximum scan fraction
numberofmpsrealizations = 1  # Number of Deesse realizations

nthreads = 1  # Number of threads for parallel processing

##################### LAUNCHING PARAMETERS #####################

configs = [
    (33, 1, 1, 10, 4, True, True),
    (33, 1, 1, 10, 4, True, False),
    (33, 1, 1, 10, 4, False, False),
    (33, 1, 1, 10, 4, False, True)
]

##################### COLORMAP PARAMETERS #####################

cm = plt.get_cmap('tab20')
myclrs = np.asarray(cm.colors)[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11], :]
n_bin = 11
cmap_name = 'my_tab20'
mycmap = LinearSegmentedColormap.from_list(cmap_name, myclrs, N=n_bin)
ticmap = LinearSegmentedColormap.from_list('ticmap', np.vstack(([0, 0, 0], myclrs)), N=n_bin + 1)


# ╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ LAUNCH THE SIMULATIONS                                                                                             ║
# ╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

def check_variables(simulated_var, auxiliary_var, names_var, types_var):
    """
    Check the validity of input variables used in a simulation or analysis.

    Parameters:
    simulated_var : numpy.ndarray
        Array containing simulated variables.
    auxiliary_var : numpy.ndarray
        Array containing auxiliary variables.
    names_var : numpy.ndarray
        Array containing names corresponding to variables in simulated_var and auxiliary_var.
    types_var : numpy.ndarray
        Array containing expected types for each variable in simulated_var and auxiliary_var.
        Expected types are "continuous" for numerical types and "categorical" for integer types.

    Returns:
    numpy.ndarray, numpy.ndarray:
        Modified arrays simulated_var and auxiliary_var where None values are replaced by -9999999.

    Raises:
    ValueError:
        - If the number of variable names does not match the number of variables in simulated_var or auxiliary_var.
        - If simulated_var and auxiliary_var do not have the same XY dimensions.
        - If auxiliary_var contains NaN values.
        - If simulated_var or auxiliary_var contains -9999999 values.
    TypeError:
        - If the type of any variable in simulated_var or auxiliary_var does not match the expected type in types_var.
    """
    # Replace None with -9999999
    simulated_var = np.where(simulated_var == None, -9999999, simulated_var)
    auxiliary_var = np.where(auxiliary_var == None, -9999999, auxiliary_var)
    
    # Check for variable names
    if simulated_var.shape != names_var.shape or auxiliary_var.shape != names_var.shape:
        message = "The number of variable names does not match the number of variables."
        if simulated_var.shape != names_var.shape:
            message += f" In simulated_var, expected {simulated_var.shape}, got {names_var.shape}."
        if auxiliary_var.shape != names_var.shape:
            message += f" In auxiliary_var, expected {auxiliary_var.shape}, got {names_var.shape}."
        raise ValueError(message)
    
    # Check dimensions XY
    if simulated_var.shape[:2] != auxiliary_var.shape[:2]:
        raise ValueError(f"simulated_var and auxiliary_var do not have the same dimensions XY. simulated_var: {simulated_var.shape[:2]}, auxiliary_var: {auxiliary_var.shape[:2]}")
    
    # Check for NaN values in auxiliary_var
    if np.isnan(auxiliary_var).any():
        raise ValueError("auxiliary_var contains NaN values, but it must be fully informed.")
    
    # Check for -9999999 values in simulated_var
    if np.any(simulated_var == -9999999):
        raise ValueError("simulated_var contains -9999999 values, which are not allowed.")
    
    # Check for -9999999 values in auxiliary_var
    if np.any(auxiliary_var == -9999999):
        raise ValueError("auxiliary_var contains -9999999 values, which are not allowed.")
    
    # Check types
    for i in range(simulated_var.shape[0]):
        for j in range(simulated_var.shape[1]):
            for k in range(simulated_var.shape[2]):
                # Check type "continuous"
                if types_var[i, j] == "continuous":
                    if not isinstance(simulated_var[i, j, k], (int, float, np.int32, np.int64, np.float32, np.float64)):
                        raise TypeError(f"Type mismatch for simulated_var[{i}, {j}, {k}]. Expected numerical type for 'continuous', got {type(simulated_var[i, j, k])}.")
                    if not isinstance(auxiliary_var[i, j, k], (int, float, np.int32, np.int64, np.float32, np.float64)):
                        raise TypeError(f"Type mismatch for auxiliary_var[{i}, {j}, {k}]. Expected numerical type for 'continuous', got {type(auxiliary_var[i, j, k])}.")
                
                # Check type "categorical"
                elif types_var[i, j] == "categorical":
                    if not isinstance(simulated_var[i, j, k], (int, np.int32, np.int64)):
                        raise TypeError(f"Type mismatch for simulated_var[{i}, {j}, {k}]. Expected integer type for 'categorical', got {type(simulated_var[i, j, k])}.")
                    if not isinstance(auxiliary_var[i, j, k], (int, np.int32, np.int64)):
                        raise TypeError(f"Type mismatch for auxiliary_var[{i}, {j}, {k}]. Expected integer type for 'categorical', got {type(auxiliary_var[i, j, k])}.")
                
                # Invalid type
                else:
                    raise ValueError(f"Invalid type '{types_var[i, j]}' specified. Expected 'continuous' or 'categorical'.")
    
    
    return simulated_var, auxiliary_var


def execute_program(ti_pct_area, ti_ndisks, ti_realid, mps_nreal, nthreads, geolcd, xycv, timesleep=0, verb=True):
    """
    Function to execute the main program with given parameters.
    """

    # Generate the TI mask
    grid_msk = gen_ti_mask(nx, ny, ti_pct_area, ti_ndisks, myseed + ti_realid)

    # Build the TI with given parameters
    geocodes, ngeocodes, tiMissingGeol, cond_data = build_ti(
        grid_msk, ti_ndisks, ti_pct_area, ti_realid, geolcd, xycv)

    # Run DEESSE simulation
    deesse_output = run_deesse(tiMissingGeol, mps_nreal, nn, dt, ms, seed, nthreads, geolcd, cond_data)

    # Retrieve the simulation results
    sim = deesse_output['sim']

    # Perform statistics on the realizations
    # Gather all realizations into one image
    all_sim = gn.img.gatherImages(sim)  # all_sim is one image with nreal variables
    # Compute the pixel-wise proportion for the given categories
    all_sim_stats = gn.img.imageCategProp(all_sim, geocodes)

    # Initialize the realizations array with NaN values
    realizations = np.ones((ny, nx, mps_nreal)) * np.nan
    for i in range(mps_nreal):
        ix = i * tiMissingGeol.nv
        realizations[:, :, i] = all_sim.val[ix, 0, :, :]  # Assign the simulation values to the realizations array

    # Create a title for the plot
    addtitle = f'geolcd: {geolcd} - xycv: {xycv}'
    # Plot the realizations and reference grid
    plot_real_and_ref(realizations, reference=grid_geo, mask=1 - grid_msk, nrealmax=mps_nreal, addtitle=addtitle)

    # Print message indicating completion
    print("Simulation and plotting complete.")

    # Additional analysis and output can be added here

if __name__ = "__main__":
    simulated_var_modified, auxiliary_var_modified = check_variables(simulated_var, auxiliary_var, names_var, types_var)
    for config in configs:
        ti_pct_area, ti_ndisks, ti_realid, mps_nreal, nthreads, geolcd, xycv = config
        print(f"Running configuration: geolcd={geolcd}, xycv={xycv}")
        execute_program(ti_pct_area, ti_ndisks, ti_realid, mps_nreal, nthreads, geolcd, xycv)
