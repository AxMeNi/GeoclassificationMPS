# Guide I  ·  NAVIGATE THE CODE
NOTE : Every python scripts of the project was placed in the "src" folder.
## I. 1. Global structuration
### ⮕ main.py
The script is designed to run tests or simulations based on the provided command-line arguments: 
```shell
python main.py principal_arg arg_optional
```

The following arguments are available:

- ***-t*** (set as a principal argument) : Launches the run_tests(verbose) function defined in the MAIN.PY script. It initializes a series of test flags corresponding to specific features and functionalities within different modules. Each flag represents a test case, and setting a flag to True enables that particular test to be run. (Almost) Each function of the project has its own test. The test are defined in the TEST.PY script. Each test can be run individuall (for more information see the TEST.PY explnation below).
- ***-s*** (set as prinicpal argument) : Launches the run_simulation(verbose) function defined in the INTERFACE.PY script. It launches a simulation with the parameters given in the INTERFACE.PY script.
- ***-v*** (set along with -s or -t as an optional argument) : Launches a detailed version of the program with text explaining the different steps of the algorithm and a timestamping of these steps.

NOTE : Work is still in progress for the verbose of the tests. Consequently, `python main.py -t -v` has not yet an effect.
 
### ⮕ General organization

<center><img src="images/Structuration of the scripts.png" alt="what image shows" width="85%"></center>

## I. 2. Precisions for tests.py, interface.py, launcher.py, variability.py, display_functions.py

### ⮕ tests.py

- Each function in the project, generically named func, corresponds to a single test function, **test_func**.
- Each test function is designed to **debug** and check the robustness of the corresponding code.
- Some tests have not yet been implemented; these can be identified by the **#TODO** comment written next to their flag in the MAIN.PY script.
- All **parameters** for each test are defined directly within the corresponding test function.
- As the TESTS.PY script is lengthy, it is recommended to **fold** all functions when using an editor like Notepad++. You can then unfold one or more functions as needed to check or modify their parameters.

### ⮕ interface.py

- Interface.py is the script where each parameters of the simulation can be defined.
- The parameter `nRandomTICDsets` is still a WIP and can create bugs, it is recommended to keep it to 1 for better results.
- The parameter reference_var is used in two ways. The first is as a reference point for the MPS representation of the indicators. The second is a background_image for the entropy.
- The prefixes correspond to the prefixes of the corresponding output folder.
- The block named "MOVE THAT INTO A FUNCTION" in the `get_simulation_info` function, is used in the case one wants to follow a design of experiment, see [Guide IV](https://github.com/AxMeNi/GeoclassificationMPS/blob/main/Guide%20IV%20%C2%B7%20Launch%20a%20simulation%20following%20a%20DOE.md). In other cases, this block can be removed.
- The execute_shorter_program function is meant to launcher smaller simulation. Work is still in progress for this function, feel free to propose any idea.

## ⮕ launcher.py

- The structure of the launcher as of 12th December 2024 is as follows when a simulation is launched :
   - Retrieving the paramters,
   - Creating the simulation grid mask,
   - If the number of sets for the TI and Conditiong Data is set to 1
      - Creating the TIs, the conditioning data, retrieving the dimensions corresponding to the method to create the TI:
      - Retrieving names and types of the variables
      - Launching the DeeSse simulation
      - Saving the output and the mask
      - Calulate entropy, Jensen-Shannon divergence, and topological adjancecy
      - Calculate standard deviations for the three indicators
      - Plot the three indicators + histograms and proportions
      - Plot standard deviations
  - If the number of sets for the TI and Conditioning Data is higher than 1
      - Generate TIs, CDs
      - Launch DeeSse simulations for each set
      - Save the simulation
      - Calculate the indicators
   
 ## ⮕ variability.py

- Variability.py calls the functions of loop-ui to calculate the indicators used in the project.
- Some functions of loop-ui were reconstructed to fit the requirements of the projects ; they were named as custom_nameoftheoriginalfunction.
- The function calculate_indicators allows to calculate three indicators : the entropy between realizations, the Jensen-Shannon divergence to compare the histograms of the realizations, and the Hamming topological distance between realizations. This function is used to compare realizations only. To compare simulations, see analyze_global_MDS.
- Dividing calculate_indicators in several smaller functions could help simplify the code.
- The analyze_global_MDS function is used to compare multiple simulations (different deesse_output results). It processes an ensemble of dissimilarity matrices computed for a specific indicator, such as Jensen-Shannon divergence or Hamming topological distance, where each matrix corresponds to a single simulation. The function identifies the four most distant points in the MDS representation of each matrix. These quadruplets are then combined into a single set to calculate the "global" Jensen-Shannon divergence and the "global" Hamming topological distance.

## ⮕ display_functions.py

- The display_functions script contains all the functions designed to plot or visualize indicators, simulations, and masks. 
- Colors used in the plots can be modified directly within these functions.
- Titles, legends, size, disposition and symbology can also be modified directly within these functions.





 
