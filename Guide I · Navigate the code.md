# Guide I  ·  NAVIGATE THE CODE
NOTE : Every python scripts of the project was placed in the "src" folder.
## I. 1. Global structuration
### ⮕ main.py
The script is designed to run tests or simulations based on the provided command-line arguments: 
> python main.py \<principal_arg\> \<arg_optional\>

The following arguments are available:

- ***-t*** (set as a principal argument) : Launches the run_tests(verbose) function defined in the MAIN.PY script. It initializes a series of test flags corresponding to specific features and functionalities within different modules. Each flag represents a test case, and setting a flag to True enables that particular test to be run. (Almost) Each function of the project has its own test. The test are defined in the TEST.PY script. Each test can be run individuall (for more information see the TEST.PY explnation below).
- ***-s*** (set as prinicpal argument) : Launches the run_simulation(verbose) function defined in the INTERFACE.PY script. It launches a simulation with the parameters given in the INTERFACE.PY script.
- ***-v*** (set along with -s or -t as an optional argument) : Launches a detailed version of the program with text explaining the different steps of the algorithm and a timestamping of these steps.

NOTE : Work is still in progress for the verbose of the tests. Consequently, "python main.py -t -v" has not yet an effect.
 
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
- The prefixes correspond to the prefixes of the corresponding output folder.
- The block named "MOVE THAT INTO A FUNCTION" in the get_simulation_info function, is used in [Guide IV](https://github.com/AxMeNi/GeoclassificationMPS/blob/main/Guide%20IV%20%C2%B7%20Launch%20a%20simulation%20following%20a%20DOE.md%20Launch%20a%20simulation%20following%20a%20DOE.md). In other cases, this block can be removed.
