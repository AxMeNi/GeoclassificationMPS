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
 
### General organization

