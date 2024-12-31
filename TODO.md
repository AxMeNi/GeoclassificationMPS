# TODO list

### TO DEBUG
- [ ] The function plot_mask in display_functions (works with the test function, but not with the masks used in the simulation)
---
### WIP
- [ ] The case nRandomTICDsets > 1 in launcher.py (still few bugs that require a complete inspection of the current implementation)
- [ ] Adjust the colors and/or the symbology of the global MDS plot to emphasize the differences
- [ ] Modify the CSV file header for the data_csv (remove the "grid" column or find a utility)
- [ ] An optimization of the method ReducedTiSg in reduced_ti_sg.py to make it run faster.
---
### TO IMPLEMENT
- [ ] All tests flagged as #TODO in main.py
- [ ] The function execute_shorter_program in interface.py
- [ ] Adding the parameter numberOfMasks (which changes the seed).
- [ ] Dividing the function calculate_indicators in variability.py in several smaller functions.
---
### LIST OF WARNINGS AND BUGS ENCOUNTERED
- [ ] When launching a simulation with multiple TIs, or when launching with the parameter ReducedTiSg, we get this error : `Invalid data in simulation grid : Too far beyond the range covered by all the TI(s) : continuous variable`
- [ ] Sometimes, we get this error message : `Invalaid data in simulation grid : Not in TI(s)`
- [ ] When launching on Kaya : `WARNING 00010: a variable in a training image is not exhaustively informed`
- [ ] When launching on Kaya : `WARNING 00003: training image values of a variable have been rescaled (linearly) to cover the range of conditioning data values`
- [ ] When launching on Kaya : `WARNING 00020: a node is chosen (in the training image) randomly, ignoring neighborhood information`
- [ ] For some combinations of parameters (especially, when 3 auxliliary variables are used), the program returns : `save_plot : line 156 : File is not a zip file`. This bug can be related to the way the plot are stored (and especially the format used).
---
