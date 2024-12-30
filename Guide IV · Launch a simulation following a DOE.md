# Guide IV  · LAUNCH A SIMULATION FOLLOWING A DESIGN OF EXPERIMENT
## III. 1. Adapt the interface.py script to allow changes via the batch script
### ⮕ Make sure the dedicated paragraph is present
- The following lines:
  ```python
  auxTI_var = {key: value for key, value in auxTI_var_temp.items() if key in arg_aux_vars}
  auxSG_var = {key: value for key, value in auxSG_var_temp.items() if key in arg_aux_vars}
  outputVarFlag = {key: value for key, value in outputVarFlag.items() if key in arg_aux_vars}
  outputVarFlag["grid_geo"]=True
  names_var = [["grid_geo"],arg_aux_vars,arg_aux_vars,[]]
  types_var[1], types_var[2] = types_var[1][:len(arg_aux_vars)], types_var[2][:len(arg_aux_vars)]
  ```
  must be inserted between this line:
  ```python
  sim_var, auxTI_var_temp, auxSG_var_temp, condIm_var = check_variables(sim_var, auxTI_var_temp, auxSG_var_temp, condIm_var, names_var, types_var, novalue)
  ```
  and that line:
  ```python
  nvar = count_variables(names_var)
  ```
  Check the [script](https://github.com/AxMeNi/GeoclassificationMPS/blob/550f1475c31712f36b88f58970c87cfa25ba08e3/src/interface.py#L135) for more clarity.
  
