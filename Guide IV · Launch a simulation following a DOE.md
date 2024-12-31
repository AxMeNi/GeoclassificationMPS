# Guide IV  · LAUNCH A SIMULATION FOLLOWING A DESIGN OF EXPERIMENT
The key idea is that for each desired combination of parameters, the program will launch a new batch job via a slurm [array](https://slurm.schedmd.com/sbatch.html#OPT_array).
## IV. 1. Adapt the interface.py script to allow changes via the batch script
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
### ⮕ Adapt the variable values:
- For the following parameters:
  - `seed`
  - `ti_pct_area`
  - `ti_nshapes`
  - `nRandomTICDsets`

   It is required to set their values to, respectively : `seed = arg_seed`, `ti_pct_area = arg_ti_pct_area`, `ti_nshape = arg_num_shape` and `nRandomTICDsets = arg_n_ti`. The reason behind this is that those parameters will be changed for each job. Therefore, they are modified directly in the batch script, and their values are then submitted when excuting the INTERFACE.PY script.
- Any other changes must be done before uploading the file to the distant storage.
## IV. 2. Prepare the environment
See [III. 2.](https://github.com/AxMeNi/GeoclassificationMPS/edit/main/Guide%20III%20%C2%B7%20Launch%20a%20simulation%20on%20a%20HPC.md#iii-2-prepare-the-environment)
## IV. 3. Launch the simulations with a batch script
### ⮕ Adapt the batch script to your project
- One example of a batch script following a design of exeperiment can be found in the folder **batch_scripts**. It is recommended to use **bs_doe.sh** for trials.
- Below is a small explanation of what the script does:
 ```batch
  #!/bin/bash
  #SBATCH --partition=work
  #SBATCH --job-name=geoclassification_mps
  #SBATCH --cpus-per-task=4
  #SBATCH --output=log/test_doe_job_task_%a.txt
  #SBATCH --array=1-140
  ```
  These are [Slurm batch](https://slurm.schedmd.com/sbatch.html) directives. They define the characteristics of the job, it is recommended to adapt them to the desired job. NOTE : The [```--array```](https://slurm.schedmd.com/sbatch.html#OPT_array) parameter is for submitting multiple jobs to be executed with identical parameters by using this unique batch script. Each job is assigned a unique job ID, which, in the latter, will be referred to as **JOBID**.
  ```batch
  # Definiion of the parameters of the design of experiment
  SEED=852
  NUM_TI_LIST=(1)
  TI_PCT_AREA_LIST=(25 55 75 90)
  NUM_SHAPE_LIST=(1 5 10 15 50)
  AUX_VARS_LIST=(
      "grid_grv"
      "grid_lmp"
      "grid_mag"
      "grid_grv,grid_lmp"
      "grid_grv,grid_mag"
      "grid_lmp,grid_mag"
      "grid_grv,grid_lmp,grid_mag"
  )
   ```
