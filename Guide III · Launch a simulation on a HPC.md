# Guide III  ·  LAUNCH SIMULATIONS ON A HIGH PERFORMANCE COMPUTER
## III. 1. Prepare the environment
### ⮕ Check the files
- Every file needed must be uploaded to the distant storage.
- Check that every file is correctly uploaded
### ⮕ Install the Python Conda environment
- Follow the [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) procedure to install the environment via: [PythonEnvGeoclassifMPS2.yml](https://github.com/AxMeNi/GeoclassificationMPS/blob/main/PythonEnvGeoclassifMPS2.yml).
- Correct any bug if necessary.
### ⮕ Activate the Conda environment
- Type the following command line in the shell where the batch job will be executed:
  ```shell
  conda activate geoclassif
  ```
- Once the environment is activated, a batch job can be launched.
## III. 2. Launch a batch job with a batch script
### ⮕ Adapt the batch script to your project
- One example of a batch script can be found in the folder **batch_scripts**. It is recomanded to use **bs_simple.sh** for trials.
- Below is a small explanation of what the script does:
  ```batch
  #!/bin/bash
  #SBATCH --partition=work
  #SBATCH --job-name=geoclassification_mps
  #SBATCH --cpus-per-task=4
  #SBATCH --output=log/test_array_job_task_%a.txt
  #SBATCH --array=1-30
  ```
  These are [Slurm batch](https://slurm.schedmd.com/sbatch.html) directives. They define the characteristics of the job, it is recommended to adapt them to the desired job. NOTE : The [```--array```](https://slurm.schedmd.com/sbatch.html#OPT_array) parameter is for submitting multiple jobs to be executed with identical parameters by using this unique batch script. Each job is assigned a unique job ID, which, in the latter, will be referred to as **JOBID**.

  ```batch
  cp path/to/interface.py path/to/interface_${SLURM_ARRAY_TASK_ID}.py
  ```
  This line copies the script INTERFACE.PY to INTERFACE_JOBID.PY to allow some changes in the interface.py file without changing the original version.
  ```batch
  sed -i "s/seed = 852/seed = ${SLURM_ARRAY_TASK_ID}/" path/to/interface_${SLURM_ARRAY_TASK_ID}.py
  ```
  This line modifies the value of the seed which will now be equal to the JOBID.
  ```batch
  python path/to/interface_${SLURM_ARRAY_TASK_ID}.py
  ```
  This line executes the modified copy of INTERFACE.PY.
  ```batch
  rm path/to/interface_${SLURM_ARRAY_TASK_ID}.py
  ```
  This line removes the copy from the storage.
