# Guide III  ·  LAUNCH SIMULATIONS ON A HIGH PERFORMANCE COMPUTER
## III. 1. Adapt the interface.py script to the desired job
- The INTERFACE.PY script must be adapted to the desired job. It is recommended to follow the dedicated paragraph in [Guide II](https://github.com/AxMeNi/GeoclassificationMPS/blob/main/Guide%20II%20%C2%B7%20Launch%20simulations.md#ii-3-provide-the-parameters) to see which changes can be operated.
- Make sure to adapt the different paths to the distant storage path.
- To test the system, one can use small parameters and small data sets to facilitate the work.
## III. 2. Prepare the environment
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
## III. 3. Launch a batch job with a batch script
### ⮕ Adapt the batch script to your project
- One example of a batch script can be found in the folder **batch_scripts**. It is recommended to use **bs_simple.sh** for trials.
- Below is a small explanation of what the script does:
  ```batch
  #!/bin/bash
  #SBATCH --partition=work
  #SBATCH --job-name=geoclassification_mps
  #SBATCH --cpus-per-task=4
  #SBATCH --output=log/test_job_task.txt
  ```
  These are [Slurm batch](https://slurm.schedmd.com/sbatch.html) directives. They define the characteristics of the job, it is recommended to adapt them to the desired job. 
  ```batch
  python path/to/main.py
  ```
  This line executes MAIN.PY.
### ⮕ Launch the script
- Type the following command line in the terminal to execute the batch job with the provided features:
  ```shell
  sbatch bs_simple.sh
  ```
## III. 4. Follow the execution of the job and retrieve the output in the provided folder
- Use [```squeue -u username```](https://slurm.schedmd.com/squeue.html)
  to check the status of the job.
- Each output files can be retrievd in the folder given in this [line](https://github.com/AxMeNi/GeoclassificationMPS/blob/d9b0fac87d78e6428cfd4703df29ef539c6bbb55/src/interface.py#L110).
