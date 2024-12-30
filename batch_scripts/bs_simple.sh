#!/bin/bash
#SBATCH --partition=work
#SBATCH --job-name=geoclassification_mps
#SBATCH --cpus-per-task=4
#SBATCH --output=log/test_array_job_task.txt

python path/to/main.py -s
