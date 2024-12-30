#!/bin/bash
#SBATCH --partition=work
#SBATCH --job-name=geoclassification_mps
#SBATCH --cpus-per-task=4
#SBATCH --output=log/test_array_job_task_%a.txt
#SBATCH --array=1-30
cp /group/ses001/amengelle/GeoclassificationMPS/src/interface.py /group/ses001/amengelle/GeoclassificationMPS/src/interface_${SLURM_ARRAY_TASK_ID}.py
sed -i "s/seed = 852/seed = ${SLURM_ARRAY_TASK_ID}/" /group/ses001/amengelle/GeoclassificationMPS/src/interface_${SLURM_ARRAY_TASK_ID}.py
python /group/ses001/amengelle/GeoclassificationMPS/src/interface_${SLURM_ARRAY_TASK_ID}.py
rm /group/ses001/amengelle/GeoclassificationMPS/src/interface_${SLURM_ARRAY_TASK_ID}.py