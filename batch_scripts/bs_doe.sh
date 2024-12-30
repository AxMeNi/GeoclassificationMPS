#!/bin/bash
#SBATCH --partition=work
#SBATCH --job-name=geoclassification_mps
#SBATCH --cpus-per-task=4
#SBATCH --output=log/test_doe_job_task_%a.txt
#SBATCH --array=1-140  # TEST TEST

# Définition des paramètres du plan d'expérience
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


# Calcul du nombre total de combinaisons
NUM_TI_COUNT=${#NUM_TI_LIST[@]}
TI_PCT_AREA_COUNT=${#TI_PCT_AREA_LIST[@]}
NUM_SHAPE_COUNT=${#NUM_SHAPE_LIST[@]}
AUX_VARS_COUNT=${#AUX_VARS_LIST[@]}

# Calcul des indices pour chaque paramètre
IDX=$((SLURM_ARRAY_TASK_ID - 1)) # Indice de la tâche, en partant de 0
IDX_TI=$((IDX % NUM_TI_COUNT))
IDX_AREA=$(((IDX / NUM_TI_COUNT) % TI_PCT_AREA_COUNT))
IDX_SHAPE=$(((IDX / (NUM_TI_COUNT * TI_PCT_AREA_COUNT)) % NUM_SHAPE_COUNT))
IDX_AUX_VARS=$(((IDX / (NUM_TI_COUNT * TI_PCT_AREA_COUNT * NUM_SHAPE_COUNT)) % AUX_VARS_COUNT))

# Récupération des paramètres correspondants
NUM_TI=${NUM_TI_LIST[$IDX_TI]}
TI_PCT_AREA=${TI_PCT_AREA_LIST[$IDX_AREA]}
NUM_SHAPE=${NUM_SHAPE_LIST[$IDX_SHAPE]}
AUX_VARS=${AUX_VARS_LIST[$IDX_AUX_VARS]}

# Exécution du script Python avec les paramètres générés
python /group/ses001/amengelle/GeoclassificationMPS/src/interface.py \
    --seed ${SEED} \
    --n_ti ${NUM_TI} \
    --ti_pct_area ${TI_PCT_AREA} \
    --num_shape ${NUM_SHAPE} \
    --aux_vars "${AUX_VARS}"
