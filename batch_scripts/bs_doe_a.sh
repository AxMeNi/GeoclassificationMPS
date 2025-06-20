#!/bin/bash
#SBATCH --partition=cet
#SBATCH --job-name=mps_clf
#SBATCH --cpus-per-task=4
#SBATCH --output=log/mps_clf_task_%a.txt
#SBATCH --array=1-280

# Definiion of the parameters of the design of experiment
SEED_LIST=(852 123)
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

# Calulcation of the total number of parameters
SEED_COUNT=${#SEED_LIST[@]}
NUM_TI_COUNT=${#NUM_TI_LIST[@]}
TI_PCT_AREA_COUNT=${#TI_PCT_AREA_LIST[@]}
NUM_SHAPE_COUNT=${#NUM_SHAPE_LIST[@]}
AUX_VARS_COUNT=${#AUX_VARS_LIST[@]}

# Calculation of the index for each parameter
IDX=$((SLURM_ARRAY_TASK_ID - 1)) # Index of the task, starting from 0
IDX_TI=$((IDX % NUM_TI_COUNT))
IDX_AREA=$(((IDX / NUM_TI_COUNT) % TI_PCT_AREA_COUNT))
IDX_SHAPE=$(((IDX / (NUM_TI_COUNT * TI_PCT_AREA_COUNT)) % NUM_SHAPE_COUNT))
IDX_AUX_VARS=$(((IDX / (NUM_TI_COUNT * TI_PCT_AREA_COUNT * NUM_SHAPE_COUNT)) % AUX_VARS_COUNT))
IDX_SEED=$(((IDX / (NUM_TI_COUNT * TI_PCT_AREA_COUNT * NUM_SHAPE_COUNT * AUX_VARS_COUNT)) % SEED_COUNT))

# Calulation of the parameters values
SEED=${SEED_LIST[$IDX_LIST]}
NUM_TI=${NUM_TI_LIST[$IDX_TI]}
TI_PCT_AREA=${TI_PCT_AREA_LIST[$IDX_AREA]}
NUM_SHAPE=${NUM_SHAPE_LIST[$IDX_SHAPE]}
AUX_VARS=${AUX_VARS_LIST[$IDX_AUX_VARS]}
SEED=${SEED_LIST[$IDX_SEED]}

OUTPUT_DIR="./output_mps_clf_${SLURM_ARRAY_TASK_ID}_${AUX_VARS}_PCT_${TI_PCT_AREA}_NSHP_${NUM_SHAPE}_SEED_${SEED}"
echo "OUTPUT_DIR ${OUTPUT_DIR}"

# Execution of the Python script with the generated parameters
echo "python src/interface.py  --output_dir ${OUTPUT_DIR} --seed ${SEED} --n_ti ${NUM_TI} --ti_pct_area ${TI_PCT_AREA} --num_shape ${NUM_SHAPE} --aux_vars "${AUX_VARS}"
"
python src/interface.py  --output_dir ${OUTPUT_DIR} --seed ${SEED} --n_ti ${NUM_TI} --ti_pct_area ${TI_PCT_AREA} --num_shape ${NUM_SHAPE} --aux_vars "${AUX_VARS}"
