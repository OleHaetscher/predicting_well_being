#!/bin/bash

: '
This SLURM script is used to generate multiple jobs at a time for a given analysis setting.

For the machine learning analysis, we added compatibility for running the analysis on a supercomputer cluster using
a SLURM script. This SLURM script is used to generate multiple jobs at a time for a given analysis setting.
Key parameters here are BASE_MINUTES and BASE_CPUS. These are the base values for the time and the number
of CPUs a certain job has. These parameters are adjusted dynamically based on the type of analysis.

Changeable variables/settings from the command line or from a SLURM script are the following (should all be specified as arrays)
    - prediction_model ("enr")  # enr, rfr
    - crit ("state_wb")  # state_wb, state_pa, state_na, trait_wb, trait_na, trait_pa
    - sample_combination ("pl")  # pl, srmc, sens, mac, pl_srmc, pl_sens, pl_srmc_sens, pl_mac, pl_srmc_mac, all
    - samples_to_include ("all")  # all, selected, control

'
PREDICTION_MODELS=("elasticnet")  # elasticnet, randomforestregressor
CRITERIA=("state_wb")  # state_wb, state_pa, state_na, trait_wb, trait_na, trait_pa
FEATURE_COMBINATIONS=("pl")  # pl, srmc, sens, mac, pl_srmc, pl_sens, pl_srmc_sens, pl_mac, pl_srmc_mac, all
SAMPLES_TO_INCLUDE=("all")  # all, selected, control

COMP_SHAP_IA_VALUES="true"
PARALLELIZE_INNER_CV="true"
PARALLELIZE_SHAP="true"
PARALLELIZE_SHAP_IA_VALUES="true"

COMP_SHAP_IA_VALUES="true"
PARALLELIZE_INNER_CV="true"
PARALLELIZE_SHAP="true"
PARALLELIZE_SHAP_IA_VALUES="true"

BASE_MINUTES=100
BASE_CPUS=10

# Base Directory for Results
BASE_DIR="/scratch/hpc-prf-mldpr/tests_022024/"

# Loop over all combinations
for crit in "${CRITERIA[@]}"; do
  for prediction_model in "${PREDICTION_MODELS[@]}"; do

    case $prediction_model in
      "rfr") PRED_MODEL_MULT=4
    esac

    for feature_combination in "${FEATURE_COMBINATIONS[@]}"; do
      for samples_to_include in "${SAMPLES_TO_INCLUDE[@]}"; do

        case $samples_to_include in
          "all") SAMPLE_MULT=4
          ;;
          "selected") SAMPLE_MULT=2
          ;;
        esac

        TOTAL_MINUTES=$((BASE_MINUTES * PRED_MODEL_MULT * SAMPLE_MULT))
        TOTAL_CPUS=$((BASE_CPUS * PRED_MODEL_MULT))  # 40 CPUs is the limit, use strategy_mult only for time

        # Convert the total minutes to the HH:MM:SS format
        HOURS=$((TOTAL_MINUTES / 60))
        MINUTES=$((TOTAL_MINUTES % 60))
        TIMELIMIT=$(printf "%02d:%02d:00" $HOURS $MINUTES)

        RESULT_DIR="${BASE_DIR}/${feature_combination}/${samples_to_include}/${crit}/${prediction_model}"
        mkdir -p "$RESULT_DIR"

        # Create also dir for SLURM logs
        LOG_BASE_DIR="slurm_logs"
        LOG_DIR="${LOG_BASE_DIR}/${feature_combination}/${samples_to_include}/${crit}/${prediction_model}"
        JOB_LOG_NAME="${SLURM_JOB_ID}_${CURRENT_TIME}"
        FULL_LOG_PATH_LOG="${LOG_DIR}/${JOB_LOG_NAME}.log"
        FULL_LOG_PATH_ERR="${LOG_DIR}/${JOB_LOG_NAME}.err"

        # Create a temporary SLURM script
        cat > tmp_slurm_script.sh << EOF

#!/bin/bash

# SLURM Directives
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=$TOTAL_CPUS # Using the above computed CPU number for every Job
#SBATCH -p normal
#SBATCH -t $TIMELIMIT  # Using the above computed TIMELIMIT for every Job
#SBATCH --mail-type ALL
#SBATCH --mail-user aeback.oh@gmail.com
#SBATCH -J ${JOB_LOG_NAME}
#SBATCH --output=${FULL_LOG_PATH_LOG}
#SBATCH --error=${FULL_LOG_PATH_ERR}

# LOAD MODULES HERE IF REQUIRED
module load python

# Your Python analysis script, with arguments
python main.py \
    --prediction_model "$prediction_model" \
    --crit "$crit" \
    --feature_combination "$feature_combination" \
    --samples_to_include "$samples_to_include" \
    --comp_shap_ia_values "$COMP_SHAP_IA_VALUES" \
    --parallelize_inner_cv "$PARALLELIZE_INNER_CV" \
    --parallelize_shap_ia_values "$PARALLELIZE_SHAP_IA_VALUES" \
    --parallelize_shap "$PARALLELIZE_SHAP" \
    --parallelize_imputations "$PARALLELIZE_IMPUTATIONS" \
    --output_path "$RESULT_DIR/" \

EOF
      done
    done
  done
done









