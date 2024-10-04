#!/bin/bash

: '
This SLURM script is used to generate multiple jobs at a time for a given analysis setting.

Key parameters:
- BASE_MINUTES and BASE_CPUS are base values adjusted dynamically based on the analysis type.

Changeable variables/settings:
- prediction_model ("elasticnet", "randomforestregressor")
- crit ("state_wb", "state_pa", "state_na", "trait_wb", "trait_na", "trait_pa")
- feature_combination ("pl", "srmc", "sens", "mac", etc.)
- samples_to_include ("all", "selected", "control")
'

# Variables
PREDICTION_MODELS=("elasticnet")  # elasticnet, randomforestregressor
CRITERIA=("state_wb")             # state_wb, state_pa, state_na, trait_wb, trait_na, trait_pa
FEATURE_COMBINATIONS=("pl")       # pl, srmc, sens, mac, etc.
SAMPLES_TO_INCLUDE=("all")        # all, selected, control

# Parameters
COMP_SHAP_IA_VALUES="true"
PARALLELIZE_INNER_CV="true"
PARALLELIZE_SHAP="true"
PARALLELIZE_SHAP_IA_VALUES="true"
PARALLELIZE_IMPUTATIONS="true"

BASE_MINUTES=10
BASE_CPUS=4

# Base Directory for Results
BASE_DIR="/scratch/hpc-prf-mldpr/tests_cocowb_012024/"

# Current Time
CURRENT_TIME=$(date +%Y%m%d%H%M%S)

# Loop over all combinations
for crit in "${CRITERIA[@]}"; do
  for prediction_model in "${PREDICTION_MODELS[@]}"; do

    # Set PRED_MODEL_MULT based on prediction_model
    case $prediction_model in
      "rfr") PRED_MODEL_MULT=4 ;;
      "elasticnet") PRED_MODEL_MULT=1 ;;
    esac

    for feature_combination in "${FEATURE_COMBINATIONS[@]}"; do
      for samples_to_include in "${SAMPLES_TO_INCLUDE[@]}"; do

        # Set SAMPLE_MULT based on samples_to_include
        case $samples_to_include in
          "all") SAMPLE_MULT=4 ;;
          "selected") SAMPLE_MULT=2 ;;
          "control") SAMPLE_MULT=1 ;;
        esac

        TOTAL_MINUTES=$((BASE_MINUTES * PRED_MODEL_MULT * SAMPLE_MULT))
        TOTAL_CPUS=$((BASE_CPUS * PRED_MODEL_MULT))

        # Cap TOTAL_CPUS at 40
        if [ $TOTAL_CPUS -gt 40 ]; then
          TOTAL_CPUS=40
        fi

        # Convert the total minutes to HH:MM:SS format
        HOURS=$((TOTAL_MINUTES / 60))
        MINUTES=$((TOTAL_MINUTES % 60))
        TIMELIMIT=$(printf "%02d:%02d:00" $HOURS $MINUTES)

        RESULT_DIR="${BASE_DIR}/${feature_combination}/${samples_to_include}/${crit}/${prediction_model}"
        mkdir -p "$RESULT_DIR"

        # Create log directory
        LOG_BASE_DIR="slurm_logs"
        LOG_DIR="${LOG_BASE_DIR}/${feature_combination}/${samples_to_include}/${crit}/${prediction_model}"
        mkdir -p "$LOG_DIR"

        JOB_LOG_NAME="job_${prediction_model}_${crit}_${feature_combination}_${samples_to_include}_${CURRENT_TIME}"
        FULL_LOG_PATH_LOG="${LOG_DIR}/${JOB_LOG_NAME}.log"
        FULL_LOG_PATH_ERR="${LOG_DIR}/${JOB_LOG_NAME}.err"

        # SLURM script filename
        SLURM_SCRIPT="slurm_script_${JOB_LOG_NAME}.sh"

        # Create a SLURM script
        cat > $SLURM_SCRIPT << EOF
#!/bin/bash

# SLURM Directives
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=$TOTAL_CPUS
#SBATCH -p normal
#SBATCH -t $TIMELIMIT
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aeback.oh@gmail.com
#SBATCH -J ${JOB_LOG_NAME}
#SBATCH --output=${FULL_LOG_PATH_LOG}
#SBATCH --error=${FULL_LOG_PATH_ERR}

# Load Modules
module load python

# Your Python analysis script, with arguments
python main.py \\
    --prediction_model "$prediction_model" \\
    --crit "$crit" \\
    --feature_combination "$feature_combination" \\
    --samples_to_include "$samples_to_include" \\
    --comp_shap_ia_values "$COMP_SHAP_IA_VALUES" \\
    --parallelize_inner_cv "$PARALLELIZE_INNER_CV" \\
    --parallelize_shap_ia_values "$PARALLELIZE_SHAP_IA_VALUES" \\
    --parallelize_shap "$PARALLELIZE_SHAP" \\
    --parallelize_imputations "$PARALLELIZE_IMPUTATIONS" \\
    --output_path "$RESULT_DIR/"
EOF

        # Submit the SLURM job
        sbatch $SLURM_SCRIPT

      done
    done
  done
done









