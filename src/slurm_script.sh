#!/bin/bash

: '
This SLURM script is used to generate multiple jobs at a time for a given analysis setting.

Key parameters:
- BASE_MINUTES is adjusted dynamically based on the analysis type.
- CPUS_PER_TASK is fixed for all analyses.

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
COMP_SHAP_IA_VALUES="false"
PARALLELIZE_INNER_CV="true"
PARALLELIZE_SHAP="true"
PARALLELIZE_SHAP_IA_VALUES="true"
PARALLELIZE_IMPUTATION_RUNS="true"

BASE_MINUTES=2000
CPUS_PER_TASK=10  # Fixed number of CPUs per analysis
NUM_NODES=10      # If set to 1, no multi-node analysis happens

# Base Directory for Results
#BASE_DIR="/scratch/hpc-prf-mldpr/tests_cocowb_012024/"
BASE_DIR="/scratch/hpc-prf-mldpr/coco_wb_ml_code/results_run_2210"

# Current Time
CURRENT_TIME=$(date +%Y%m%d%H%M%S)

# Loop over all combinations
for crit in "${CRITERIA[@]}"; do
  for prediction_model in "${PREDICTION_MODELS[@]}"; do

    # Set PRED_MODEL_MULT based on prediction_model
    case $prediction_model in
      "randomforestregressor") PRED_MODEL_MULT=2 ;;
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

        # Check if "sens" is in feature_combination and set MULT accordingly
        if [[ $feature_combination == *"sens"* ]]; then
          FEATURE_MULT=2
        else
          FEATURE_MULT=1
        fi

        # Calculate the total minutes using all the multipliers
        TOTAL_MINUTES=$((BASE_MINUTES * PRED_MODEL_MULT * SAMPLE_MULT * FEATURE_MULT))

        # Convert the total minutes to HH:MM:SS format
        HOURS=$((TOTAL_MINUTES / 60))
        MINUTES=$((TOTAL_MINUTES % 60))
        TIMELIMIT=$(printf "%02d:%02d:00" $HOURS $MINUTES)

        RESULT_DIR="${BASE_DIR}/${feature_combination}/${samples_to_include}/${crit}/${prediction_model}"
        mkdir -p "$RESULT_DIR"

        # Create log directory
        LOG_BASE_DIR="../slurm_logs"
        LOG_DIR="${LOG_BASE_DIR}/${feature_combination}/${samples_to_include}/${crit}/${prediction_model}"
        mkdir -p "$LOG_DIR"

        JOB_LOG_NAME="job_${CURRENT_TIME}"
        FULL_LOG_PATH_LOG="${LOG_DIR}/${JOB_LOG_NAME}.log"
        FULL_LOG_PATH_ERR="${LOG_DIR}/${JOB_LOG_NAME}.err"

        # SLURM script filename
        SLURM_SCRIPT="${LOG_DIR}/slurm_script_${JOB_LOG_NAME}.sh"  # Save SLURM script in the same directory

        # Determine if multi-node parallelism is needed
        NODES=$NUM_NODES

        if [ "$NUM_NODES" -eq 1 ]; then
          DISTRIBUTED=false
          NTASKS=1
        else
          DISTRIBUTED=true
          NTASKS=$NODES
        fi

        # Create a SLURM script
        cat > $SLURM_SCRIPT << EOF
#!/bin/bash

# SLURM Directives
#SBATCH -N $NODES
#SBATCH -n $NTASKS
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH -p normal
#SBATCH -t $TIMELIMIT
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aeback.oh@gmail.com
#SBATCH -J ${feature_combination}_${samples_to_include}_${crit}_${prediction_model}
#SBATCH --output=${FULL_LOG_PATH_LOG}
#SBATCH --error=${FULL_LOG_PATH_ERR}

# Load Modules
module load python

EOF

        if [ "$DISTRIBUTED" == "true" ]; then
          # Add MPI-specific code
          cat >> $SLURM_SCRIPT << EOF
# Load MPI module
module load mpi4py

# Activate your Python environment if needed
# source activate your_python_environment

# Run the MPI program using srun (for multi-node parallelism)
srun python main.py \\
    --prediction_model "$prediction_model" \\
    --crit "$crit" \\
    --feature_combination "$feature_combination" \\
    --samples_to_include "$samples_to_include" \\
    --comp_shap_ia_values "$COMP_SHAP_IA_VALUES" \\
    --parallelize_inner_cv "$PARALLELIZE_INNER_CV" \\
    --parallelize_shap_ia_values "$PARALLELIZE_SHAP_IA_VALUES" \\
    --parallelize_shap "$PARALLELIZE_SHAP" \\
    --parallelize_imputation_runs "$PARALLELIZE_IMPUTATION_RUNS" \\
    --output_path "$RESULT_DIR/"
EOF

        else
          # Single-node execution using srun
          cat >> $SLURM_SCRIPT << EOF
# Activate your Python environment if needed
# source activate your_python_environment

# Run the Python script using srun (for single-node jobs)
srun python main.py \\
    --prediction_model "$prediction_model" \\
    --crit "$crit" \\
    --feature_combination "$feature_combination" \\
    --samples_to_include "$samples_to_include" \\
    --comp_shap_ia_values "$COMP_SHAP_IA_VALUES" \\
    --parallelize_inner_cv "$PARALLELIZE_INNER_CV" \\
    --parallelize_shap_ia_values "$PARALLELIZE_SHAP_IA_VALUES" \\
    --parallelize_shap "$PARALLELIZE_SHAP" \\
    --parallelize_imputation_runs "$PARALLELIZE_IMPUTATION_RUNS" \\
    --output_path "$RESULT_DIR/"
EOF

        fi

        # Submit the SLURM job
        sbatch $SLURM_SCRIPT

      done
    done
  done
done







