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
PREDICTION_MODELS=("randomforestregressor")  # elasticnet, randomforestregressor
CRITERIA=("state_wb")             # state_wb, state_pa, state_na, trait_wb, trait_na, trait_pa
FEATURE_COMBINATIONS=("srmc" "mac")     # pl, srmc, sens, mac, etc.
SAMPLES_TO_INCLUDE=("selected")   # all, selected, control

# Parameters
COMP_SHAP_IA_VALUES="true"
PARALLELIZE_INNER_CV="true"
PARALLELIZE_SHAP="true"
PARALLELIZE_SHAP_IA_VALUES="true"
PARALLELIZE_IMPUTATION_RUNS="true"

# New parameters
SPLIT_REPS="true"    # If "true", split repetitions into separate jobs
NUM_REPS=10           # Adjust as needed

BASE_MINUTES=2000
CPUS_PER_TASK=10     # Fixed number of CPUs per analysis
NUM_NODES=1          # If set to 1, no multi-node analysis happens

# Memory specification based on COMP_SHAP_IA_VALUES
if [ "$COMP_SHAP_IA_VALUES" == "true" ]; then
  MEMORY_REQUEST="#SBATCH --mem-per-cpu=15G"
else
  MEMORY_REQUEST=""
fi

# TODO: Change when using different accounts
BASE_DIR="/scratch/tmp/nkuper2/coco_wb_ml_code/ia_2011"  # ia_2011, fs_2011
ENV_PATH="/scratch/tmp/nkuper2/coco_wb_ml_code/palma_env/bin/activate"
PYTHONPATH_BASE="/scratch/tmp/nkuper2/coco_wb_ml_code"

# Current Time
CURRENT_TIME=$(date +%Y%m%d%H%M%S)

# Loop over all combinations
for crit in "${CRITERIA[@]}"; do
  for prediction_model in "${PREDICTION_MODELS[@]}"; do

    # Set PRED_MODEL_MULT based on prediction_model
    case $prediction_model in
      "randomforestregressor") PRED_MODEL_MULT=1 ;;
      "elasticnet") PRED_MODEL_MULT=1 ;;
    esac

    for feature_combination in "${FEATURE_COMBINATIONS[@]}"; do
      for samples_to_include in "${SAMPLES_TO_INCLUDE[@]}"; do

        # Set SAMPLE_MULT based on samples_to_include
        case $samples_to_include in
          "all") SAMPLE_MULT=1 ;;
          "selected") SAMPLE_MULT=1 ;;
          "control") SAMPLE_MULT=1 ;;
        esac

        # Check if "sens" is in feature_combination and set FEATURE_MULT accordingly
        if [[ $feature_combination == *"sens"* ]]; then
          FEATURE_MULT=1
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

        # Decide on DISTRIBUTED, NTASKS
        if [ "$NUM_NODES" -eq 1 ]; then
          DISTRIBUTED=false
          NTASKS=1
        else
          DISTRIBUTED=true
          NTASKS=1 # Adjust if necessary
        fi

        # Set RUN_COMMAND
        RUN_COMMAND="python main.py"

        if [ "$SPLIT_REPS" == "true" ]; then
          for rep_id in $(seq 0 $(($NUM_REPS - 1))); do

            JOB_NAME="${feature_combination}_${samples_to_include}_${crit}_${prediction_model}_rep${rep_id}"

            JOB_LOG_NAME="job_${CURRENT_TIME}_rep${rep_id}"
            FULL_LOG_PATH_LOG="${LOG_DIR}/${JOB_LOG_NAME}.log"
            FULL_LOG_PATH_ERR="${LOG_DIR}/${JOB_LOG_NAME}.err"

            # SLURM script filename
            SLURM_SCRIPT="${LOG_DIR}/slurm_script_${JOB_LOG_NAME}.sh"  # Save SLURM script in the same directory

            # Create a SLURM script
            cat > $SLURM_SCRIPT << EOF
#!/bin/bash

# SLURM Directives
#SBATCH --nodes=$NUM_NODES
#SBATCH --ntasks-per-node=$NTASKS
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --partition=long
#SBATCH --time=$TIMELIMIT
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aeback.oh@gmail.com
#SBATCH -J $JOB_NAME
#SBATCH --output=${FULL_LOG_PATH_LOG}
#SBATCH --error=${FULL_LOG_PATH_ERR}
$MEMORY_REQUEST

# Load Modules
module purge
module load palma/2023a
module load GCCcore/12.3.0
# module load Python/3.11.3

module load palma/2023b
module load GCC/13.2.0
module load OpenMPI/4.1.6

module load mpi4py

# Activate your Python environment
unset PYTHONPATH
source $ENV_PATH
export PYTHONPATH=${PYTHONPATH_BASE}:${PYTHONPATH_BASE}/src/:\$PYTHONPATH

# Navigate to script directory
cd ${PYTHONPATH_BASE}/src/

# Debugging statements
echo "Python executable: \$(which python)"
echo "Python version: \$(python --version)"
echo "PYTHONPATH: \$PYTHONPATH"
echo "SPLIT_REPS: $SPLIT_REPS"
echo "rep_id: $rep_id"

# Run the program
$RUN_COMMAND \\
    --prediction_model "$prediction_model" \\
    --crit "$crit" \\
    --feature_combination "$feature_combination" \\
    --samples_to_include "$samples_to_include" \\
    --comp_shap_ia_values "$COMP_SHAP_IA_VALUES" \\
    --parallelize_inner_cv "$PARALLELIZE_INNER_CV" \\
    --parallelize_shap_ia_values "$PARALLELIZE_SHAP_IA_VALUES" \\
    --parallelize_shap "$PARALLELIZE_SHAP" \\
    --parallelize_imputation_runs "$PARALLELIZE_IMPUTATION_RUNS" \\
    --split_reps "$SPLIT_REPS" \\
    --rep "$rep_id" \\
    --output_path "$RESULT_DIR/"
EOF

            # Submit the SLURM job
            sbatch $SLURM_SCRIPT

          done
        else
          # When SPLIT_REPS is not "true", create a single job without repetitions
          JOB_NAME="${feature_combination}_${samples_to_include}_${crit}_${prediction_model}"

          JOB_LOG_NAME="job_${CURRENT_TIME}"
          FULL_LOG_PATH_LOG="${LOG_DIR}/${JOB_LOG_NAME}.log"
          FULL_LOG_PATH_ERR="${LOG_DIR}/${JOB_LOG_NAME}.err"

          # SLURM script filename
          SLURM_SCRIPT="${LOG_DIR}/slurm_script_${JOB_LOG_NAME}.sh"

          # Create a SLURM script
          cat > $SLURM_SCRIPT << EOF
#!/bin/bash

# SLURM Directives
#SBATCH --nodes=$NUM_NODES
#SBATCH --ntasks-per-node=$NTASKS
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --partition=normal
#SBATCH --time=$TIMELIMIT
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aeback.oh@gmail.com
#SBATCH -J $JOB_NAME
#SBATCH --output=${FULL_LOG_PATH_LOG}
#SBATCH --error=${FULL_LOG_PATH_ERR}

# Load Modules
module purge
module load palma/2023a
module load GCCcore/12.3.0
# module load Python/3.11.3

module load palma/2023b
module load GCC/13.2.0
module load OpenMPI/4.1.6

module load mpi4py

# Activate your Python environment
unset PYTHONPATH
source $ENV_PATH
export PYTHONPATH=${PYTHONPATH_BASE}:${PYTHONPATH_BASE}/src/:\$PYTHONPATH

# Navigate to script directory
cd ${PYTHONPATH_BASE}/src/

# Debugging statements
echo "Python executable: \$(which python)"
echo "Python version: \$(python --version)"
echo "PYTHONPATH: \$PYTHONPATH"
echo "SPLIT_REPS: $SPLIT_REPS"

# Run the program
$RUN_COMMAND \\
    --prediction_model "$prediction_model" \\
    --crit "$crit" \\
    --feature_combination "$feature_combination" \\
    --samples_to_include "$samples_to_include" \\
    --comp_shap_ia_values "$COMP_SHAP_IA_VALUES" \\
    --parallelize_inner_cv "$PARALLELIZE_INNER_CV" \\
    --parallelize_shap_ia_values "$PARALLELIZE_SHAP_IA_VALUES" \\
    --parallelize_shap "$PARALLELIZE_SHAP" \\
    --parallelize_imputation_runs "$PARALLELIZE_IMPUTATION_RUNS" \\
    --split_reps "$SPLIT_REPS" \\
    --output_path "$RESULT_DIR/"
EOF

          # Submit the SLURM job
          sbatch $SLURM_SCRIPT

        fi

      done
    done
  done
done