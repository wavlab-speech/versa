#!/bin/bash
#SBATCH -p ghx4
#SBATCH --time=2-0:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60000M
#SBATCH --gres=gpu:1
#SBATCH --account=bbjs-dtai-gh
#SBATCH --job-name=qwen_std
#SBATCH --output=qwen_std_%j.out
#SBATCH --error=qwen_std_%j.err

# Print some information about the job
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "Allocated SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "Allocated CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Activate your conda environment if needed
# Replace 'your_env_name' with your actual environment name
# source activate your_env_name

# Set up any environment variables
export PYTHONUNBUFFERED=1
# export TRANSFORMERS_CACHE=/path/to/cache  # Set this to your preferred cache location

# Define paths
INPUT_DIR="/path/to/your/qwen_outputs"      # Replace with your input directory
OUTPUT_DIR="/path/to/your/standardized_outputs"  # Replace with your output directory
LOG_FILE="$OUTPUT_DIR/standardization_$(date +%Y%m%d_%H%M%S).log"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the standardization script
# Replace with actual paths to your script and any other parameters
python batch_processing_script.py \
    --directory $INPUT_DIR \
    --output $OUTPUT_DIR \
    --workers $SLURM_CPUS_PER_TASK \
    --log-file $LOG_FILE \
    --log-level INFO

# Check if the job completed successfully
if [ $? -eq 0 ]; then
    echo "Job completed successfully at $(date)"
else
    echo "Job failed at $(date)"
    exit 1
fi

# Optional: Send email notification when job is complete
# mail -s "Qwen Standardization Job Complete" your.email@example.com <<< "Your standardization job has completed."

# Print a summary of the results
echo "Standardization complete. Results are in $OUTPUT_DIR"
echo "Log file: $LOG_FILE"

# Print runtime statistics
echo "Job finished at $(date)"
