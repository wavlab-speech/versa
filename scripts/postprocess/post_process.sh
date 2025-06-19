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
# OUTPUT_DIR="/work/nvme/bbjs/shi3/evaluation/espnet/tools/versa/qwen_voicebank-nisqa-voicemos/standard_result/"
# INPUT_DIR="/work/nvme/bbjs/shi3/evaluation/espnet/tools/versa/qwen_voicebank-nisqa-voicemos/result/voicebank-nisqa-voicemos.scp_$1.result.gpu.txt"
# OUTPUT_PATH=${OUTPUT_DIR}/voicebank-nisqa-voicemos.scp_$1.result.gpu.txt
# OUTPUT_DIR="/work/nvme/bbjs/shi3/evaluation/espnet/tools/versa/qwen_urgent24/standard_result"
# INPUT_DIR="/work/nvme/bbjs/shi3/evaluation/espnet/tools/versa/qwen_urgent24/result/urgent_wav.scp_$1.result.gpu.txt"
# OUTPUT_PATH=${OUTPUT_DIR}/urgent_wav.scp_$1.result.gpu.txt

# OUTPUT_DIR="/work/nvme/bbjs/shi3/evaluation/espnet/tools/versa/owsm_part1/standard_result"
# INPUT_DIR="/work/nvme/bbjs/shi3/evaluation/espnet/tools/versa/owsm_part1/result/owsm_all_wav_part1.scp_$1.result.gpu.txt"
# OUTPUT_PATH=${OUTPUT_DIR}/owsm_all_wav_part1.scp_$1.result.gpu.txt

# OUTPUT_DIR=/work/nvme/bbjs/shi3/evaluation/espnet/tools/versa/qwen_voicebank-nisqa_test/standard_result
# INPUT_DIR="/work/nvme/bbjs/shi3/evaluation/espnet/tools/versa/qwen_voicebank-nisqa_test/result/voicebank-nisqa-voicemos_test.scp_${1}.result.cpu.txt"
# OUTPUT_PATH=${OUTPUT_DIR}/voicebank-nisqa-voicemos_test_$1.result.gpu.txt

# OUTPUT_DIR=/work/nvme/bbjs/shi3/evaluation/espnet/tools/versa/qwen_urgent_train_all/standard_result
# INPUT_DIR=/work/nvme/bbjs/shi3/evaluation/espnet/tools/versa/qwen_urgent_train_all/result/urgent_train_noisy.scp_$1.result.gpu.txt
# OUTPUT_PATH=${OUTPUT_DIR}/urgent_train_noisy.scp_$1.result.gpu.txt

OUTPUT_DIR=/work/nvme/bbjs/shi3/evaluation/espnet/tools/versa/qwen_emotion/standard_result
INPUT_DIR=/work/nvme/bbjs/shi3/evaluation/espnet/tools/versa/qwen_emotion/result/emotion.scp_$1.result.cpu.txt
OUTPUT_PATH=${OUTPUT_DIR}/emotion.scp_$1.result.cpu.txt

LOG_FILE="$OUTPUT_DIR/../logs/standardization_$(date +%Y%m%d_%H%M%S).log"
# MODEL="Qwen/Qwen2.5-7B-Instruct-1M"
# MODEL="Qwen/Qwen2.5-7B"
MODEL=None
BATCH_SIZE=64
PATTERN="*.txt"


# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/../logs

# Run the standardization script
# Replace with actual paths to your script and any other parameters
python qwen2_audio_jsonl_standardizer_batch.py \
    $INPUT_DIR \
    --output $OUTPUT_PATH \
    --batch-size $BATCH_SIZE \
    --model $MODEL \
    --device "cuda" \
    --pattern "*.txt" \
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
