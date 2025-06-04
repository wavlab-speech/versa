#!/bin/bash
#
# GPU Score Runner for VERSA Processing
# ------------------------------------
# This script processes audio files using GPU acceleration
#
# Usage: ./run_gpu.sh <pred_wavscp> <gt_wavscp> <output_file> <config_file> [io_type] [text_file] [gpu_rank]
set -e  # Exit immediately if a command exits with non-zero status

# Check if minimum required arguments are provided
if [ $# -lt 4 ]; then
    echo "Error: Insufficient number of arguments"
    echo "Usage: $0 <pred_wavscp> <gt_wavscp> <output_file> <config_file> [io_type] [text_file] [gpu_rank]"
    echo "  pred_wavscp: Path to prediction wav.scp file"
    echo "  gt_wavscp: Path to ground truth wav.scp file (use 'None' if not available)"
    echo "  output_file: Path to output results file"
    echo "  config_file: Path to configuration file"
    echo "  io_type: Optional parameter - soundfile (default), dir, or kaldi"
    echo "  text_file: Optional path to text file for processing"
    echo "  gpu_rank: Optional GPU rank/device ID (0, 1, 2, etc.)"
    exit 1
fi

# Parse command line arguments
PRED=$1
GT=$2
OUTPUT=$3
CONFIG=$4

# Set default IO type if not provided
IO_TYPE=${5:-soundfile}

# Optional text file parameter
TEXT_FILE=${6:-}

# Optional GPU rank parameter
GPU_RANK=${7:-}

# Validate IO type
if [[ "$IO_TYPE" != "soundfile" && "$IO_TYPE" != "dir" && "$IO_TYPE" != "kaldi" ]]; then
    echo "Error: Invalid IO type '$IO_TYPE'"
    echo "Valid options are: soundfile, dir, kaldi"
    exit 1
fi

# Validate text file if provided
if [ -n "$TEXT_FILE" ] && [ "$TEXT_FILE" != "" ] && [ ! -f "$TEXT_FILE" ]; then
    echo "Error: Text file '$TEXT_FILE' not found"
    exit 1
fi

# Validate GPU rank if provided
if [ -n "$GPU_RANK" ] && [ "$GPU_RANK" != "" ]; then
    if ! [[ "$GPU_RANK" =~ ^[0-9]+$ ]]; then
        echo "Error: GPU rank must be a non-negative integer"
        exit 1
    fi
    echo "GPU rank set to: $GPU_RANK"
fi

# Display job information
echo "==== GPU Job Information ===="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo "Prediction file: $PRED"
echo "Ground truth file: $GT"
echo "Output file: $OUTPUT"
echo "Config file: $CONFIG"
echo "IO type: $IO_TYPE"
if [ -n "$TEXT_FILE" ] && [ "$TEXT_FILE" != "" ]; then
    echo "Text file: $TEXT_FILE"
else
    echo "Text file: Not provided"
fi
if [ -n "$GPU_RANK" ] && [ "$GPU_RANK" != "" ]; then
    echo "GPU rank: $GPU_RANK"
else
    echo "GPU rank: Auto-detect"
fi
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-Auto-detect}"
echo "============================="

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT")
mkdir -p "$OUTPUT_DIR"

# Run the scoring script with GPU acceleration
echo "Starting GPU processing at $(date)"

# Build the command with optional text parameter
CMD_ARGS=(
    "--pred" "${PRED}"
    "--gt" "${GT}"
    "--io" "${IO_TYPE}"
    "--output_file" "${OUTPUT}"
    "--score_config" "${CONFIG}"
    "--use_gpu" "true"
)

# Add text file argument if provided
if [ -n "$TEXT_FILE" ] && [ "$TEXT_FILE" != "" ]; then
    CMD_ARGS+=("--text" "${TEXT_FILE}")
    echo "Including text file in processing: $TEXT_FILE"
fi

# Add GPU rank argument if provided
if [ -n "$GPU_RANK" ] && [ "$GPU_RANK" != "" ]; then
    CMD_ARGS+=("--rank" "${GPU_RANK}")
    echo "Using GPU rank: $GPU_RANK"
fi

# Set up error handling and run the scoring script
if ! python versa/bin/scorer.py "${CMD_ARGS[@]}"; then
    echo "Error: GPU scoring failed with exit code $?"
    exit 1
fi

echo "GPU processing completed at $(date)"
echo "Results saved to: $OUTPUT"

# Optional: Append completion summary to a tracking file
JOB_TRACK_FILE=$(dirname "$OUTPUT")/../job_status.txt
if [ -n "$GPU_RANK" ] && [ "$GPU_RANK" != "" ]; then
    gpu_info=" - GPU: $GPU_RANK"
else
    gpu_info=""
fi

if [ -n "$TEXT_FILE" ] && [ "$TEXT_FILE" != "" ]; then
    echo "$(date) - GPU Job ${SLURM_JOB_ID:-LOCAL} complete - $(basename "$PRED") - IO: $IO_TYPE - Text: $(basename "$TEXT_FILE")$gpu_info" >> "$JOB_TRACK_FILE"
else
    echo "$(date) - GPU Job ${SLURM_JOB_ID:-LOCAL} complete - $(basename "$PRED") - IO: $IO_TYPE$gpu_info" >> "$JOB_TRACK_FILE"
fi
