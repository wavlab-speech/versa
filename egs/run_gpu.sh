#!/bin/bash
#
# GPU Score Runner for VERSA Processing
# ------------------------------------
# This script processes audio files using GPU acceleration
#
# Usage: ./run_gpu.sh <pred_wavscp> <gt_wavscp> <output_file> <config_file>

set -e  # Exit immediately if a command exits with non-zero status

# Check if all required arguments are provided
if [ $# -ne 4 ]; then
    echo "Error: Incorrect number of arguments"
    echo "Usage: $0 <pred_wavscp> <gt_wavscp> <output_file> <config_file>"
    exit 1
fi

# Parse command line arguments
PRED=$1
GT=$2
OUTPUT=$3
CONFIG=$4

# Display job information
echo "==== GPU Job Information ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo "Prediction file: $PRED"
echo "Ground truth file: $GT"
echo "Output file: $OUTPUT"
echo "Config file: $CONFIG"
echo "GPU(s): $CUDA_VISIBLE_DEVICES"
echo "============================="

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT")
mkdir -p "$OUTPUT_DIR"

# Run the scoring script with GPU acceleration
echo "Starting GPU processing at $(date)"

# Set up error handling
if ! python versa/bin/scorer.py \
    --pred "${PRED}" \
    --gt "${GT}" \
    --io soundfile \
    --output_file "${OUTPUT}" \
    --score_config "${CONFIG}" \
    --use_gpu true; then
    
    echo "Error: GPU scoring failed with exit code $?"
    exit 1
fi

echo "GPU processing completed at $(date)"
echo "Results saved to: $OUTPUT"

# Optional: Append completion summary to a tracking file
JOB_TRACK_FILE=$(dirname "$OUTPUT")/../job_status.txt
echo "$(date) - GPU Job $SLURM_JOB_ID complete - $(basename "$PRED")" >> "$JOB_TRACK_FILE"
