#!/bin/bash
#
# CPU Score Runner for Audio Processing
# ------------------------------------
# This script processes audio files using CPU only
#
# Usage: ./run_cpu.sh <pred_wavscp> <gt_wavscp> <output_file> <config_file>

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
echo "==== CPU Job Information ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo "Prediction file: $PRED"
echo "Ground truth file: $GT"
echo "Output file: $OUTPUT"
echo "Config file: $CONFIG"
echo "Number of cores: $SLURM_CPUS_PER_TASK"
echo "============================="

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT")
mkdir -p "$OUTPUT_DIR"

# Run the scoring script on CPU
echo "Starting CPU processing at $(date)"

# Set up error handling
if ! python versa/bin/scorer.py \
    --pred "${PRED}" \
    --gt "${GT}" \
    --io soundfile \
    --output_file "${OUTPUT}" \
    --score_config "${CONFIG}"; then
    
    echo "Error: CPU scoring failed with exit code $?"
    exit 1
fi

echo "CPU processing completed at $(date)"
echo "Results saved to: $OUTPUT"

# Optional: Append completion summary to a tracking file
JOB_TRACK_FILE=$(dirname "$OUTPUT")/../job_status.txt
echo "$(date) - CPU Job $SLURM_JOB_ID complete - $(basename "$PRED")" >> "$JOB_TRACK_FILE"
