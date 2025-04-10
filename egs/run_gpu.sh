#!/bin/bash
#
# GPU Score Runner for VERSA Processing
# ------------------------------------
# This script processes audio files using GPU acceleration
#
# Usage: ./run_gpu.sh <pred_wavscp> <gt_wavscp> <output_file> <config_file> [io_type]

set -e  # Exit immediately if a command exits with non-zero status

# Check if minimum required arguments are provided
if [ $# -lt 4 ]; then
    echo "Error: Insufficient number of arguments"
    echo "Usage: $0 <pred_wavscp> <gt_wavscp> <output_file> <config_file> [io_type]"
    echo "io_type: Optional parameter - soundfile (default), dir, or kaldi"
    exit 1
fi

# Parse command line arguments
PRED=$1
GT=$2
OUTPUT=$3
CONFIG=$4

# Set default IO type if not provided
IO_TYPE=${5:-soundfile}

# Validate IO type
if [[ "$IO_TYPE" != "soundfile" && "$IO_TYPE" != "dir" && "$IO_TYPE" != "kaldi" ]]; then
    echo "Error: Invalid IO type '$IO_TYPE'"
    echo "Valid options are: soundfile, dir, kaldi"
    exit 1
fi

# Display job information
echo "==== GPU Job Information ===="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo "Prediction file: $PRED"
echo "Ground truth file: $GT"
echo "Output file: $OUTPUT"
echo "Config file: $CONFIG"
echo "IO type: $IO_TYPE"
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
    --io "${IO_TYPE}" \
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
echo "$(date) - GPU Job $SLURM_JOB_ID complete - $(basename "$PRED") - IO: $IO_TYPE" >> "$JOB_TRACK_FILE"
