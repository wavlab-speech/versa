#!/bin/bash
#
# CPU Score Runner for Audio Processing
# ------------------------------------
# This script processes audio files using CPU only
#
# Usage: ./run_cpu.sh <pred_wavscp> <gt_wavscp> <output_file> <config_file> [io_type] [text_file]
set -e  # Exit immediately if a command exits with non-zero status

# Check if minimum required arguments are provided
if [ $# -lt 4 ]; then
    echo "Error: Insufficient number of arguments"
    echo "Usage: $0 <pred_wavscp> <gt_wavscp> <output_file> <config_file> [io_type] [text_file]"
    echo "  pred_wavscp: Path to prediction wav.scp file"
    echo "  gt_wavscp: Path to ground truth wav.scp file (use 'None' if not available)"
    echo "  output_file: Path to output results file"
    echo "  config_file: Path to configuration file"
    echo "  io_type: Optional parameter - soundfile (default), dir, or kaldi"
    echo "  text_file: Optional path to text file for processing"
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

# Display job information
echo "==== CPU Job Information ===="
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
echo "Number of cores: ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "============================="

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT")
mkdir -p "$OUTPUT_DIR"

# Run the scoring script on CPU
echo "Starting CPU processing at $(date)"

# Build the command with optional text parameter
CMD_ARGS=(
    "--pred" "${PRED}"
    "--gt" "${GT}"
    "--io" "${IO_TYPE}"
    "--output_file" "${OUTPUT}"
    "--score_config" "${CONFIG}"
)

# Add text file argument if provided
if [ -n "$TEXT_FILE" ] && [ "$TEXT_FILE" != "" ]; then
    CMD_ARGS+=("--text" "${TEXT_FILE}")
    echo "Including text file in processing: $TEXT_FILE"
fi

# Set up error handling and run the scoring script
if ! python versa/bin/scorer.py "${CMD_ARGS[@]}"; then
    echo "Error: CPU scoring failed with exit code $?"
    exit 1
fi

echo "CPU processing completed at $(date)"
echo "Results saved to: $OUTPUT"

# Optional: Append completion summary to a tracking file
JOB_TRACK_FILE=$(dirname "$OUTPUT")/../job_status.txt
if [ -n "$TEXT_FILE" ] && [ "$TEXT_FILE" != "" ]; then
    echo "$(date) - CPU Job ${SLURM_JOB_ID:-LOCAL} complete - $(basename "$PRED") - IO: $IO_TYPE - Text: $(basename "$TEXT_FILE")" >> "$JOB_TRACK_FILE"
else
    echo "$(date) - CPU Job ${SLURM_JOB_ID:-LOCAL} complete - $(basename "$PRED") - IO: $IO_TYPE" >> "$JOB_TRACK_FILE"
fi
