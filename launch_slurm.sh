#!/bin/bash
#
# Slurm Launcher for VERSA Processing
# -------------------------------------------
# This script splits input audio files and launches Slurm jobs for parallel processing
# using either GPU, CPU, or both resources based on user selection.
#
# Usage: ./launcher.sh <pred_wavscp> <gt_wavscp> <score_dir> <split_size> [--cpu-only|--gpu-only] [--text=FILE]
#   <pred_wavscp>: Path to prediction wav.scp file
#   <gt_wavscp>: Path to ground truth wav.scp file (use "None" if not available)
#   <score_dir>: Directory to store results
#   <split_size>: Number of chunks to split the data into
#   --cpu-only: Optional flag to run only CPU jobs
#   --gpu-only: Optional flag to run only GPU jobs
#   --text=FILE: Path to text file to be processed (optional)
#   --yes: Skip confirmation prompt
#
# Example: ./launcher.sh data/pred.scp data/gt.scp results/experiment1 10
# Example: ./launcher.sh data/pred.scp data/gt.scp results/experiment1 10 --cpu-only
# Example: ./launcher.sh data/pred.scp data/gt.scp results/experiment1 10 --text=data/transcripts.txt

set -e  # Exit immediately if a command exits with non-zero status

# Define color codes for output messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================
# Safeguard defaults — tune these to your cluster's fair-share
# budget to avoid accidental resource overconsumption.
# Override via environment variables if needed.
# ============================================================
MAX_JOBS=${MAX_JOBS:-50}             # Max simultaneous job submissions
MAX_TOTAL_CPU_HOURS=${MAX_TOTAL_CPU_HOURS:-5000}  # Abort if estimated CPU-hours exceed this

# Function to display usage
show_usage() {
    echo -e "${BLUE}Usage: $0 <pred_wavscp> <gt_wavscp> <score_dir> <split_size> [--cpu-only|--gpu-only] [--text=FILE] [--yes]${NC}"
    echo -e "  <pred_wavscp>: Path to prediction wav script file"
    echo -e "  <gt_wavscp>: Path to ground truth wav script file (use \"None\" if not available)"
    echo -e "  <score_dir>: Directory to store results"
    echo -e "  <split_size>: Number of chunks to split the data into"
    echo -e "  --cpu-only: Optional flag to run only CPU jobs"
    echo -e "  --gpu-only: Optional flag to run only GPU jobs"
    echo -e "  --text=FILE: Path to text file to be processed (optional)"
    echo -e "  --yes: Skip the confirmation prompt"
    echo ""
    echo -e "${YELLOW}Safeguard environment variables:${NC}"
    echo -e "  MAX_JOBS=N             Max number of jobs to submit (default: 50)"
    echo -e "  MAX_TOTAL_CPU_HOURS=N  Abort if estimated total CPU-hours exceed this (default: 5000)"
    echo -e "  GPU_TIME=D-HH:MM:SS   GPU job time limit (default: 0-12:00:00)"
    echo -e "  CPU_TIME=D-HH:MM:SS   CPU job time limit (default: 0-12:00:00)"
    echo -e "  CPUS=N                 CPUs per task (default: 4)"
    echo -e "  MEM=N                  Memory per CPU in MB (default: 2000)"
}

# Check for minimum required arguments
if [ $# -lt 4 ]; then
    echo -e "${RED}Error: Insufficient arguments${NC}"
    show_usage
    exit 1
fi

# Parse command line arguments
PRED_WAVSCP=$1
GT_WAVSCP=$2
SCORE_DIR=$3
SPLIT_SIZE=$4
IO_TYPE=${IO_TYPE:-soundfile}
echo ${IO_TYPE}

# Default to running both CPU and GPU jobs
RUN_CPU=true
RUN_GPU=true
TEXT_FILE=""  # Optional text file
SKIP_CONFIRM=false

# Parse optional arguments
shift 4
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-only)
            RUN_GPU=false
            RUN_CPU=true
            echo -e "${YELLOW}Running in CPU-only mode${NC}"
            shift
            ;;
        --gpu-only)
            RUN_GPU=true
            RUN_CPU=false
            echo -e "${YELLOW}Running in GPU-only mode${NC}"
            shift
            ;;
        --text=*)
            TEXT_FILE="${1#*=}"
            if [ ! -f "${TEXT_FILE}" ]; then
                echo -e "${RED}Error: Text file '${TEXT_FILE}' not found${NC}"
                exit 1
            fi
            echo -e "${YELLOW}Text file set to: ${TEXT_FILE}${NC}"
            shift
            ;;
        --yes)
            SKIP_CONFIRM=true
            shift
            ;;
        *)
            echo -e "${RED}Error: Unknown option '$1'${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Validate inputs
if [ ! -f "${PRED_WAVSCP}" ]; then
    echo -e "${RED}Error: Prediction wav script file '${PRED_WAVSCP}' not found${NC}"
    exit 1
fi

if [ "${GT_WAVSCP}" != "None" ] && [ ! -f "${GT_WAVSCP}" ]; then
    echo -e "${RED}Error: Ground truth wav script file '${GT_WAVSCP}' not found${NC}"
    exit 1
fi

if ! [[ "${SPLIT_SIZE}" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: Split size must be a positive integer${NC}"
    exit 1
fi

# Configure Slurm partitions (can be modified based on your cluster setup)
GPU_PART=${GPU_PARTITION:-general}
CPU_PART=${CPU_PARTITION:-general}

# Configure resource requirements — LOWERED DEFAULTS as safeguard
GPU_TIME=${GPU_TIME:-0-12:00:00}     # 12 hours (was 2 days)
CPU_TIME=${CPU_TIME:-0-12:00:00}     # 12 hours (was 2 days)
CPUS_PER_TASK=${CPUS:-4}             # 4 CPUs per task (was 8)
MEM_PER_CPU=${MEM:-2000}             # 2000MB per CPU
GPU_TYPE=${GPU_TYPE:-}               # GPU type
CPU_OTHER_OPTS=${CPU_OTHER_OPTS:-}   # other options for cpu
GPU_OTHER_OPTS=${GPU_OTHER_OPTS:-}   # other options for gpu

# ============================================================
# Safeguard 1: Cap on max simultaneous jobs
# ============================================================
num_jobs_per_chunk=0
$RUN_GPU && ((num_jobs_per_chunk+=1))
$RUN_CPU && ((num_jobs_per_chunk+=1))
total_jobs=$((SPLIT_SIZE * num_jobs_per_chunk))

if [ "${total_jobs}" -gt "${MAX_JOBS}" ]; then
    echo -e "${RED}Error: Would submit ${total_jobs} jobs, which exceeds MAX_JOBS=${MAX_JOBS}.${NC}"
    echo -e "${YELLOW}Reduce --split_size or increase MAX_JOBS to proceed.${NC}"
    echo -e "${YELLOW}  Suggested split_size: $(( MAX_JOBS / num_jobs_per_chunk ))${NC}"
    exit 1
fi

# ============================================================
# Safeguard 2: Estimate total CPU-hours and warn/abort
# ============================================================
# Parse time limit into hours (handles D-HH:MM:SS and HH:MM:SS)
parse_time_to_hours() {
    local time_str=$1
    local days=0 hours=0 mins=0 secs=0

    if [[ "${time_str}" == *-* ]]; then
        days="${time_str%%-*}"
        time_str="${time_str#*-}"
    fi

    IFS=':' read -r hours mins secs <<< "${time_str}"
    # Remove leading zeros to avoid octal interpretation
    days=$((10#${days}))
    hours=$((10#${hours}))
    mins=$((10#${mins}))
    secs=$((10#${secs:-0}))

    echo $(( days * 24 + hours + (mins > 30 ? 1 : 0) ))
}

estimate_cpu_hours() {
    local time_str=$1
    local cpus=$2
    local num_jobs=$3
    local hours
    hours=$(parse_time_to_hours "${time_str}")
    echo $(( hours * cpus * num_jobs ))
}

total_estimated_cpu_hours=0

if $RUN_GPU; then
    gpu_cpu_hours=$(estimate_cpu_hours "${GPU_TIME}" "${CPUS_PER_TASK}" "${SPLIT_SIZE}")
    total_estimated_cpu_hours=$((total_estimated_cpu_hours + gpu_cpu_hours))
fi
if $RUN_CPU; then
    cpu_cpu_hours=$(estimate_cpu_hours "${CPU_TIME}" "${CPUS_PER_TASK}" "${SPLIT_SIZE}")
    total_estimated_cpu_hours=$((total_estimated_cpu_hours + cpu_cpu_hours))
fi

# Print configuration summary
echo -e "${BLUE}=== Configuration Summary ===${NC}"
echo -e "Prediction WAV script: ${PRED_WAVSCP}"
echo -e "Ground truth WAV script: ${GT_WAVSCP}"
echo -e "Output directory: ${SCORE_DIR}"
echo -e "Split size: ${SPLIT_SIZE}"
if [ -n "${TEXT_FILE}" ]; then
    echo -e "Text file: ${TEXT_FILE}"
else
    echo -e "Text file: Not provided"
fi
if $RUN_GPU; then
    echo -e "GPU processing: Enabled"
    echo -e "GPU partition: ${GPU_PART}"
    echo -e "GPU type: ${GPU_TYPE}"
    echo -e "GPU time limit: ${GPU_TIME}"
else
    echo -e "GPU processing: Disabled"
fi
if $RUN_CPU; then
    echo -e "CPU processing: Enabled"
    echo -e "CPU partition: ${CPU_PART}"
    echo -e "CPU time limit: ${CPU_TIME}"
else
    echo -e "CPU processing: Disabled"
fi
echo -e "Resources per job: ${CPUS_PER_TASK} CPUs, ${MEM_PER_CPU}MB per CPU"
echo ""
echo -e "${YELLOW}=== Resource Estimate ===${NC}"
echo -e "Total jobs to submit: ${total_jobs}"
echo -e "Max estimated CPU-hours: ${total_estimated_cpu_hours} (worst case, all jobs run to time limit)"
if $RUN_GPU; then
    echo -e "  GPU jobs: ${SPLIT_SIZE} x ${CPUS_PER_TASK} CPUs x $(parse_time_to_hours "${GPU_TIME}")h = ${gpu_cpu_hours} CPU-hours"
fi
if $RUN_CPU; then
    echo -e "  CPU jobs: ${SPLIT_SIZE} x ${CPUS_PER_TASK} CPUs x $(parse_time_to_hours "${CPU_TIME}")h = ${cpu_cpu_hours} CPU-hours"
fi
echo ""

if [ "${total_estimated_cpu_hours}" -gt "${MAX_TOTAL_CPU_HOURS}" ]; then
    echo -e "${RED}Error: Estimated CPU-hours (${total_estimated_cpu_hours}) exceeds MAX_TOTAL_CPU_HOURS=${MAX_TOTAL_CPU_HOURS}.${NC}"
    echo -e "${YELLOW}Reduce split_size, time limits, or CPUs per task. Or override with MAX_TOTAL_CPU_HOURS=${total_estimated_cpu_hours}${NC}"
    exit 1
fi

# ============================================================
# Safeguard 3: Interactive confirmation before submission
# ============================================================
if [ "${SKIP_CONFIRM}" = false ]; then
    echo -e "${YELLOW}Proceed with submission? [y/N]${NC}"
    read -r response
    if [[ ! "${response}" =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted by user.${NC}"
        exit 0
    fi
fi

# Create directory structure
echo -e "${GREEN}Creating directory structure...${NC}"
mkdir -p "${SCORE_DIR}"
mkdir -p "${SCORE_DIR}/pred"
mkdir -p "${SCORE_DIR}/gt"
mkdir -p "${SCORE_DIR}/text"
mkdir -p "${SCORE_DIR}/result"
mkdir -p "${SCORE_DIR}/logs"

# Split prediction files
total_lines=$(wc -l < "${PRED_WAVSCP}")
source_wavscp=$(basename "${PRED_WAVSCP}")
lines_per_piece=$(( (total_lines + SPLIT_SIZE - 1) / SPLIT_SIZE ))  # Ceiling division

echo -e "${GREEN}Splitting ${total_lines} lines into ${SPLIT_SIZE} pieces (${lines_per_piece} lines per piece)...${NC}"
split -l "${lines_per_piece}" -d -a 3 "${PRED_WAVSCP}" "${SCORE_DIR}/pred/${source_wavscp}_"
pred_list=("${SCORE_DIR}/pred/${source_wavscp}_"*)

# Split ground truth files if provided
if [ "${GT_WAVSCP}" = "None" ]; then
    echo -e "${YELLOW}No ground truth audio provided, evaluation will be reference-free${NC}"
    gt_list=()
else
    target_wavscp=$(basename "${GT_WAVSCP}")
    split -l "${lines_per_piece}" -d -a 3 "${GT_WAVSCP}" "${SCORE_DIR}/gt/${target_wavscp}_"
    gt_list=("${SCORE_DIR}/gt/${target_wavscp}_"*)

    if [ ${#pred_list[@]} -ne ${#gt_list[@]} ]; then
        echo -e "${RED}Error: The number of split ground truth (${#gt_list[@]}) and predictions (${#pred_list[@]}) does not match.${NC}"
        exit 1
    fi
fi

# Split text file if provided
text_list=()
if [ -n "${TEXT_FILE}" ]; then
    echo -e "${GREEN}Splitting text file...${NC}"
    text_basename=$(basename "${TEXT_FILE}")
    split -l "${lines_per_piece}" -d -a 3 "${TEXT_FILE}" "${SCORE_DIR}/text/${text_basename}_"
    text_list=("${SCORE_DIR}/text/${text_basename}_"*)

    if [ ${#pred_list[@]} -ne ${#text_list[@]} ]; then
        echo -e "${RED}Error: The number of split text files (${#text_list[@]}) and predictions (${#pred_list[@]}) does not match.${NC}"
        exit 1
    fi
fi

# Generate job IDs file for tracking
JOB_IDS_FILE="${SCORE_DIR}/job_ids.txt"
> "${JOB_IDS_FILE}"  # Clear file if it exists

echo -e "${GREEN}Submitting jobs...${NC}"

for ((i=0; i<${#pred_list[@]}; i++)); do
    sub_pred_wavscp=${pred_list[${i}]}
    job_prefix=$(basename "${sub_pred_wavscp}")

    if [ "${GT_WAVSCP}" = "None" ]; then
        sub_gt_wavscp="None"
    else
        sub_gt_wavscp=${gt_list[${i}]}
    fi
    
    # Set text file for this chunk
    if [ -n "${TEXT_FILE}" ]; then
        sub_text_file=${text_list[${i}]}
    else
        sub_text_file=""
    fi
    
    echo -e "${BLUE}Processing chunk $((i+1))/${#pred_list[@]}: ${sub_pred_wavscp}${NC}"
    if [ -n "${sub_text_file}" ]; then
        echo -e "${BLUE}  Text file: ${sub_text_file}${NC}"
    fi

    # Submit GPU job if enabled
    if $RUN_GPU; then

        # Set up GPU resource specification based on whether GPU_TYPE is empty
        if [ -n "${GPU_TYPE}" ]; then
            GPU_GRES="--gres=gpu:${GPU_TYPE}:1"
        else
            GPU_GRES="--gres=gpu:1"
        fi

        gpu_job_id=$(sbatch \
            --parsable \
            -p "${GPU_PART}" \
            --time "${GPU_TIME}" \
            --cpus-per-task "${CPUS_PER_TASK}" \
            --mem-per-cpu "${MEM_PER_CPU}M" \
            ${GPU_GRES} ${GPU_OTHER_OPTS} \
            -J "gpu_${job_prefix}" \
            -o "${SCORE_DIR}/logs/gpu_${job_prefix}_%j.out" \
            -e "${SCORE_DIR}/logs/gpu_${job_prefix}_%j.err" \
            ./egs/run_gpu.sh \
                "${sub_pred_wavscp}" \
                "${sub_gt_wavscp}" \
                "${SCORE_DIR}/result/$(basename "${sub_pred_wavscp}").result.gpu.txt" \
                egs/universa_prepare/gpu_subset.yaml \
                ${IO_TYPE} \
                "${sub_text_file}")

        echo "GPU:${gpu_job_id} CHUNK:$((i+1))/${#pred_list[@]} FILE:${job_prefix}" >> "${JOB_IDS_FILE}"
        echo -e "  Submitted GPU job: ${gpu_job_id}"
    fi

    # Submit CPU job if enabled
    if $RUN_CPU; then
        cpu_job_id=$(sbatch \
            --parsable \
            -p "${CPU_PART}" \
            --time "${CPU_TIME}" \
            --cpus-per-task "${CPUS_PER_TASK}" \
            --mem-per-cpu "${MEM_PER_CPU}M" \
            ${CPU_OTHER_OPTS} \
            -J "cpu_${job_prefix}" \
            -o "${SCORE_DIR}/logs/cpu_${job_prefix}_%j.out" \
            -e "${SCORE_DIR}/logs/cpu_${job_prefix}_%j.err" \
            ./egs/run_cpu.sh \
                "${sub_pred_wavscp}" \
                "${sub_gt_wavscp}" \
                "${SCORE_DIR}/result/$(basename "${sub_pred_wavscp}").result.cpu.txt" \
                egs/universa_prepare/cpu_subset.yaml \
                ${IO_TYPE} \
                "${sub_text_file}")

        echo "CPU:${cpu_job_id} CHUNK:$((i+1))/${#pred_list[@]} FILE:${job_prefix}" >> "${JOB_IDS_FILE}"
        echo -e "  Submitted CPU job: ${cpu_job_id}"
    fi
done

# Create instructions for merging results
echo -e "${GREEN}All jobs submitted. Job IDs saved to: ${JOB_IDS_FILE}${NC}"

# Get job IDs for dependency specification
if $RUN_GPU && $RUN_CPU; then
    all_gpu_jobs=$(grep "GPU:" "${JOB_IDS_FILE}" | cut -d':' -f2 | cut -d' ' -f1 | paste -sd, -)
    all_cpu_jobs=$(grep "CPU:" "${JOB_IDS_FILE}" | cut -d':' -f2 | cut -d' ' -f1 | paste -sd, -)
    all_jobs="${all_gpu_jobs},${all_cpu_jobs}"
    job_type="GPU and CPU"
elif $RUN_GPU; then
    all_jobs=$(grep "GPU:" "${JOB_IDS_FILE}" | cut -d':' -f2 | cut -d' ' -f1 | paste -sd, -)
    job_type="GPU"
else
    all_jobs=$(grep "CPU:" "${JOB_IDS_FILE}" | cut -d':' -f2 | cut -d' ' -f1 | paste -sd, -)
    job_type="CPU"
fi

echo -e "${YELLOW}To create a dependent job that processes all ${job_type} results after completion, use:${NC}"
echo -e "sbatch --dependency=afterok:${all_jobs} ./scripts/show_result.sh ${SCORE_DIR}/result ${SCORE_DIR}/final_results.txt"

echo -e "${GREEN}Done! Monitor jobs with 'squeue -u $(whoami)'${NC}"
