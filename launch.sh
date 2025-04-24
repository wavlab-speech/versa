#!/bin/bash
#
# Slurm Launcher for VERSA Processing
# -------------------------------------------
# This script splits input audio files and launches Slurm jobs for parallel processing
# using either GPU, CPU, or both resources based on user selection.
#
# Usage: ./launcher.sh <pred_wavscp> <gt_wavscp> <score_dir> <split_size> [--cpu-only|--gpu-only]
#   <pred_wavscp>: Path to prediction wav.scp file
#   <gt_wavscp>: Path to ground truth wav.scp file (use "None" if not available)
#   <score_dir>: Directory to store results
#   <split_size>: Number of chunks to split the data into
#   --cpu-only: Optional flag to run only CPU jobs
#   --gpu-only: Optional flag to run only GPU jobs
#
# Example: ./launcher.sh data/pred.scp data/gt.scp results/experiment1 10
# Example: ./launcher.sh data/pred.scp data/gt.scp results/experiment1 10 --cpu-only

set -e  # Exit immediately if a command exits with non-zero status

# Define color codes for output messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check for minimum required arguments
if [ $# -lt 4 ]; then
    echo -e "${RED}Error: Insufficient arguments${NC}"
    echo -e "${BLUE}Usage: $0 <pred_wavscp> <gt_wavscp> <score_dir> <split_size> [--cpu-only|--gpu-only]${NC}"
    echo -e "  <pred_wavscp>: Path to prediction wav script file"
    echo -e "  <gt_wavscp>: Path to ground truth wav script file (use \"None\" if not available)"
    echo -e "  <score_dir>: Directory to store results"
    echo -e "  <split_size>: Number of chunks to split the data into"
    echo -e "  --cpu-only: Optional flag to run only CPU jobs"
    echo -e "  --gpu-only: Optional flag to run only GPU jobs"
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

# Check for optional flags
if [ $# -ge 5 ]; then
    if [ "$5" = "--cpu-only" ]; then
        RUN_GPU=false
        RUN_CPU=true
        echo -e "${YELLOW}Running in CPU-only mode${NC}"
    elif [ "$5" = "--gpu-only" ]; then
        RUN_GPU=true
        RUN_CPU=false
        echo -e "${YELLOW}Running in GPU-only mode${NC}"
    else
        echo -e "${RED}Error: Unknown option '$5'${NC}"
        echo -e "${BLUE}Valid options are: --cpu-only, --gpu-only${NC}"
        exit 1
    fi
fi

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

# Configure resource requirements
GPU_TIME=${GPU_TIME:-2-0:00:00}      # 2 days
CPU_TIME=${CPU_TIME:-2-0:00:00}      # 2 days
CPUS_PER_TASK=${CPUS:-8}             # 8 CPUs per task
MEM_PER_CPU=${MEM:-2000}             # 2000MB per CPU
GPU_TYPE=${GPU_TYPE:-}               # GPU type
CPU_OTHER_OPTS=${CPU_OTHER_OPTS:-}   # other options for cpu
GPU_OTHER_OPTS=${GPU_OTHER_OPTS:-}   # other options for gpu

# Print configuration summary
echo -e "${BLUE}=== Configuration Summary ===${NC}"
echo -e "Prediction WAV script: ${PRED_WAVSCP}"
echo -e "Ground truth WAV script: ${GT_WAVSCP}"
echo -e "Output directory: ${SCORE_DIR}"
echo -e "Split size: ${SPLIT_SIZE}"
if $RUN_GPU; then
    echo -e "GPU processing: Enabled"
    echo -e "GPU partition: ${GPU_PART}"
    echo -e "GPU type: ${GPU_TYPE}"
else
    echo -e "GPU processing: Disabled"
fi
if $RUN_CPU; then
    echo -e "CPU processing: Enabled"
    echo -e "CPU partition: ${CPU_PART}"
else
    echo -e "CPU processing: Disabled"
fi
echo -e "Resources per job: ${CPUS_PER_TASK} CPUs, ${MEM_PER_CPU}MB per CPU"
echo ""

# Create directory structure
echo -e "${GREEN}Creating directory structure...${NC}"
mkdir -p "${SCORE_DIR}"
mkdir -p "${SCORE_DIR}/pred"
mkdir -p "${SCORE_DIR}/gt"
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
    
    echo -e "${BLUE}Processing chunk $((i+1))/${#pred_list[@]}: ${sub_pred_wavscp}${NC}"

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
                ${IO_TYPE})

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
                ${IO_TYPE})

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
echo -e "sbatch --dependency=afterok:${all_jobs} ./merge_results.sh ${SCORE_DIR}/result ${SCORE_DIR}/final_results.txt"

echo -e "${GREEN}Done! Monitor jobs with 'squeue -u $(whoami)'${NC}"
