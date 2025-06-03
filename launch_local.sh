#!/bin/bash
#
# Local Launcher for VERSA Processing
# -------------------------------------------
# This script splits input audio files and launches parallel processes locally
# using either GPU, CPU, or both resources based on user selection.
#
# Usage: ./local_launcher.sh <pred_wavscp> <gt_wavscp> <score_dir> <split_size> [--cpu-only|--gpu-only] [--max-parallel=N] [--text=FILE]
#   <pred_wavscp>: Path to prediction wav.scp file
#   <gt_wavscp>: Path to ground truth wav.scp file (use "None" if not available)
#   <score_dir>: Directory to store results
#   <split_size>: Number of chunks to split the data into
#   --cpu-only: Optional flag to run only CPU jobs
#   --gpu-only: Optional flag to run only GPU jobs
#   --max-parallel=N: Maximum number of parallel processes (default: number of CPU cores)
#   --text=FILE: Path to text file to be processed (optional)
#
# Example: ./local_launcher.sh data/pred.scp data/gt.scp results/experiment1 10
# Example: ./local_launcher.sh data/pred.scp data/gt.scp results/experiment1 10 --cpu-only --max-parallel=4
# Example: ./local_launcher.sh data/pred.scp data/gt.scp results/experiment1 10 --text=data/transcripts.txt

set -e  # Exit immediately if a command exits with non-zero status

# Define color codes for output messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display usage
show_usage() {
    echo -e "${BLUE}Usage: $0 <pred_wavscp> <gt_wavscp> <score_dir> <split_size> [--cpu-only|--gpu-only] [--max-parallel=N] [--text=FILE]${NC}"
    echo -e "  <pred_wavscp>: Path to prediction wav script file"
    echo -e "  <gt_wavscp>: Path to ground truth wav script file (use \"None\" if not available)"
    echo -e "  <score_dir>: Directory to store results"
    echo -e "  <split_size>: Number of chunks to split the data into"
    echo -e "  --cpu-only: Optional flag to run only CPU jobs"
    echo -e "  --gpu-only: Optional flag to run only GPU jobs"
    echo -e "  --max-parallel=N: Maximum number of parallel processes (default: auto-detect CPU cores)"
    echo -e "  --text=FILE: Path to text file to be processed (optional)"
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

# Default settings
RUN_CPU=true
RUN_GPU=true
MAX_PARALLEL=$(nproc)  # Default to number of CPU cores
TEXT_FILE=""  # Optional text file

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
        --max-parallel=*)
            MAX_PARALLEL="${1#*=}"
            if ! [[ "${MAX_PARALLEL}" =~ ^[0-9]+$ ]] || [ "${MAX_PARALLEL}" -eq 0 ]; then
                echo -e "${RED}Error: max-parallel must be a positive integer${NC}"
                exit 1
            fi
            echo -e "${YELLOW}Maximum parallel processes set to: ${MAX_PARALLEL}${NC}"
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

# Check for GPU availability if GPU processing is enabled
if $RUN_GPU; then
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}Warning: nvidia-smi not found. GPU processing may not work properly.${NC}"
        echo -e "${YELLOW}Continuing anyway... (use --cpu-only to disable GPU processing)${NC}"
    else
        gpu_count=$(nvidia-smi -L | wc -l)
        echo -e "${GREEN}Found ${gpu_count} GPU(s)${NC}"
    fi
fi

# Print configuration summary
echo -e "${BLUE}=== Configuration Summary ===${NC}"
echo -e "Prediction WAV script: ${PRED_WAVSCP}"
echo -e "Ground truth WAV script: ${GT_WAVSCP}"
echo -e "Output directory: ${SCORE_DIR}"
echo -e "Split size: ${SPLIT_SIZE}"
echo -e "Maximum parallel processes: ${MAX_PARALLEL}"
echo -e "Available CPU cores: $(nproc)"
if [ -n "${TEXT_FILE}" ]; then
    echo -e "Text file: ${TEXT_FILE}"
else
    echo -e "Text file: Not provided"
fi
if $RUN_GPU; then
    echo -e "GPU processing: Enabled"
else
    echo -e "GPU processing: Disabled"
fi
if $RUN_CPU; then
    echo -e "CPU processing: Enabled"
else
    echo -e "CPU processing: Disabled"
fi
echo ""

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

# Function to run a single processing job
run_job() {
    local job_type=$1
    local sub_pred_wavscp=$2
    local sub_gt_wavscp=$3
    local sub_text_file=$4
    local output_file=$5
    local config_file=$6
    local job_prefix=$7
    local chunk_info=$8
    
    local log_file="${SCORE_DIR}/logs/${job_type}_${job_prefix}_$(date +%s).log"
    
    echo -e "${BLUE}Starting ${job_type} job for ${chunk_info}: ${job_prefix}${NC}"
    
    if [ "${job_type}" = "gpu" ]; then
        ./egs/run_gpu.sh \
            "${sub_pred_wavscp}" \
            "${sub_gt_wavscp}" \
            "${output_file}" \
            "${config_file}" \
            "${IO_TYPE}" \
            "${sub_text_file}" > "${log_file}" 2>&1
    else
        ./egs/run_cpu.sh \
            "${sub_pred_wavscp}" \
            "${sub_gt_wavscp}" \
            "${output_file}" \
            "${config_file}" \
            "${IO_TYPE}" \
            "${sub_text_file}" > "${log_file}" 2>&1
    fi
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}Completed ${job_type} job for ${chunk_info}: ${job_prefix}${NC}"
    else
        echo -e "${RED}Failed ${job_type} job for ${chunk_info}: ${job_prefix} (exit code: ${exit_code})${NC}"
        echo -e "${RED}Check log file: ${log_file}${NC}"
    fi
    
    return $exit_code
}

# Job tracking
declare -a job_pids=()
declare -a job_info=()
RUNNING_JOBS_FILE="${SCORE_DIR}/running_jobs.txt"
COMPLETED_JOBS_FILE="${SCORE_DIR}/completed_jobs.txt"
FAILED_JOBS_FILE="${SCORE_DIR}/failed_jobs.txt"

# Clear tracking files
> "${RUNNING_JOBS_FILE}"
> "${COMPLETED_JOBS_FILE}"
> "${FAILED_JOBS_FILE}"

echo -e "${GREEN}Starting parallel processing...${NC}"

# Function to wait for available slot
wait_for_slot() {
    while [ ${#job_pids[@]} -ge $MAX_PARALLEL ]; do
        # Check for completed jobs
        for i in "${!job_pids[@]}"; do
            if ! kill -0 "${job_pids[$i]}" 2>/dev/null; then
                # Job has finished
                wait "${job_pids[$i]}"
                local exit_code=$?
                
                if [ $exit_code -eq 0 ]; then
                    echo "${job_info[$i]}" >> "${COMPLETED_JOBS_FILE}"
                else
                    echo "${job_info[$i]}" >> "${FAILED_JOBS_FILE}"
                fi
                
                # Remove from tracking arrays
                unset job_pids[$i]
                unset job_info[$i]
                
                # Rebuild arrays to remove gaps
                job_pids=("${job_pids[@]}")
                job_info=("${job_info[@]}")
                break
            fi
        done
        sleep 1
    done
}

# Submit jobs
for ((i=0; i<${#pred_list[@]}; i++)); do
    sub_pred_wavscp=${pred_list[${i}]}
    job_prefix=$(basename "${sub_pred_wavscp}")
    chunk_info="$((i+1))/${#pred_list[@]}"

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
    
    echo -e "${BLUE}Processing chunk ${chunk_info}: ${sub_pred_wavscp}${NC}"
    if [ -n "${sub_text_file}" ]; then
        echo -e "${BLUE}  Text file: ${sub_text_file}${NC}"
    fi

    # Submit GPU job if enabled
    if $RUN_GPU; then
        wait_for_slot
        
        run_job "gpu" \
            "${sub_pred_wavscp}" \
            "${sub_gt_wavscp}" \
            "${sub_text_file}" \
            "${SCORE_DIR}/result/$(basename "${sub_pred_wavscp}").result.gpu.txt" \
            "egs/quality_check.yaml" \
            "${job_prefix}" \
            "${chunk_info}" &
        
        gpu_pid=$!
        job_pids+=($gpu_pid)
        job_info+=("GPU:$gpu_pid CHUNK:${chunk_info} FILE:${job_prefix}")
        echo "GPU:$gpu_pid CHUNK:${chunk_info} FILE:${job_prefix}" >> "${RUNNING_JOBS_FILE}"
        echo -e "  Started GPU job: PID ${gpu_pid}"
    fi

    # Submit CPU job if enabled
    if $RUN_CPU; then
        wait_for_slot
        
        run_job "cpu" \
            "${sub_pred_wavscp}" \
            "${sub_gt_wavscp}" \
            "${sub_text_file}" \
            "${SCORE_DIR}/result/$(basename "${sub_pred_wavscp}").result.cpu.txt" \
            "egs/universa_prepare/cpu_subset.yaml" \
            "${job_prefix}" \
            "${chunk_info}" &
        
        cpu_pid=$!
        job_pids+=($cpu_pid)
        job_info+=("CPU:$cpu_pid CHUNK:${chunk_info} FILE:${job_prefix}")
        echo "CPU:$cpu_pid CHUNK:${chunk_info} FILE:${job_prefix}" >> "${RUNNING_JOBS_FILE}"
        echo -e "  Started CPU job: PID ${cpu_pid}"
    fi
done

# Wait for all remaining jobs to complete
echo -e "${YELLOW}Waiting for all jobs to complete...${NC}"
for pid in "${job_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        wait "$pid"
        exit_code=$?
        
        # Find job info for this PID
        for info in "${job_info[@]}"; do
            if [[ "$info" == *":$pid "* ]]; then
                if [ $exit_code -eq 0 ]; then
                    echo "$info" >> "${COMPLETED_JOBS_FILE}"
                else
                    echo "$info" >> "${FAILED_JOBS_FILE}"
                fi
                break
            fi
        done
    fi
done

# Generate summary
echo -e "${GREEN}=== Processing Summary ===${NC}"
if [ -f "${COMPLETED_JOBS_FILE}" ]; then
    completed_count=$(wc -l < "${COMPLETED_JOBS_FILE}" 2>/dev/null || echo "0")
    echo -e "${GREEN}Completed jobs: ${completed_count}${NC}"
fi

if [ -f "${FAILED_JOBS_FILE}" ] && [ -s "${FAILED_JOBS_FILE}" ]; then
    failed_count=$(wc -l < "${FAILED_JOBS_FILE}")
    echo -e "${RED}Failed jobs: ${failed_count}${NC}"
    echo -e "${RED}Failed job details:${NC}"
    cat "${FAILED_JOBS_FILE}"
fi

# Create merge script
MERGE_SCRIPT="${SCORE_DIR}/merge_results.sh"
cat > "${MERGE_SCRIPT}" << 'EOF'
#!/bin/bash
# Merge results script
RESULT_DIR=$1
OUTPUT_FILE=$2

if [ $# -ne 2 ]; then
    echo "Usage: $0 <result_dir> <output_file>"
    exit 1
fi

echo "Merging results from ${RESULT_DIR} to ${OUTPUT_FILE}"

# Merge GPU results if they exist
if ls "${RESULT_DIR}"/*.result.gpu.txt 1> /dev/null 2>&1; then
    echo "Merging GPU results..."
    cat "${RESULT_DIR}"/*.result.gpu.txt > "${OUTPUT_FILE%.txt}.gpu.txt"
fi

# Merge CPU results if they exist
if ls "${RESULT_DIR}"/*.result.cpu.txt 1> /dev/null 2>&1; then
    echo "Merging CPU results..."
    cat "${RESULT_DIR}"/*.result.cpu.txt > "${OUTPUT_FILE%.txt}.cpu.txt"
fi

echo "Results merged successfully!"
EOF

chmod +x "${MERGE_SCRIPT}"

echo -e "${YELLOW}To merge all results, run:${NC}"
echo -e "${MERGE_SCRIPT} ${SCORE_DIR}/result ${SCORE_DIR}/final_results.txt"

echo -e "${GREEN}All processing completed!${NC}"
echo -e "Logs are available in: ${SCORE_DIR}/logs/"
echo -e "Results are available in: ${SCORE_DIR}/result/"
