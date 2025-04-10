#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Batch Processing Script for Qwen2-Audio JSON Files

This script standardizes multiple JSON files containing Qwen2-Audio outputs.
It supports processing individual files, directories, or reading from a CSV file
with a list of files to process.
"""

import os
import argparse
import json
import csv
import logging
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# Import the standardizer
from qwen_output_standardizer import JsonOutputStandardizer

def setup_logger(log_file=None, log_level="INFO"):
    """Set up logging"""
    numeric_level = getattr(logging, log_level.upper(), None)
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

def process_file(file_path, output_dir, standardizer=None):
    """
    Process a single JSON file.
    
    Args:
        file_path (str): Path to JSON file
        output_dir (str): Directory to save output
        standardizer (JsonOutputStandardizer): Standardizer instance
        
    Returns:
        dict: Processing result
    """
    try:
        # Create standardizer if not provided
        if standardizer is None:
            standardizer = JsonOutputStandardizer()
        
        # Load JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Standardize the output
        standardized = standardizer.standardize_json(data)
        
        # Create output path
        file_name = os.path.basename(file_path)
        stem = os.path.splitext(file_name)[0]
        output_path = os.path.join(output_dir, f"{stem}_standardized.json")
        
        # Save standardized output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(standardized, f, indent=2, ensure_ascii=False)
        
        return {
            "input_file": file_path,
            "output_file": output_path,
            "success": True
        }
        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return {
            "input_file": file_path,
            "success": False,
            "error": str(e)
        }

def process_batch(file_paths, output_dir, max_workers=4):
    """
    Process a batch of JSON files.
    
    Args:
        file_paths (list): List of JSON file paths
        output_dir (str): Directory to save outputs
        max_workers (int): Maximum number of parallel workers
        
    Returns:
        dict: Summary of processing results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a shared standardizer
    standardizer = JsonOutputStandardizer()
    
    # Process files
    results = {}
    
    # Use concurrent processing if multiple files
    if len(file_paths) > 1 and max_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_file, file_path, output_dir): file_path
                for file_path in file_paths
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_file), 
                              total=len(file_paths),
                              desc="Processing files"):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results[file_path] = result
                except Exception as e:
                    results[file_path] = {
                        "input_file": file_path,
                        "success": False,
                        "error": str(e)
                    }
    else:
        # Process sequentially
        for file_path in tqdm(file_paths, desc="Processing files"):
            results[file_path] = process_file(file_path, output_dir, standardizer)
    
    # Create summary
    success_count = sum(1 for r in results.values() if r.get("success", False))
    failed_count = len(results) - success_count
    
    summary = {
        "total_files": len(file_paths),
        "processed_successfully": success_count,
        "failed": failed_count,
        "output_directory": output_dir
    }
    
    # Save summary
    summary_path = os.path.join(output_dir, "processing_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": summary,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    return summary

def process_directory(input_dir, output_dir, file_pattern="*.json", max_workers=4):
    """
    Process all JSON files in a directory.
    
    Args:
        input_dir (str): Input directory
        output_dir (str): Output directory
        file_pattern (str): File pattern to match
        max_workers (int): Maximum number of parallel workers
        
    Returns:
        dict: Summary of processing results
    """
    # Get list of JSON files
    input_path = Path(input_dir)
    json_files = list(input_path.glob(file_pattern))
    
    if not json_files:
        logging.warning(f"No files found in {input_dir} matching pattern '{file_pattern}'")
        return {"error": f"No files found matching pattern '{file_pattern}'"}
    
    # Process the batch
    return process_batch([str(f) for f in json_files], output_dir, max_workers)

def process_csv_list(csv_file, output_dir, file_column="file_path", max_workers=4):
    """
    Process JSON files listed in a CSV file.
    
    Args:
        csv_file (str): Path to CSV file
        output_dir (str): Output directory
        file_column (str): Column name containing file paths
        max_workers (int): Maximum number of parallel workers
        
    Returns:
        dict: Summary of processing results
    """
    # Read file paths from CSV
    file_paths = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Check if the specified column exists
        if file_column not in reader.fieldnames:
            error_msg = f"Column '{file_column}' not found in CSV. Available columns: {', '.join(reader.fieldnames)}"
            logging.error(error_msg)
            return {"error": error_msg}
        
        for row in reader:
            file_path = row[file_column].strip()
            if file_path and os.path.isfile(file_path):
                file_paths.append(file_path)
            else:
                logging.warning(f"File not found: {file_path}")
    
    if not file_paths:
        logging.warning(f"No valid file paths found in CSV column '{file_column}'")
        return {"error": "No valid file paths found in CSV"}
    
    # Process the batch
    return process_batch(file_paths, output_dir, max_workers)

def main():
    parser = argparse.ArgumentParser(description="Batch process Qwen2-Audio JSON files")
    
    # Input source group
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", "-f", help="Process a single JSON file")
    input_group.add_argument("--directory", "-d", help="Process all JSON files in a directory")
    input_group.add_argument("--csv", "-c", help="Process files listed in a CSV file")
    
    # Output
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    
    # Options
    parser.add_argument("--pattern", default="*.json", help="File pattern when using directory mode (default: *.json)")
    parser.add_argument("--csv-column", default="file_path", help="Column name with file paths in CSV mode (default: file_path)")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--log-file", help="Save logs to file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                       help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger(args.log_file, args.log_level)
    
    # Process based on input type
    if args.file:
        if not os.path.isfile(args.file):
            logger.error(f"File not found: {args.file}")
            return
        
        logger.info(f"Processing single file: {args.file}")
        os.makedirs(args.output, exist_ok=True)
        result = process_file(args.file, args.output)
        
        if result["success"]:
            logger.info(f"Successfully processed {args.file} -> {result['output_file']}")
        else:
            logger.error(f"Failed to process {args.file}: {result.get('error', 'Unknown error')}")
    
    elif args.directory:
        if not os.path.isdir(args.directory):
            logger.error(f"Directory not found: {args.directory}")
            return
        
        logger.info(f"Processing directory: {args.directory}")
        summary = process_directory(args.directory, args.output, args.pattern, args.workers)
        
        logger.info(f"Processing complete. Summary:")
        for key, value in summary.items():
            if key != "results":
                logger.info(f"  {key}: {value}")
    
    elif args.csv:
        if not os.path.isfile(args.csv):
            logger.error(f"CSV file not found: {args.csv}")
            return
        
        logger.info(f"Processing files from CSV: {args.csv}")
        summary = process_csv_list(args.csv, args.output, args.csv_column, args.workers)
        
        logger.info(f"Processing complete. Summary:")
        for key, value in summary.items():
            if key != "results":
                logger.info(f"  {key}: {value}")
    
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
