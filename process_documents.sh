#!/bin/bash

# Set directories for processing
job_description_dir="Data/JobDescription"
resumes_dir="Data/Resumes"

# Model to be used
model="llama3.2"

# Function to process files in a directory
process_files() {
    local dir=$1
    local model=$2
    for file in "$dir"/*; do
        if [[ -f "$file" ]]; then
            echo "Processing: $file"
            time python document_classifier.py "$file" --model="$model"
        fi
    done
}

# Process Job Descriptions
echo "Processing Job Descriptions:"
process_files "$job_description_dir" "$model"

# Process Resumes
echo "Processing Resumes:"
process_files "$resumes_dir" "$model"

echo "All files processed."
