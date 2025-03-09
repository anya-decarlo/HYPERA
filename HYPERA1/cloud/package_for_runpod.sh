#!/bin/bash

# Package HYPERA for RunPod
# This script creates a zip file of the HYPERA project for easy upload to RunPod

# Set the output zip file name
OUTPUT_ZIP="HYPERA_for_RunPod.zip"

# Create a temporary directory
TEMP_DIR="tmp_runpod_package"
mkdir -p $TEMP_DIR

# Copy the necessary files and directories
echo "Copying HYPERA files..."
cp -r HYPERA1 $TEMP_DIR/
cp -r results_hyperparameter_agent $TEMP_DIR/ 2>/dev/null || echo "No results directory found, skipping..."
cp HYPERA1/cloud/HYPERA_Training.ipynb $TEMP_DIR/
cp HYPERA1/cloud/RUNPOD_README.md $TEMP_DIR/README.md

# Remove any unnecessary large files (like cached data, __pycache__, etc.)
echo "Cleaning up unnecessary files..."
find $TEMP_DIR -name "__pycache__" -type d -exec rm -rf {} +
find $TEMP_DIR -name "*.pyc" -delete
find $TEMP_DIR -name ".DS_Store" -delete
find $TEMP_DIR -name ".ipynb_checkpoints" -type d -exec rm -rf {} +

# Create the zip file
echo "Creating zip file: $OUTPUT_ZIP"
cd $TEMP_DIR
zip -r ../$OUTPUT_ZIP .
cd ..

# Clean up
echo "Cleaning up temporary directory..."
rm -rf $TEMP_DIR

echo "Done! Your HYPERA package is ready at: $OUTPUT_ZIP"
echo "Upload this file to RunPod and extract it there."
