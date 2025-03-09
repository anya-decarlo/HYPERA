#!/usr/bin/env python
"""
Download Results Script for HYPERA1 Training on Google Cloud
This script helps download training results from Google Cloud Storage.
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path

def check_gcloud_installation():
    """Check if Google Cloud SDK is installed."""
    try:
        subprocess.run(["gcloud", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✅ Google Cloud SDK is installed.")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("❌ Google Cloud SDK is not installed or not in PATH.")
        print("Please install from: https://cloud.google.com/sdk/docs/install")
        return False

def check_authentication():
    """Check if user is authenticated with Google Cloud."""
    try:
        result = subprocess.run(["gcloud", "auth", "list"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if "No credentialed accounts." in result.stdout.decode():
            print("❌ Not authenticated with Google Cloud.")
            return False
        else:
            print("✅ Authenticated with Google Cloud.")
            return True
    except subprocess.SubprocessError:
        print("❌ Error checking authentication status.")
        return False

def authenticate():
    """Authenticate with Google Cloud."""
    try:
        subprocess.run(["gcloud", "auth", "login"], check=True)
        print("✅ Authentication successful.")
        return True
    except subprocess.SubprocessError:
        print("❌ Authentication failed.")
        return False

def list_jobs(project_id):
    """List all AI Platform jobs in the project."""
    try:
        result = subprocess.run(
            ["gcloud", "ai-platform", "jobs", "list", "--project", project_id],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print("Available jobs:")
        print(result.stdout.decode())
        return True
    except subprocess.SubprocessError:
        print("❌ Failed to list jobs.")
        return False

def get_job_directory(project_id, job_name):
    """Get the output directory for a job."""
    try:
        result = subprocess.run(
            ["gcloud", "ai-platform", "jobs", "describe", job_name, "--project", project_id],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output = result.stdout.decode()
        
        # Parse the output to find the job directory
        for line in output.split("\n"):
            if "jobDir:" in line:
                job_dir = line.split("jobDir:")[1].strip()
                return job_dir
        
        print("❌ Could not find job directory in job description.")
        return None
    except subprocess.SubprocessError:
        print(f"❌ Failed to get description for job {job_name}.")
        return None

def download_results(job_dir, output_dir):
    """Download results from Google Cloud Storage."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # First, list all files in the job directory
        result = subprocess.run(
            ["gsutil", "ls", "-r", f"{job_dir}/**"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        files = result.stdout.decode().strip().split("\n")
        
        # Download each file
        for file in files:
            if not file:  # Skip empty lines
                continue
                
            # Create the local directory structure
            rel_path = file.replace(job_dir, "").lstrip("/")
            local_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download the file
            print(f"Downloading {file} to {local_path}...")
            subprocess.run(["gsutil", "cp", file, local_path], check=True)
        
        print(f"✅ Results downloaded to {output_dir}")
        return True
    except subprocess.SubprocessError as e:
        print(f"❌ Failed to download results: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download HYPERA1 training results from Google Cloud.")
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--job-name", help="AI Platform job name")
    parser.add_argument("--output-dir", help="Local directory to download results to")
    parser.add_argument("--list-jobs", action="store_true", help="List all jobs in the project")
    
    args = parser.parse_args()
    
    # Check Google Cloud SDK installation
    if not check_gcloud_installation():
        return 1
    
    # Check authentication
    if not check_authentication():
        if not authenticate():
            return 1
    
    # List jobs if requested
    if args.list_jobs:
        list_jobs(args.project_id)
        return 0
    
    # Ensure job name is provided
    if not args.job_name:
        print("❌ Please provide a job name with --job-name.")
        list_jobs(args.project_id)
        return 1
    
    # Get job directory
    job_dir = get_job_directory(args.project_id, args.job_name)
    if not job_dir:
        return 1
    
    # Set default output directory if not provided
    if not args.output_dir:
        args.output_dir = os.path.join(os.getcwd(), "results", "cloud", args.job_name)
    
    # Download results
    if not download_results(job_dir, args.output_dir):
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
