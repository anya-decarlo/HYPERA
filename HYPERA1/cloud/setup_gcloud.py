#!/usr/bin/env python
"""
Google Cloud Setup Script for HYPERA1 Training
This script helps set up and deploy training jobs to Google Cloud AI Platform (Vertex AI).
"""

import os
import sys
import argparse
import json
import subprocess
import time
import tarfile
from pathlib import Path

GCLOUD_PATH = "gcloud"  # Default to just 'gcloud', will be updated if needed

def check_gcloud_installation():
    """Check if Google Cloud SDK is installed."""
    global GCLOUD_PATH
    
    # First try the PATH
    try:
        subprocess.run(["gcloud", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        GCLOUD_PATH = "gcloud"
        print("✅ Google Cloud SDK is installed.")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        # Try the specific installation directory
        custom_path = "/Users/anyadecarlo/HYPERA/google-cloud-sdk/bin/gcloud"
        if os.path.exists(custom_path):
            try:
                subprocess.run([custom_path, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                GCLOUD_PATH = custom_path
                print(f"✅ Google Cloud SDK found at {custom_path}")
                return True
            except subprocess.SubprocessError:
                pass
                
        print("❌ Google Cloud SDK is not installed or not in PATH.")
        print("Please install from: https://cloud.google.com/sdk/docs/install")
        return False

def check_authentication():
    """Check if user is authenticated with Google Cloud."""
    try:
        result = subprocess.run(
            [GCLOUD_PATH, "auth", "list"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        
        if "No credentialed accounts." in result.stdout:
            print("❌ Not authenticated with Google Cloud.")
            return False
        
        print("✅ Authenticated with Google Cloud.")
        return True
    except subprocess.SubprocessError:
        print("❌ Failed to check authentication status.")
        return False

def authenticate():
    """Authenticate with Google Cloud."""
    try:
        subprocess.run([GCLOUD_PATH, "auth", "login"], check=True)
        print("✅ Authentication successful.")
        return True
    except subprocess.SubprocessError:
        print("❌ Authentication failed.")
        return False

def set_project(project_id):
    """Set the Google Cloud project."""
    try:
        subprocess.run([GCLOUD_PATH, "config", "set", "project", project_id], check=True)
        print(f"✅ Project set to {project_id}.")
        return True
    except subprocess.SubprocessError:
        print(f"❌ Failed to set project to {project_id}.")
        return False

def enable_apis():
    """Enable required Google Cloud APIs."""
    apis = [
        "compute.googleapis.com",
        "aiplatform.googleapis.com",
        "storage.googleapis.com",
        "containerregistry.googleapis.com"
    ]
    
    for api in apis:
        try:
            print(f"Enabling {api}...")
            subprocess.run([GCLOUD_PATH, "services", "enable", api], check=True)
            print(f"✅ {api} enabled.")
        except subprocess.SubprocessError:
            print(f"❌ Failed to enable {api}.")
            return False
    
    return True

def create_bucket(bucket_name, region):
    """Create a Google Cloud Storage bucket."""
    try:
        # Check if bucket exists
        result = subprocess.run(
            [GCLOUD_PATH, "storage", "buckets", "describe", f"gs://{bucket_name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        
        if result.returncode == 0:
            print(f"✅ Bucket gs://{bucket_name} already exists.")
            return True
        
        # Create bucket
        print(f"Creating bucket gs://{bucket_name}...")
        subprocess.run(
            [GCLOUD_PATH, "storage", "buckets", "create", f"gs://{bucket_name}", "--location", region],
            check=True
        )
        print(f"✅ Bucket gs://{bucket_name} created.")
        return True
    except subprocess.SubprocessError:
        print(f"❌ Failed to create bucket gs://{bucket_name}.")
        return False

def package_code(bucket_name, project_dir, package_name="hypera1_training"):
    """Package the code for deployment to Google Cloud."""
    temp_dir = os.path.join(project_dir, "tmp", "cloud_package")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a package.py file that will be the entry point
    package_py = os.path.join(temp_dir, "package.py")
    with open(package_py, "w") as f:
        f.write("""
import os
import sys
import subprocess
import tarfile
import tempfile

def main():
    # Extract the code
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = os.path.join(tmp_dir, "code.tar.gz")
        with open(tar_path, "wb") as f:
            f.write(CODE)
        
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=tmp_dir)
        
        # Run the training script
        subprocess.run([sys.executable, os.path.join(tmp_dir, "HYPERA1", "legacy", "train_bbbc039_with_agents.py")] + sys.argv[1:], check=True)

if __name__ == "__main__":
    main()
""")
    
    # Create a setup.py file
    setup_py = os.path.join(temp_dir, "setup.py")
    with open(setup_py, "w") as f:
        f.write(f"""
from setuptools import setup, find_packages

setup(
    name="{package_name}",
    version="0.1",
    packages=find_packages(),
    py_modules=["package"],
    entry_points={{
        'console_scripts': [
            '{package_name}=package:main',
        ],
    }},
)
""")
    
    # Package the code
    code_tar = os.path.join(temp_dir, "code.tar.gz")
    with tarfile.open(code_tar, "w:gz") as tar:
        for root, dirs, files in os.walk(project_dir):
            # Skip the tmp directory and any __pycache__ directories
            if "tmp" in root or "__pycache__" in root or ".git" in root:
                continue
            for file in files:
                if file.endswith(".py") or file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, project_dir)
                    tar.add(file_path, arcname=arcname)
    
    # Read the code.tar.gz file as bytes
    with open(code_tar, "rb") as f:
        code_bytes = f.read()
    
    # Create a constants.py file with the code bytes
    constants_py = os.path.join(temp_dir, "constants.py")
    with open(constants_py, "w") as f:
        f.write(f"CODE = {code_bytes!r}\n")
    
    # Build the package
    subprocess.run([sys.executable, "setup.py", "sdist", "bdist_wheel"], cwd=temp_dir, check=True)
    
    # Upload the package to Google Cloud Storage
    wheel_file = None
    for file in os.listdir(os.path.join(temp_dir, "dist")):
        if file.endswith(".whl"):
            wheel_file = os.path.join(temp_dir, "dist", file)
            break
    
    if wheel_file:
        subprocess.run([GCLOUD_PATH, "storage", "cp", wheel_file, f"gs://{bucket_name}/packages/"], check=True)
        print(f"✅ Package uploaded to gs://{bucket_name}/packages/{os.path.basename(wheel_file)}")
        return f"gs://{bucket_name}/packages/{os.path.basename(wheel_file)}"
    else:
        print("❌ Failed to build package.")
        return None

def submit_job(project_id, region, package_uri, bucket_name, job_name=None, machine_type="n1-standard-8", 
               accelerator_type="NVIDIA_TESLA_T4", accelerator_count=1, experiment_type="agent_factory"):
    """Submit a training job to Google Cloud AI Platform."""
    if job_name is None:
        job_name = f"hypera1-training-{int(time.time())}"
    
    # Create a directory for config files
    config_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(config_dir, exist_ok=True)
    
    # Create a config file for Vertex AI
    vertex_config = {
        "display_name": job_name,
        "job_spec": {
            "worker_pool_specs": [
                {
                    "machine_spec": {
                        "machine_type": machine_type,
                    },
                    "replica_count": 1,
                    "python_package_spec": {
                        "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-cpu.1-13:latest" if accelerator_count == 0 else "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest",
                        "package_uris": [package_uri],
                        "python_module": "package",
                        "args": [
                            "--experiment_type", experiment_type,
                            "--epochs", "100",
                            "--batch_size", "16",
                            "--early_stopping", "20"
                        ]
                    }
                }
            ]
        }
    }
    
    # Add accelerator config only if GPUs are requested
    if accelerator_count > 0:
        vertex_config["job_spec"]["worker_pool_specs"][0]["machine_spec"]["accelerator_type"] = accelerator_type
        vertex_config["job_spec"]["worker_pool_specs"][0]["machine_spec"]["accelerator_count"] = accelerator_count
    
    vertex_config_file = os.path.join(config_dir, f"{job_name}_vertex_config.json")
    with open(vertex_config_file, "w") as f:
        json.dump(vertex_config, f, indent=2)
    
    # Create a config file for AI Platform
    ai_platform_config = {
        "jobId": job_name,
        "trainingInput": {
            "scaleTier": "CUSTOM",
            "masterType": machine_type,
            "masterConfig": {
                "acceleratorConfig": {
                    "count": str(accelerator_count),
                    "type": accelerator_type
                } if accelerator_count > 0 else None
            },
            "packageUris": [package_uri],
            "pythonModule": "package",
            "args": [
                "--experiment_type", experiment_type,
                "--epochs", "100",
                "--batch_size", "16",
                "--early_stopping", "20"
            ],
            "region": region,
            "jobDir": f"gs://{bucket_name}/jobs/{job_name}"
        }
    }
    
    ai_platform_config_file = os.path.join(config_dir, f"{job_name}_ai_platform_config.json")
    with open(ai_platform_config_file, "w") as f:
        json.dump(ai_platform_config, f, indent=2)
    
    # Submit the job
    try:
        # First try the new Vertex AI API
        try:
            print("Attempting to submit job using Vertex AI...")
            subprocess.run([
                GCLOUD_PATH, "ai", "custom-jobs", "create",
                "--region", region,
                "--display-name", job_name,
                "--config", vertex_config_file,
                "--project", project_id
            ], check=True)
            
            print(f"✅ Job {job_name} submitted successfully using Vertex AI.")
            print(f"Monitor job at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project_id}")
            return job_name
        except subprocess.SubprocessError as e:
            print(f"Failed to submit job using Vertex AI: {e}")
            print("Falling back to AI Platform...")
            
        # Fall back to the legacy AI Platform
        subprocess.run([
            GCLOUD_PATH, "ai-platform", "jobs", "submit", "training", job_name,
            "--config", ai_platform_config_file,
            "--region", region,
            "--project", project_id
        ], check=True)
        print(f"✅ Job {job_name} submitted successfully using AI Platform.")
        print(f"Monitor job at: https://console.cloud.google.com/ai-platform/jobs/{job_name}?project={project_id}")
        return job_name
    except subprocess.SubprocessError as e:
        print(f"❌ Failed to submit job {job_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Set up and deploy HYPERA1 training to Google Cloud.")
    parser.add_argument("--project-id", required=True, help="Google Cloud project ID")
    parser.add_argument("--region", default="us-central1", help="Google Cloud region")
    parser.add_argument("--bucket-name", help="Google Cloud Storage bucket name (defaults to project-id-hypera1)")
    parser.add_argument("--machine-type", default="n1-standard-8", help="Machine type for training")
    parser.add_argument("--accelerator-type", default="NVIDIA_TESLA_T4", 
                        choices=["NVIDIA_TESLA_K80", "NVIDIA_TESLA_P100", "NVIDIA_TESLA_P4", "NVIDIA_TESLA_T4", "NVIDIA_TESLA_V100"],
                        help="GPU accelerator type")
    parser.add_argument("--accelerator-count", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--experiment-type", default="agent_factory", 
                        choices=["agent_factory", "no_agent", "grid_search"],
                        help="Type of experiment to run")
    parser.add_argument("--setup-only", action="store_true", help="Only set up Google Cloud, don't submit a job")
    
    args = parser.parse_args()
    
    # Set default bucket name if not provided
    if not args.bucket_name:
        args.bucket_name = f"{args.project_id}-hypera1"
    
    # Check Google Cloud SDK installation
    if not check_gcloud_installation():
        return 1
    
    # Check authentication
    if not check_authentication():
        if not authenticate():
            return 1
    
    # Set project
    if not set_project(args.project_id):
        return 1
    
    # Enable APIs
    if not enable_apis():
        return 1
    
    # Create bucket
    if not create_bucket(args.bucket_name, args.region):
        return 1
    
    if args.setup_only:
        print("✅ Google Cloud setup complete.")
        return 0
    
    # Get the project directory (parent of the directory containing this script)
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Package the code
    package_uri = package_code(args.bucket_name, project_dir)
    if not package_uri:
        return 1
    
    # Submit the job
    job_name = submit_job(
        args.project_id, 
        args.region, 
        package_uri, 
        args.bucket_name,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
        experiment_type=args.experiment_type
    )
    
    if not job_name:
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
