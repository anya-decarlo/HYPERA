# Google Cloud Training Guide for HYPERA

This guide will walk you through setting up and running your HYPERA training job on Google Cloud using the web console.

## Prerequisites

1. You have a Google Cloud account with billing enabled
2. Your project ID is set up (e.g., `hypera-training`)
3. You have uploaded your training package to Google Cloud Storage
4. You have enabled the Vertex AI API (run: `/Users/anyadecarlo/HYPERA/google-cloud-sdk/bin/gcloud services enable aiplatform.googleapis.com --project hypera-training`)

## Step 1: Access Vertex AI Custom Jobs

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project (`hypera-training`) from the dropdown at the top of the page
3. Click on the navigation menu (hamburger icon ☰) in the top-left corner
4. Scroll down to "Artificial Intelligence" section and click on "Vertex AI"
5. In the Vertex AI sidebar menu, click on "Training"
6. Click on the "Custom jobs" tab in the center of the page

## Step 2: Create a New Custom Job

1. Click the "Train new model" button at the top of the page
2. On the next page, select "Custom training (advanced)" option
3. Fill in the following details:
   - **Name**: `hypera1-training-[timestamp]` (e.g., `hypera1-training-20250308`)
   - **Region**: `us-west2` (Los Angeles) or your preferred region

## Step 3: Configure the Python Package

1. In the **Model settings** section, select **Python Package**
2. Fill in the following details:
   - **Package location**: `gs://hypera-training-hypera1/packages/hypera1_training-0.1-py3-none-any.whl`
   - **Python module**: `package`
   - **Container image**: `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest`
   - **Arguments**: Add the following arguments one by one by clicking "ADD ARGUMENT":
      - `--experiment_type=agent_factory`
      - `--epochs=100`
      - `--batch_size=16`
      - `--early_stopping=20`

## Step 4: Configure Compute Resources

1. In the **Compute and pricing** section:
   - **Machine type**: `n1-standard-8`
   - **Accelerator type**: `NVIDIA_TESLA_V100`
   - **Accelerator count**: `1`

## Step 5: Start the Training Job

1. Click **START TRAINING** at the bottom of the page to start the training job

## Handling Quota Issues

If you encounter quota errors like:
```
The following quota metrics exceed quota limits: 
aiplatform.googleapis.com/custom_model_training_cpus,
aiplatform.googleapis.com/custom_model_training_nvidia_t4_gpus
```

You have several options:

### Option 1: Run with CPU-only (No GPU)

1. Follow the same steps as above, but in Step 4:
   - **Machine type**: Choose a smaller machine like `n1-standard-4`
   - **Accelerator type**: Do not select any accelerator (CPU-only)
   - **Container image**: Make sure to use the CPU version: `us-docker.pkg.dev/vertex-ai/training/pytorch-cpu.1-13:latest`

### Option 2: Try a Different Region

Different regions have different quota allocations. Try:
- `us-central1` (Iowa)
- `us-east1` (South Carolina)
- `us-east4` (Northern Virginia)

### Option 3: Request a Quota Increase

1. Go to IAM & Admin > Quotas
2. Filter for "custom_model_training_cpus" and "custom_model_training_nvidia_t4_gpus"
3. Select these quotas and click "EDIT QUOTAS"
4. Request an increase (this may take 24-48 hours to be approved)

### Option 4: Run with No-Agent Mode

If you're just testing the infrastructure, you can run with:
- Add the argument: `--experiment_type=no_agent`
- This will disable the agent system and use fixed hyperparameters
- This requires less computational resources

## Monitoring Your Job

1. Go to **Vertex AI** > **Training** > **Custom Jobs**
2. Click on your job name to see details
3. You can view logs, metrics, and the status of your job

## Accessing Results

Your training results will be stored in Google Cloud Storage at:
`gs://hypera-training-hypera1/jobs/[job-name]/`

You can download them using the Google Cloud Console or the `gcloud` command:

```bash
/Users/anyadecarlo/HYPERA/google-cloud-sdk/bin/gcloud storage cp -r gs://hypera-training-hypera1/jobs/[job-name]/ /Users/anyadecarlo/HYPERA/results/
```

## Running Locally for Testing

You can also run the training script locally to test it before submitting to the cloud:

```bash
cd /Users/anyadecarlo/HYPERA/HYPERA1/cloud
./run_training.py --experiment_type agent_factory --epochs 5 --batch_size 8
```

This will run a shorter training job locally to verify everything works correctly.
