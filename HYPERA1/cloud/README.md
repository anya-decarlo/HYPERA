# Google Cloud Integration for HYPERA1

This directory contains scripts to help you run HYPERA1 training jobs on Google Cloud AI Platform (Vertex AI), which provides access to powerful GPU resources for faster training.

## Prerequisites

1. **Google Cloud Account**: You need a Google Cloud account. If you don't have one, sign up at [cloud.google.com](https://cloud.google.com/).

2. **Google Cloud SDK**: Install the Google Cloud SDK by following the instructions at [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install).

3. **Google Cloud Project**: Create a project in the Google Cloud Console at [console.cloud.google.com](https://console.cloud.google.com/).

4. **Billing Account**: Enable billing for your project. Google Cloud offers a free tier with $300 in credits for new users.

## Setting Up and Running Training on Google Cloud

### 1. Make the scripts executable

```bash
chmod +x setup_gcloud.py download_results.py
```

### 2. Set up Google Cloud and submit a training job

```bash
./setup_gcloud.py --project-id YOUR_PROJECT_ID --experiment-type agent_factory
```

Replace `YOUR_PROJECT_ID` with your Google Cloud project ID. This script will:

- Check if Google Cloud SDK is installed
- Authenticate with Google Cloud
- Set the project
- Enable required APIs
- Create a storage bucket
- Package the code
- Submit a training job with a T4 GPU

### 3. Monitor the job

You can monitor the job in the Google Cloud Console or by running:

```bash
gcloud ai-platform jobs describe JOB_NAME --project YOUR_PROJECT_ID
```

Replace `JOB_NAME` with the name of the job (printed by the setup script).

### 4. Download the results

Once the job is complete, download the results:

```bash
./download_results.py --project-id YOUR_PROJECT_ID --job-name JOB_NAME
```

This will download all the results to a local directory.

## Advanced Options

### Choosing different GPU types

You can choose different GPU types by using the `--accelerator-type` flag:

```bash
./setup_gcloud.py --project-id YOUR_PROJECT_ID --accelerator-type NVIDIA_TESLA_V100 --accelerator-count 1
```

Available options:
- `NVIDIA_TESLA_K80`
- `NVIDIA_TESLA_P100`
- `NVIDIA_TESLA_P4`
- `NVIDIA_TESLA_T4`
- `NVIDIA_TESLA_V100`

### Running different experiment types

You can run different experiment types by using the `--experiment-type` flag:

```bash
./setup_gcloud.py --project-id YOUR_PROJECT_ID --experiment-type no_agent
```

Available options:
- `agent_factory`
- `no_agent`
- `grid_search`

### Setting up without submitting a job

If you just want to set up Google Cloud without submitting a job:

```bash
./setup_gcloud.py --project-id YOUR_PROJECT_ID --setup-only
```

### Listing all jobs

To list all jobs in your project:

```bash
./download_results.py --project-id YOUR_PROJECT_ID --list-jobs
```

## Cost Management

Remember that running jobs on Google Cloud incurs costs. To manage costs:

1. Use the smallest machine type and GPU that meets your needs
2. Set appropriate early stopping parameters
3. Monitor your billing in the Google Cloud Console
4. Delete resources when you're done with them

## Troubleshooting

If you encounter issues:

1. Check the job logs in the Google Cloud Console
2. Ensure you have sufficient quota for the resources you're requesting
3. Verify that billing is enabled for your project
4. Make sure all required APIs are enabled
