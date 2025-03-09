# HYPERA on RunPod

## Quick Start Guide

This guide will help you get HYPERA running on RunPod with GPU acceleration.

### Step 1: Sign Up for RunPod

1. Go to [https://www.runpod.io/](https://www.runpod.io/)
2. Create an account and add credit ($25 is a good starting amount)

### Step 2: Upload Your Code

1. Use the provided packaging script to create a zip file:
   ```bash
   cd /Users/anyadecarlo/HYPERA/HYPERA1/cloud
   ./package_for_runpod.sh
   ```
2. This creates `HYPERA_for_RunPod.zip` which contains everything needed to run HYPERA

### Step 3: Deploy a GPU Pod on RunPod

1. From the RunPod dashboard, click "Deploy"
2. Select a GPU (L4 is recommended for a good balance of performance and cost)
3. Choose the PyTorch template
4. Set your pod configuration:
   - Name: `HYPERA-Training`
   - Container Disk: `80 GB`
   - Volume Disk: `50 GB`
5. Click "Deploy"

### Step 4: Set Up and Run HYPERA

1. Once your pod is running, click on it in the dashboard
2. Click "Connect" and select "Jupyter Lab"
3. In Jupyter Lab, upload `HYPERA_for_RunPod.zip`
4. Open a terminal in Jupyter and run:
   ```bash
   unzip HYPERA_for_RunPod.zip
   ```
5. Open the `HYPERA_Training.ipynb` notebook
6. Follow the steps in the notebook to run your training

## RunPod-Specific Optimizations

The HYPERA codebase has been optimized for RunPod with the following changes:

1. **DataLoader Configuration**: 
   - When running on RunPod, the DataLoader automatically sets `num_workers=0` to avoid multiprocessing issues
   - This is detected via the `RUNPOD_POD_ID` environment variable, which the notebook sets automatically

2. **Memory Management**:
   - The training notebook includes steps to monitor GPU memory usage
   - Batch size can be adjusted if memory issues occur

3. **Error Handling**:
   - Enhanced error logging has been added to help diagnose RunPod-specific issues
   - The notebook includes cells to verify the environment is properly configured

## Cost Management

- RunPod charges by the hour for GPU usage
- When not actively using your pod:
  - Stop the pod to pause billing
  - Resume when ready to continue
- Make sure to download important results before stopping

## Advantages of RunPod for HYPERA

1. **Immediate GPU Access**: No waiting for quota approvals
2. **Powerful GPUs**: L4 GPUs are more powerful than T4s on Google Cloud
3. **Simple Pricing**: Pay-as-you-go with no hidden costs
4. **Jupyter Integration**: Perfect for research workflows
5. **Fast Setup**: Get running in minutes instead of days

## Troubleshooting

If you encounter any issues:

1. **CUDA Out of Memory**: Reduce batch size in the notebook
2. **Package Installation Errors**: Try installing dependencies one by one
3. **Dataset Issues**: Make sure your dataset is properly uploaded and structured

If you encounter any issues with the DataLoader on RunPod:

1. Verify that the RunPod environment variable is set:
   ```python
   import os
   print(os.environ.get('RUNPOD_POD_ID'))
   ```
   This should return a value (not None)

2. Check that the training script includes RunPod detection:
   ```python
   with open('HYPERA1/legacy/train_bbbc039_with_agents.py', 'r') as f:
       print('RunPod detection:' in f.read())
   ```
   This should return True

3. If issues persist, manually set num_workers=0 in the DataLoader configuration

For more detailed instructions, see the full guide at `RUNPOD_SETUP.md`.
