# Setting Up HYPERA on RunPod

This guide will walk you through setting up and running your HYPERA training on RunPod with GPU acceleration.

## Step 1: Create a RunPod Account

1. Go to [https://www.runpod.io/](https://www.runpod.io/)
2. Sign up for an account
3. Add credit to your account ($25 is a good starting point)

## Step 2: Deploy a GPU Pod

1. From the RunPod dashboard, click "Deploy" in the left sidebar
2. Select a GPU type - the L4 is recommended for a good balance of performance and cost
3. Choose a template - select "PyTorch" from the template dropdown
4. Set your pod configuration:
   - Name: `HYPERA-Training`
   - Container Disk: `80 GB` (default)
   - Volume Disk: `50 GB` (should be enough for your dataset)
5. Click "Deploy" to create your pod

## Step 3: Access Your Pod

1. Once your pod is running, click on it in the dashboard
2. Click "Connect" and select "Jupyter Lab" to open the Jupyter interface
3. You'll be taken to a Jupyter Lab environment running on your GPU pod

## Step 4: Upload Your HYPERA Code

There are two ways to get your code onto RunPod:

### Option 1: Upload via Jupyter (Recommended for First Time)

1. In Jupyter Lab, click the upload button (arrow up icon) in the file browser
2. Select all your HYPERA project files or zip them first and upload the zip file
3. If you uploaded a zip file, open a terminal in Jupyter and run:
   ```bash
   unzip your-zipfile.zip
   ```

### Option 2: Clone from GitHub (If Your Code is in a Repository)

1. Open a terminal in Jupyter Lab
2. Run:
   ```bash
   git clone https://github.com/your-username/HYPERA.git
   ```

## Step 5: Install Dependencies

1. Open a terminal in Jupyter Lab
2. Create a new conda environment for HYPERA:
   ```bash
   conda create -n hypera python=3.10
   conda activate hypera
   ```
3. Install the required packages:
   ```bash
   pip install torch torchvision torchaudio
   pip install monai
   pip install matplotlib scikit-learn scikit-image
   pip install tensorboard
   pip install stable-baselines3
   ```

## Step 6: Prepare Your Dataset

1. Upload your BBBC039 dataset to the pod or download it directly:
   ```bash
   # Create directories
   mkdir -p BBBC039/images BBBC039/masks BBBC039_metadata
   
   # Download dataset (example - adjust URLs as needed)
   wget -P BBBC039/images https://data.broadinstitute.org/bbbc/BBBC039/images.zip
   wget -P BBBC039/masks https://data.broadinstitute.org/bbbc/BBBC039/masks.zip
   
   # Extract files
   unzip BBBC039/images/images.zip -d BBBC039/images/
   unzip BBBC039/masks/masks.zip -d BBBC039/masks/
   ```

2. Create training and validation split files:
   ```bash
   # Create metadata files (you'll need to adjust this based on your actual dataset)
   ls BBBC039/masks/*.png > BBBC039_metadata/all_masks.txt
   
   # Split into training and validation (80/20 split)
   head -n $(( $(wc -l < BBBC039_metadata/all_masks.txt) * 80 / 100 )) BBBC039_metadata/all_masks.txt > BBBC039_metadata/training.txt
   tail -n $(( $(wc -l < BBBC039_metadata/all_masks.txt) * 20 / 100 )) BBBC039_metadata/all_masks.txt > BBBC039_metadata/validation.txt
   ```

## Step 7: Create a Training Notebook

Create a new Jupyter notebook (`HYPERA_Training.ipynb`) with the following content:

```python
# HYPERA Training on RunPod
import os
import sys
import matplotlib.pyplot as plt
import torch

# Add HYPERA to path
hypera_path = os.path.abspath("HYPERA1")
if hypera_path not in sys.path:
    sys.path.append(hypera_path)

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Import HYPERA training module
from legacy.train_bbbc039_with_agents import main

# Run training with agent-based hyperparameter optimization
main(
    experiment_type="agent_factory",  # Use the agent-based approach
    epochs=100,                       # Number of epochs
    batch_size=16,                    # Batch size
    early_stopping=20,                # Early stopping patience
    use_cloud=False                   # Don't use cloud storage
)
```

## Step 8: Run Your Training

1. Open the `HYPERA_Training.ipynb` notebook
2. Make sure your conda environment is selected as the kernel
3. Run the notebook cells to start training

## Step 9: Monitor Training Progress

1. The training progress will be displayed in the notebook output
2. You can also use TensorBoard to visualize training metrics:
   ```python
   # In a new notebook cell
   %load_ext tensorboard
   %tensorboard --logdir=./runs
   ```

## Step 10: Save Your Results

1. When training is complete, download your model and results:
   - In Jupyter Lab, right-click on the results folder and select "Download"
2. Alternatively, you can use the RunPod file browser to download files

## Cost Management

- RunPod charges by the hour for GPU usage
- When you're not actively using your pod, you can:
  1. Stop the pod (which will save your data but stop billing for GPU time)
  2. Resume the pod when you're ready to continue working
- Make sure to download important results before stopping your pod

## Comparing Agent vs. No-Agent Approaches

To compare your agent-based approach with traditional hyperparameter tuning:

1. Run the training with agents first:
   ```python
   main(experiment_type="agent_factory", epochs=100, batch_size=16)
   ```

2. Then run without agents:
   ```python
   main(experiment_type="no_agent", epochs=100, batch_size=16)
   ```

3. Compare the results in terms of:
   - Final model performance
   - Training time
   - Convergence speed
