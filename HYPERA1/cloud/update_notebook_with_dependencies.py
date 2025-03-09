#!/usr/bin/env python3
"""
Script to update the HYPERA_Training.ipynb notebook with a dependencies installation cell.
"""
import json
import os

def update_notebook():
    """Add a dependency installation cell to the notebook."""
    notebook_path = os.path.join(os.path.dirname(__file__), "HYPERA_Training.ipynb")
    
    # Read the notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Create a new cell for installing dependencies
    dependencies_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": [
            "# Install required dependencies\n",
            "# Run this cell first to ensure all necessary packages are installed\n",
            "\n",
            "# Essential packages\n",
            "!pip install ruptures==1.1.9  # Change point detection\n",
            "!pip install git+https://github.com/Project-MONAI/MONAI.git@7c26e5af385eb5f7a813fa405c6f3fc87b7511fa  # Medical image processing\n",
            "!pip install torch==2.7.0.dev20250221 torchvision==0.22.0.dev20250221  # Deep learning\n",
            "!pip install numpy==1.26.4 pandas==2.2.3 matplotlib==3.10.0  # Data analysis and visualization\n",
            "!pip install scikit-learn scikit-image==0.25.2 scipy==1.13.1  # Scientific computing\n",
            "!pip install tqdm tensorboard==2.19.0  # Progress and visualization\n",
            "!pip install nibabel==5.3.2  # Medical image I/O\n",
            "!pip install statsmodels==0.14.4  # Time series analysis and statistical models\n",
            "!pip install torchmetrics==1.2.1  # Additional PyTorch metrics\n",
            "\n",
            "# Optional packages\n",
            "!pip install h5py==3.13.0 SimpleITK==2.4.1 opencv-python networkx pillow\n",
            "\n",
            "# Restart the kernel to ensure changes take effect\n",
            "import os\n",
            "os.kill(os.getpid(), 9)\n"
        ],
        "outputs": []
    }
    
    # Create a markdown cell to explain
    dependencies_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 0. Install Dependencies\n",
            "\n",
            "First, let's make sure all required packages are installed. Run the cell below to install the necessary dependencies.\n",
            "\n",
            "**Note**: After running this cell, you'll need to restart the kernel as the cell will force a restart."
        ]
    }
    
    # Insert the cells at the beginning of the notebook (after any potential header cells)
    # Find a good position to insert (after any introductory markdown)
    insert_position = 0
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown' and any("Introduction" in line for line in cell['source']):
            insert_position = i + 1
            break
    
    notebook['cells'].insert(insert_position, dependencies_markdown)
    notebook['cells'].insert(insert_position + 1, dependencies_cell)
    
    # Write the updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Updated {notebook_path} with dependency installation cell")

if __name__ == "__main__":
    update_notebook()
