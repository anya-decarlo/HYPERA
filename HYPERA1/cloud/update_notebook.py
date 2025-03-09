#!/usr/bin/env python3
"""
Script to update the HYPERA_Training.ipynb notebook with the latest fixes
"""
import json
import os
import re

def update_notebook():
    """Update the HYPERA_Training.ipynb notebook with the latest fixes"""
    notebook_path = os.path.join(os.path.dirname(__file__), "HYPERA_Training.ipynb")
    
    # Read the notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Uncomment the dataset download code
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            # Check if this is the dataset download cell
            if any("if not os.path.exists(\"BBBC039\")" in line for line in source):
                # Uncomment the download lines
                for i, line in enumerate(source):
                    if re.match(r'\s*#\s*!mkdir -p BBBC039', line):
                        source[i] = line.replace('# !', '!')
                    elif re.match(r'\s*#\s*!wget -P BBBC039', line):
                        source[i] = line.replace('# !', '!')
                    elif re.match(r'\s*#\s*!unzip BBBC039', line):
                        source[i] = line.replace('# !', '!')
    
    # Add a new cell to fix the loss function configuration
    fix_loss_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Fix the loss function configuration in train_bbbc039_with_agents.py\n",
            "import re\n",
            "\n",
            "train_file_path = os.path.join(hypera_path, \"legacy\", \"train_bbbc039_with_agents.py\")\n",
            "\n",
            "with open(train_file_path, 'r') as file:\n",
            "    content = file.read()\n",
            "\n",
            "# Check if the fix is already applied\n",
            "if \"to_onehot_y=True\" in content:\n",
            "    print(\"Applying loss function fix to train_bbbc039_with_agents.py...\")\n",
            "    \n",
            "    # Replace to_onehot_y=True with to_onehot_y=False in loss functions\n",
            "    content = re.sub(\n",
            "        r'to_onehot_y=True',\n",
            "        r'to_onehot_y=False',  # Labels are already one-hot encoded',\n",
            "        content\n",
            "    )\n",
            "    \n",
            "    with open(train_file_path, 'w') as file:\n",
            "        file.write(content)\n",
            "    \n",
            "    print(\"Loss function fix applied successfully!\")\n",
            "else:\n",
            "    print(\"Loss function fix already applied or not needed.\")\n"
        ]
    }
    
    # Add a new cell for RunPod-specific fixes
    runpod_fix_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Set environment variable to indicate we're on RunPod\n",
            "import os\n",
            "os.environ['RUNPOD_POD_ID'] = 'NOTEBOOK_ENVIRONMENT'\n",
            "print(\"Set RunPod environment variable to ensure proper DataLoader configuration\")\n",
            "\n",
            "# Check if our RunPod detection code is present\n",
            "train_file_path = os.path.join(hypera_path, \"legacy\", \"train_bbbc039_with_agents.py\")\n",
            "\n",
            "with open(train_file_path, 'r') as file:\n",
            "    content = file.read()\n",
            "\n",
            "if \"is_runpod = os.environ.get('RUNPOD_POD_ID')\" not in content:\n",
            "    print(\"WARNING: RunPod detection code not found in training script.\")\n",
            "    print(\"The script may not be configured for RunPod environment.\")\n",
            "    print(\"Please check the latest version of the code.\")\n",
            "else:\n",
            "    print(\"RunPod detection code found in training script.\")\n",
            "    print(\"DataLoader will use 0 worker processes to avoid multiprocessing issues.\")\n"
        ]
    }
    
    # Find the position to insert the new cells
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown' and '## 5. Fix MONAI Error' in ''.join(cell['source']):
            # Insert after this markdown cell and its corresponding code cell
            insert_position = i + 2
            break
    else:
        # If not found, insert before the training cell
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'markdown' and '## 6. Run Training' in ''.join(cell['source']):
                insert_position = i
                break
        else:
            # If still not found, append to the end
            insert_position = len(notebook['cells'])
    
    # Insert a new markdown cell for the loss function fix
    loss_fix_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5b. Fix Loss Function Configuration\n",
            "\n",
            "Apply the fix for the loss function configuration to avoid double one-hot encoding."
        ]
    }
    
    # Insert a new markdown cell for the RunPod fix
    runpod_fix_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5c. Apply RunPod-Specific Fixes\n",
            "\n",
            "Apply fixes to ensure compatibility with RunPod environment."
        ]
    }
    
    # Insert the cells
    notebook['cells'].insert(insert_position, loss_fix_markdown)
    notebook['cells'].insert(insert_position + 1, fix_loss_cell)
    notebook['cells'].insert(insert_position + 2, runpod_fix_markdown)
    notebook['cells'].insert(insert_position + 3, runpod_fix_cell)
    
    # Write the updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Updated {notebook_path} with the latest fixes")

if __name__ == "__main__":
    update_notebook()
