#!/usr/bin/env python3
"""
Script to fix the main() function call in HYPERA_Training.ipynb
"""
import json
import os
import re

def fix_notebook():
    """Fix the main() function call in HYPERA_Training.ipynb"""
    notebook_path = os.path.join(os.path.dirname(__file__), "HYPERA_Training.ipynb")
    
    # Read the notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Find the cell with the main() function call
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            # Check if this is the cell calling main() with arguments
            if any("main(" in line and "experiment_type" in line for line in source):
                # Replace with the wrapper function
                new_source = [
                    "# Import HYPERA training module\n",
                    "from legacy.train_bbbc039_with_agents import main\n",
                    "\n",
                    "def run_training_with_agents(epochs=100, batch_size=16, early_stopping=20):\n",
                    "    \"\"\"Run HYPERA training with agent-based hyperparameter optimization\"\"\"\n",
                    "    import sys\n",
                    "    from legacy.train_bbbc039_with_agents import main\n",
                    "    \n",
                    "    # Set command line arguments\n",
                    "    sys.argv = [\n",
                    "        'train_bbbc039_with_agents.py',\n",
                    "        '--experiment_type', 'agent_factory',\n",
                    "        '--epochs', str(epochs),\n",
                    "        '--batch_size', str(batch_size),\n",
                    "        '--early_stopping', str(early_stopping)\n",
                    "    ]\n",
                    "    \n",
                    "    # Run training\n",
                    "    main()\n",
                    "\n",
                    "# Run training with agent-based hyperparameter optimization\n",
                    "run_training_with_agents()\n"
                ]
                cell['source'] = new_source
                break
    
    # Write the updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Updated {notebook_path} with correct main() function call")

if __name__ == "__main__":
    fix_notebook()
