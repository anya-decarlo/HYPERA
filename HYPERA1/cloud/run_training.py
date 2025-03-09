#!/usr/bin/env python
"""
Script to run HYPERA1 training directly.
This script can be used to run training locally or in the cloud.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run HYPERA1 training.")
    parser.add_argument("--experiment_type", default="agent_factory", 
                        choices=["agent_factory", "no_agent", "grid_search"],
                        help="Type of experiment to run")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--early_stopping", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--use_cloud", action="store_true", help="Use cloud storage for results")
    
    args = parser.parse_args()
    
    # Add the project directory to the path
    project_dir = str(Path(__file__).parent.parent)
    sys.path.insert(0, project_dir)
    
    # Import the train_bbbc039_with_agents module
    try:
        from legacy import train_bbbc039_with_agents
    except ImportError:
        # Try to load it directly from the file
        spec = importlib.util.spec_from_file_location(
            "train_bbbc039_with_agents", 
            os.path.join(project_dir, "legacy", "train_bbbc039_with_agents.py")
        )
        train_bbbc039_with_agents = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_bbbc039_with_agents)
    
    # Run the training
    train_bbbc039_with_agents.main(
        experiment_type=args.experiment_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping=args.early_stopping,
        use_cloud=args.use_cloud
    )

if __name__ == "__main__":
    main()
