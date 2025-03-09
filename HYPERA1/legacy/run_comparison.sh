#!/bin/bash
# Script to compare standard MONAI UNet vs. agent-based hyperparameter optimization

# Create results directories
mkdir -p results/bbbc039/standard
mkdir -p results/bbbc039/agent

# Run training with standard MONAI UNet (no agents)
echo "Running training with standard MONAI UNet (no agents)..."
python legacy/train_bbbc039_without_agents.py \
  --optimizer SGD \
  --loss DiceCE \
  --learning_rate 0.001 \
  --epochs 100 \
  --batch_size 8 \
  --augmentations All \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --gradient_clip 1.0 \
  --early_stopping 30 \
  --output_dir results/bbbc039/standard \
  > results/bbbc039/standard/training_log.txt

# Run training with agent-based hyperparameter optimization
echo "Running training with agent-based hyperparameter optimization..."
python legacy/train_bbbc039_with_agents.py \
  --optimizer SGD \
  --loss DiceCE \
  --learning_rate 0.001 \
  --epochs 100 \
  --batch_size 8 \
  --augmentations All \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --gradient_clip 1.0 \
  --early_stopping 30 \
  --experiment_type agent_factory \
  --conflict_resolution priority \
  --agent_update_frequency 1 \
  --output_dir results/bbbc039/agent \
  > results/bbbc039/agent/training_log.txt

# Compare results
echo "Comparison complete. Check results in results/bbbc039/standard and results/bbbc039/agent directories."
echo "To visualize and compare the results, run: python legacy/compare_results.py --standard_dir results/bbbc039/standard --agent_dir results/bbbc039/agent"
