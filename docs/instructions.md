# HYPERA Training Instructions

This guide provides quick commands for running the different training scripts in the HYPERA project.

## 1. Segmentation Agent Training

### Basic Training with Default Settings
```bash
cd /Users/anyadecarlo/HYPERA
python HYPERA1/train_segmentation_agents.py
```

### Training with Custom Parameters
```bash
cd /Users/anyadecarlo/HYPERA
python HYPERA1/train_segmentation_agents.py --num_epochs 10 --batch_size 4 --learning_rate 0.001 --verbose
```

### Available Parameters
- `--num_epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 4)
- `--learning_rate` or `--lr`: Learning rate (default: 0.001)
- `--verbose`: Enable verbose logging
- `--device`: Specify device (e.g., 'cuda:0', 'cpu')
- `--results_dir`: Directory to save results (default: 'results_with_segmentation_agents')
- `--data_dir`: Directory containing BBBC039 dataset (default: 'BBBC039')

### Training with Specific Agents
```bash
cd /Users/anyadecarlo/HYPERA
# Use region agent only
python HYPERA1/train_segmentation_agents.py --use_region_agent

# Use boundary agent only
python HYPERA1/train_segmentation_agents.py --use_boundary_agent

# Use shape agent only
python HYPERA1/train_segmentation_agents.py --use_shape_agent

# Use foreground-background balance agent only
python HYPERA1/train_segmentation_agents.py --use_fg_balance_agent
```

### Training with All Agents
```bash
cd /Users/anyadecarlo/HYPERA
python HYPERA1/train_segmentation_agents.py --use_all_agents
```

## 2. BBBC039 Training with Agents

### Basic Training with Agents
```bash
cd /Users/anyadecarlo/HYPERA
python HYPERA1/legacy/train_bbbc039_with_agents.py
```

### Training with Custom Parameters
```bash
cd /Users/anyadecarlo/HYPERA
python HYPERA1/legacy/train_bbbc039_with_agents.py --num_epochs 10 --batch_size 4 --learning_rate 0.001 --verbose
```

## 3. BBBC039 Training without Agents (Baseline)

### Basic Training without Agents
```bash
cd /Users/anyadecarlo/HYPERA
python HYPERA1/legacy/train_bbbc039_without_agents.py
```

### Training with Custom Parameters
```bash
cd /Users/anyadecarlo/HYPERA
python HYPERA1/legacy/train_bbbc039_without_agents.py --num_epochs 10 --batch_size 4 --learning_rate 0.001 --verbose
```

## 4. Testing Segmentation Agents

```bash
cd /Users/anyadecarlo/HYPERA
python HYPERA1/test_segmentation_agents.py --model_path /path/to/saved/model.pth
```

## 5. Comparing Results

```bash
cd /Users/anyadecarlo/HYPERA
python HYPERA1/legacy/compare_results.py --with_agents_dir results_with_agents --without_agents_dir results_no_agent
```

## 6. Creating Region Adjacency Graphs (RAGs)

```bash
cd /Users/anyadecarlo/HYPERA
python HYPERA1/legacy/RAGS.py --mask_dir /path/to/segmentation/masks --output_dir /path/to/output
```

## Tips for Best Results

1. **For Visualization**: Add the `--verbose` flag to see more detailed logs and visualizations.

2. **Saving Segmentation Masks**: The segmentation masks are automatically saved in the results directory when running `train_segmentation_agents.py`. These masks can be used for creating Region Adjacency Graphs.

3. **GPU Acceleration**: If you have a CUDA-capable GPU, the scripts will automatically use it. You can specify a different device using the `--device` parameter.

4. **Batch Size**: Adjust the batch size according to your GPU memory. Smaller batch sizes (2-4) work well for most setups.

5. **Learning Rate**: A learning rate of 0.001 is a good starting point. For fine-tuning, try values between 0.0001 and 0.0005.

6. **Number of Epochs**: For quick tests, use 1-5 epochs. For better results, train for 10-50 epochs.

7. **Using Multiple Agents**: For best segmentation results, use all agents together with `--use_all_agents`. Each agent specializes in different aspects of segmentation:
   - Region Agent: Focuses on optimizing regional overlap (Dice score)
   - Boundary Agent: Focuses on optimizing boundary accuracy
   - Shape Agent: Focuses on optimizing shape regularization
   - FG Balance Agent: Focuses on optimizing foreground-background balance

## Troubleshooting

- If you encounter CUDA out-of-memory errors, reduce the batch size.
- If training is unstable, reduce the learning rate.
- If segmentation results are poor, try increasing the number of epochs or adjusting the model architecture.
