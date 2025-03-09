# MONAI UNet Hyperparameters Guide

This document provides a comprehensive overview of hyperparameters in MONAI's UNet implementation, categorized by when and how they can be modified.

## Table of Contents
1. [Fixed Architecture Hyperparameters](#fixed-architecture-hyperparameters)
2. [Modifiable During Training Hyperparameters](#modifiable-during-training-hyperparameters)
3. [Automatically Adapting Parameters](#automatically-adapting-parameters)
4. [Training Configuration Parameters](#training-configuration-parameters)
5. [Loss Function Parameters](#loss-function-parameters)
6. [Data Augmentation Parameters](#data-augmentation-parameters)
7. [Recommended Hyperparameter Ranges](#recommended-hyperparameter-ranges)

## Fixed Architecture Hyperparameters

These hyperparameters define the UNet architecture and **cannot be changed during training** without redefining the model:

| Hyperparameter | Description | Default Value | Notes |
|----------------|-------------|---------------|-------|
| `dimensions` | Number of spatial dimensions (2D or 3D) | Required | Must be 2 or 3 |
| `in_channels` | Number of input channels | Required | Typically 1 for grayscale, 3 for RGB |
| `out_channels` | Number of output channels | Required | Typically number of segmentation classes |
| `channels` | Sequence of feature channel numbers | (16, 32, 64, 128, 256) | Controls network capacity at each level |
| `strides` | Sequence of stride for each down/upsampling | (2, 2, 2, 2) | Controls spatial reduction between levels |
| `kernel_size` | Convolution kernel size | 3 | Affects receptive field size |
| `up_kernel_size` | Transposed convolution kernel size | 3 | For upsampling operations |
| `num_res_units` | Number of residual units per layer | 0 | Increases model capacity and gradient flow |
| `act` | Activation function | "PRELU" | Options include "RELU", "LEAKYRELU", etc. |
| `norm` | Normalization type | "INSTANCE" | Options include "BATCH", "GROUP", etc. |
| `dropout` | Dropout rate | 0.0 | Controls regularization strength |
| `bias` | Whether to have bias terms | True | Affects optimization dynamics |
| `adn_ordering` | Order of Act, Dropout, Norm | "NDA" | Affects gradient flow and training dynamics |

## Modifiable During Training Hyperparameters

These hyperparameters **can be modified during training** without requiring model redefinition:

| Hyperparameter | Description | How to Modify | Effect of Change |
|----------------|-------------|---------------|------------------|
| `learning_rate` | Step size for optimizer updates | `optimizer.param_groups[0]['lr'] = new_lr` | Affects convergence speed and stability |
| `momentum` | Momentum for SGD optimizer | `optimizer.param_groups[0]['momentum'] = new_momentum` | Controls optimization trajectory |
| `weight_decay` | L2 regularization strength | `optimizer.param_groups[0]['weight_decay'] = new_wd` | Affects model generalization |
| `dropout_prob` | Dropout probability during inference | `model.eval()` with `torch.nn.functional.dropout(x, p=new_prob, training=True)` | Controls prediction variability |
| `aux_loss_weight` | Weight for auxiliary losses | Direct assignment to loss weight variable | Balances multiple loss components |
| `class_weights` | Per-class weights in loss function | `loss_fn.weight = new_weights` | Addresses class imbalance |
| `focal_loss_gamma` | Focusing parameter in focal loss | `loss_fn.gamma = new_gamma` | Controls focus on hard examples |
| `dice_loss_smooth` | Smoothing factor in Dice loss | `loss_fn.smooth = new_smooth` | Prevents division by zero, affects gradient |
| `loss_function_weights` | Weights for combined losses | Direct assignment to weight variables | Balances different loss components |

### Learning Rate Schedulers

MONAI/PyTorch provides several learning rate schedulers that automatically modify the learning rate during training:

| Scheduler | Description | Key Parameters | When LR Changes |
|-----------|-------------|----------------|-----------------|
| `ReduceLROnPlateau` | Reduces LR when metric plateaus | `factor`, `patience`, `min_lr` | After validation step when metric stagnates |
| `StepLR` | Reduces LR at fixed intervals | `step_size`, `gamma` | Every `step_size` epochs |
| `CosineAnnealingLR` | Cosine annealing schedule | `T_max`, `eta_min` | Every optimizer step |
| `WarmupCosineSchedule` | Warmup + cosine decay | `warmup_steps`, `t_total` | Every optimizer step |
| `PolynomialLR` | Polynomial decay | `power`, `total_iters` | Every optimizer step |

## Automatically Adapting Parameters

These parameters **automatically adapt during training** without explicit modification:

| Parameter | Description | Adaptation Mechanism | Notes |
|-----------|-------------|----------------------|-------|
| `BatchNorm` statistics | Running mean and variance | Updated during forward passes in training mode | Tracks data distribution changes |
| `InstanceNorm` statistics | Per-instance normalization | Computed on-the-fly for each input | Adapts to each input independently |
| `Adaptive optimizers` states | Optimizer internal states | Updated based on gradients (e.g., Adam's momentum buffers) | Adapts step sizes per parameter |
| `Gradient scaling` | Scale factor for mixed precision | Adjusted based on gradient overflow detection | Only in AMP training |
| `Attention weights` | Self-attention mechanism weights | Updated through backpropagation | Only in attention-based UNet variants |

## Training Configuration Parameters

These parameters configure the training process but are not part of the model architecture:

| Parameter | Description | Typical Values | Notes |
|-----------|-------------|----------------|-------|
| `batch_size` | Number of samples per batch | 2-32 (depends on GPU memory) | Affects optimization dynamics and memory usage |
| `num_epochs` | Total training epochs | 100-1000 | Depends on dataset size and complexity |
| `validation_interval` | Epochs between validations | 1-5 | Trade-off between training speed and monitoring |
| `amp` | Whether to use mixed precision | True/False | Speeds up training on compatible hardware |
| `clip_grad_norm` | Gradient clipping threshold | 1.0-10.0 | Prevents exploding gradients |
| `early_stopping_patience` | Epochs before early stopping | 10-50 | Prevents overfitting |
| `sliding_window_batch_size` | Batch size for inference | 1-4 | Affects inference memory usage |
| `sliding_window_overlap` | Overlap ratio for inference | 0.25-0.75 | Affects inference quality and speed |
| `optimizer_type` | Choice of optimizer | "Adam", "SGD", "AdamW" | Affects convergence behavior |
| `scheduler_type` | Type of LR scheduler | "plateau", "cosine", "step" | Controls learning rate trajectory |
| `gradient_accumulation_steps` | Steps before optimizer update | 1-8 | Simulates larger batch sizes |

## Loss Function Parameters

These parameters control the loss functions used for training and **can be modified during training**:

| Parameter | Description | Typical Values | How to Modify |
|-----------|-------------|----------------|---------------|
| `class_weights` | Per-class weights in loss function | Inversely proportional to class frequency | `loss_fn.weight = torch.tensor([w1, w2, ...])` |
| `dice_include_background` | Whether to include background in Dice | True/False | `loss_fn.include_background = new_value` |
| `dice_to_ce_ratio` | Ratio of Dice to CE loss | 0.5-1.0 | Direct assignment to combined loss weights |
| `focal_loss_alpha` | Class balancing in focal loss | 0.25-0.75 | `loss_fn.alpha = new_alpha` |
| `focal_loss_gamma` | Focus on hard examples | 1.0-5.0 | `loss_fn.gamma = new_gamma` |
| `boundary_loss_weight` | Weight for boundary loss | 0.0-1.0 | Direct assignment to loss weight |
| `hausdorff_loss_weight` | Weight for Hausdorff loss | 0.0-1.0 | Direct assignment to loss weight |
| `deep_supervision_weights` | Weights for deep supervision | Decreasing sequence (e.g., [1.0, 0.8, 0.6, 0.4]) | Direct assignment to weight list |
| `loss_reduction` | How to reduce loss values | "mean", "sum", "none" | `loss_fn.reduction = new_reduction` |

## Data Augmentation Parameters

These parameters control data augmentation and preprocessing, and **can be modified between epochs**:

| Parameter | Description | Typical Values | Notes |
|-----------|-------------|----------------|-------|
| `intensity_transform_prob` | Probability of intensity transforms | 0.0-1.0 | Controls brightness/contrast augmentation frequency |
| `spatial_transform_prob` | Probability of spatial transforms | 0.0-1.0 | Controls rotation/scaling augmentation frequency |
| `random_crop_size` | Size of random crops | (64,64,64) to (256,256,256) | Trade-off between context and variety |
| `rotation_range` | Range of random rotations | (-30,30) to (-180,180) degrees | Controls rotation augmentation strength |
| `scale_range` | Range of random scaling | (0.8,1.2) to (0.5,1.5) | Controls scale augmentation strength |
| `flip_prob` | Probability of random flips | 0.0-0.5 | Controls flip augmentation frequency |
| `elastic_transform_sigma` | Elastic deformation sigma | 5.0-15.0 | Controls elastic deformation strength |
| `intensity_window` | Intensity windowing range | Dataset-specific | Controls contrast in CT/MRI |
| `normalize_mode` | Normalization strategy | "channel", "instance" | Affects input data distribution |
| `resample_spacing` | Target voxel spacing | Dataset-specific | Controls resolution consistency |
| `gaussian_noise_std` | Standard deviation of noise | 0.01-0.1 | Controls noise augmentation strength |
| `gamma_range` | Range for gamma correction | (0.7,1.5) | Controls contrast augmentation strength |

## Recommended Hyperparameter Ranges

Based on empirical results from medical image segmentation tasks:

### 2D UNet

| Hyperparameter | Recommended Range | Comments |
|----------------|-------------------|----------|
| `learning_rate` | 1e-4 to 1e-2 | Start with 1e-3 and adjust |
| `batch_size` | 8 to 64 | Larger is generally better |
| `channels` | (16,32,64,128,256) to (64,128,256,512,1024) | Scale based on dataset complexity |
| `dropout` | 0.0 to 0.5 | Higher for smaller datasets |
| `weight_decay` | 1e-5 to 1e-3 | Higher for smaller datasets |
| `num_res_units` | 0 to 3 | Higher for deeper feature extraction |

### 3D UNet

| Hyperparameter | Recommended Range | Comments |
|----------------|-------------------|----------|
| `learning_rate` | 5e-5 to 5e-3 | Start with 5e-4 and adjust |
| `batch_size` | 1 to 8 | Limited by GPU memory |
| `channels` | (16,32,64,128) to (32,64,128,256) | Smaller than 2D due to memory constraints |
| `patch_size` | (64,64,64) to (128,128,128) | Trade-off between context and memory |
| `dropout` | 0.0 to 0.3 | Lower than 2D due to inherent regularization of 3D |
| `weight_decay` | 1e-5 to 1e-3 | Similar to 2D |
| `num_res_units` | 0 to 2 | Lower than 2D due to memory constraints |

### Dynamic Adaptation Recommendations

For parameters that can be modified during training:

1. **Learning Rate**:
   - Start with a moderate value (1e-3 for 2D, 5e-4 for 3D)
   - Use ReduceLROnPlateau with factor=0.1, patience=10
   - Set minimum LR to 1e-6

2. **Weight Decay**:
   - Start with 1e-4
   - Increase if validation loss shows overfitting
   - Decrease if training is unstable

3. **Dropout**:
   - Start with 0.2 for 2D, 0.1 for 3D
   - Increase if validation loss shows overfitting
   - Can be dynamically adjusted based on validation performance

4. **Loss Weights** (for multi-component losses):
   - Start with equal weights
   - Gradually increase weights for components with higher validation error
   - Implement a schedule that shifts focus from pixel-wise to structural metrics over time
