# Specific Actions to Fix Segmentation Agent Script

## FGBalanceAgent Fixes

### In file: `/Users/anyadecarlo/HYPERA/HYPERA1/segmentation/agents/fg_balance_agent.py`

1. **Fix save method (line 494):**
   ```python
   # Change this line:
   "learning_rate": self.learning_rate,

   # To this:
   "lr": self.lr,
   ```

2. **Fix load method (line 528):**
   ```python
   # Change this line:
   self.learning_rate = state_dict["learning_rate"]

   # To this:
   self.lr = state_dict["lr"]
   ```

3. **Fix get_state method (line 438):**
   ```python
   # Change this line:
   "learning_rate": self.learning_rate,

   # To this:
   "lr": self.lr,
   ```

4. **Fix set_state method (lines 464-468):**
   ```python
   # Change these lines:
   if "learning_rate" in state:
       self.learning_rate = state["learning_rate"]
       # Update optimizer learning rate
       for param_group in self.optimizer.param_groups:
           param_group["lr"] = self.learning_rate

   # To these:
   if "lr" in state:
       self.lr = state["lr"]
       # Update optimizer learning rate
       for param_group in self.optimizer.param_groups:
           param_group["lr"] = self.lr
   ```

## BoundaryAgent State Dimension Fix

### In file: `/Users/anyadecarlo/HYPERA/HYPERA1/segmentation/agents/boundary_agent.py`

1. **Ensure state dimension is consistent (line 572-648):**
   ```python
   # Verify that get_state_representation method properly handles state_dim=10
   # Make sure the following is correct:
   state = np.zeros((combined_input.size(0), self.state_dim), dtype=np.float32)
   
   # And these lines properly fill positions 0-7 with features and 8-9 with metrics:
   feature_size = min(boundary_features.shape[1], self.state_dim - 2)
   state[:, :feature_size] = boundary_features[:, :feature_size]
   
   # Add scalar metrics to the state
   if "hausdorff_distance" in features:
       state[:, -2] = features["hausdorff_distance"]
   if "boundary_dice" in features:
       state[:, -1] = features["boundary_dice"]
   ```

## RegionAgent Fixes

### In file: `/Users/anyadecarlo/HYPERA/HYPERA1/segmentation/agents/region_agent.py`

1. **Check for any learning_rate references in save/load methods**
   ```python
   # If any references to learning_rate exist, change them to lr
   ```

## ShapeAgent Fixes

### In file: `/Users/anyadecarlo/HYPERA/HYPERA1/segmentation/agents/shape_agent.py`

1. **Check for any learning_rate references in save/load methods**
   ```python
   # If any references to learning_rate exist, change them to lr
   ```

## ObjectDetectionAgent Fixes

### In file: `/Users/anyadecarlo/HYPERA/HYPERA1/segmentation/agents/object_detection_agent.py`

1. **Check for any learning_rate references in save/load methods**
   ```python
   # If any references to learning_rate exist, change them to lr
   ```

## SegmentationAgentCoordinator Fixes

### In file: `/Users/anyadecarlo/HYPERA/HYPERA1/segmentation/agents/segmentation_agent_coordinator.py`

1. **Ensure state reshaping is consistent (check method that handles agent states):**
   ```python
   # Verify that agent state representations are properly reshaped before passing to SAC
   # Make sure all states have batch dimension [batch_size, state_dim]
   ```

2. **Check agent saving logic (around line 454):**
   ```python
   # Make sure the save method properly handles all agent types
   # Verify this section:
   agent.save(agent_path)
   ```

## Training Script Fixes

### In file: `/Users/anyadecarlo/HYPERA/HYPERA1/train_segmentation_agents.py`

1. **Add error handling for NaN Dice scores:**
   ```python
   # Add this code where Dice scores are calculated:
   if np.isnan(dice_score):
       # Handle NaN values - replace with small value or previous value
       dice_score = previous_dice_score if 'previous_dice_score' in locals() else 0.01
   previous_dice_score = dice_score
   ```

2. **Add gradient clipping to prevent exploding gradients:**
   ```python
   # Add this code in the training loop where optimizer steps are taken:
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. **Add early stopping based on validation performance:**
   ```python
   # Add this code after validation evaluation:
   if val_dice_refined > best_val_dice:
       best_val_dice = val_dice_refined
       patience_counter = 0
       # Save best model
       agent_coordinator.save(os.path.join(output_dir, "models", "best_agent_coordinator.pt"))
   else:
       patience_counter += 1
       if patience_counter >= patience_limit:
           logger.info(f"Early stopping triggered after {epoch+1} epochs")
           break
   ```

## Configuration File Updates

### In relevant config files:

1. **Update parameter names for consistency:**
   ```python
   # Change any instances of:
   "learning_rate": 0.001,
   
   # To:
   "lr": 0.001,
   ```

2. **Add parameter conversion in factory methods:**
   ```python
   # Ensure all create_*_agent methods have this conversion:
   if "learning_rate" in config:
       config["lr"] = config.pop("learning_rate")
   ```

## Debugging Additions

1. **Add detailed logging for state dimensions:**
   ```python
   # Add this code in get_state_representation methods:
   logger.debug(f"State shape before reshaping: {state.shape}")
   # After reshaping
   logger.debug(f"State shape after reshaping: {state.shape}, expected: [batch_size, {self.state_dim}]")
   ```

2. **Add NaN checking for critical values:**
   ```python
   # Add this code where important metrics are calculated:
   if torch.isnan(metric).any():
       logger.warning(f"NaN detected in {metric_name}. Using fallback value.")
       metric = torch.where(torch.isnan(metric), torch.tensor(0.01, device=metric.device), metric)
   ```

3. **Add tensor device checking:**
   ```python
   # Add this code where tensors from different sources are combined:
   logger.debug(f"Tensor A device: {tensor_a.device}, Tensor B device: {tensor_b.device}")
   # Ensure tensors are on same device
   tensor_b = tensor_b.to(tensor_a.device)
   ```
