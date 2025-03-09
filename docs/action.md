# Specific Actions to Improve Dice Score in HYPERA Agents

## Code Fixes

1. **Fix parameter naming inconsistencies in FGBalanceAgent**:
   - Change `"learning_rate": self.learning_rate` to `"lr": self.lr` in `save()` method (line 494)
   - Change `self.learning_rate = state_dict["learning_rate"]` to `self.lr = state_dict["lr"]` in `load()` method (line 528)
   - Update `param_group["lr"] = self.learning_rate` to `param_group["lr"] = self.lr` (line 468)

## Hyperparameter Optimizations

1. **Increase learning rate for BoundaryAgent**:
   - Change `lr` from 3e-4 to 5e-4 in `create_boundary_agent()` method
   - This will help the agent learn boundary features faster

2. **Adjust SAC temperature parameter**:
   - Change `alpha` from 0.2 to 0.1 for RegionAgent
   - Lower temperature makes the policy more deterministic, focusing on high-reward actions

3. **Modify replay buffer size**:
   - Increase `replay_buffer_size` from 10000 to 20000 for all agents
   - Larger buffer retains more diverse experiences for better generalization

4. **Update batch size for training**:
   - Change `batch_size` from 64 to 128 for RegionAgent and BoundaryAgent
   - Larger batches provide more stable gradient estimates

5. **Adjust discount factor**:
   - Change `gamma` from 0.99 to 0.95 for all agents
   - Lower discount factor emphasizes immediate rewards (Dice improvement)

## Feature Extraction Improvements

1. **Increase feature extraction depth**:
   - Change `feature_channels` from 32 to 64 in BoundaryAgent and RegionAgent
   - More channels capture finer details in boundaries and regions

2. **Add batch normalization**:
   - Insert `nn.BatchNorm2d(out_channels)` after each convolutional layer in feature extractors
   - Stabilizes training and allows higher learning rates

3. **Implement feature fusion**:
   - Modify `_extract_features()` to combine low-level and high-level features
   - Creates multi-scale awareness of both local details and global context

## Reward Function Adjustments

1. **Increase weight for boundary accuracy**:
   - Change boundary component weight from 0.2 to 0.4 in `MultiObjectiveRewardCalculator`
   - Emphasizes precise boundary delineation

2. **Implement reward normalization**:
   - Add z-score normalization for rewards in `RewardStatisticsTracker`
   - Prevents large rewards from dominating training

3. **Add reward shaping**:
   - Implement incremental rewards for partial improvements in Dice score
   - Provides better learning signal for small improvements

## Agent Coordination Improvements

1. **Modify agent weighting strategy**:
   - Change `weighted_average` strategy to prioritize RegionAgent (weight 0.5) and BoundaryAgent (weight 0.3)
   - These agents have most direct impact on Dice score

2. **Implement adaptive coordination**:
   - Add dynamic weight adjustment based on current Dice performance
   - Increases influence of better-performing agents during training
