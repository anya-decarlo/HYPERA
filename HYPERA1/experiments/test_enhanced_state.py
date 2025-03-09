#!/usr/bin/env python
# Test script for enhanced state representation

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from HYPERA1.0.agents.shared_state import SharedStateManager
from HYPERA1.0.agents.agent_factory import AgentFactory

def generate_synthetic_metrics(epochs: int = 50) -> Dict[str, List[float]]:
    """
    Generate synthetic training metrics for testing.
    
    Args:
        epochs: Number of epochs to simulate
        
    Returns:
        Dictionary of synthetic metrics
    """
    # Initialize metrics
    metrics = {
        "loss": [],
        "val_loss": [],
        "dice_score": [],
        "val_dice_score": [],
        "dice_class_0": [],
        "dice_class_1": [],
        "dice_class_2": []
    }
    
    # Generate synthetic metrics with realistic patterns
    for epoch in range(epochs):
        # Training loss decreases with noise and plateaus
        train_loss = 1.0 / (1 + 0.1 * epoch) + 0.1 * np.random.randn()
        train_loss = max(0.1, train_loss)
        
        # Validation loss follows training loss but with overfitting after some point
        if epoch < epochs // 2:
            val_loss = train_loss + 0.05 + 0.15 * np.random.randn()
        else:
            # Introduce overfitting
            val_loss = train_loss + 0.05 + 0.05 * epoch / epochs + 0.15 * np.random.randn()
        val_loss = max(0.1, val_loss)
        
        # Dice score increases with noise and plateaus
        dice = 0.5 + 0.4 * (1 - np.exp(-0.1 * epoch)) + 0.05 * np.random.randn()
        dice = min(0.95, max(0.1, dice))
        
        # Validation dice follows training dice but with overfitting
        if epoch < epochs // 2:
            val_dice = dice - 0.05 + 0.1 * np.random.randn()
        else:
            # Introduce overfitting
            val_dice = dice - 0.05 - 0.05 * epoch / epochs + 0.1 * np.random.randn()
        val_dice = min(0.95, max(0.1, val_dice))
        
        # Class-specific dice scores
        dice_class_0 = dice + 0.1 + 0.05 * np.random.randn()  # Background class (easier)
        dice_class_1 = dice - 0.1 + 0.05 * np.random.randn()  # First structure
        dice_class_2 = dice - 0.2 + 0.05 * np.random.randn()  # Second structure (harder)
        
        # Clip values
        dice_class_0 = min(0.98, max(0.1, dice_class_0))
        dice_class_1 = min(0.95, max(0.05, dice_class_1))
        dice_class_2 = min(0.9, max(0.01, dice_class_2))
        
        # Store metrics
        metrics["loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["dice_score"].append(dice)
        metrics["val_dice_score"].append(val_dice)
        metrics["dice_class_0"].append(dice_class_0)
        metrics["dice_class_1"].append(dice_class_1)
        metrics["dice_class_2"].append(dice_class_2)
    
    return metrics

def test_enhanced_state():
    """
    Test the enhanced state representation with synthetic metrics.
    """
    # Create shared state manager with enhanced metrics enabled
    state_manager = SharedStateManager(
        history_size=100,
        verbose=True,
        total_epochs=50,
        enable_enhanced_metrics=True,
        short_window=5,
        medium_window=20,
        long_window=50
    )
    
    # Generate synthetic metrics
    metrics = generate_synthetic_metrics(epochs=50)
    
    # Record metrics in shared state manager
    for epoch in range(len(metrics["loss"])):
        epoch_metrics = {key: values[epoch] for key, values in metrics.items()}
        state_manager.record_metrics(epoch, epoch_metrics)
    
    # Create agents with enhanced state
    agent_factory = AgentFactory()
    
    # Create learning rate agent
    lr_agent = agent_factory.create_agent(
        agent_type="learning_rate",
        shared_state_manager=state_manager,
        use_enhanced_state=True,
        state_dim=20
    )
    
    # Create weight decay agent
    wd_agent = agent_factory.create_agent(
        agent_type="weight_decay",
        shared_state_manager=state_manager,
        use_enhanced_state=True,
        state_dim=20
    )
    
    # Create class weights agent
    cw_agent = agent_factory.create_agent(
        agent_type="class_weights",
        shared_state_manager=state_manager,
        use_enhanced_state=True,
        state_dim=20
    )
    
    # Get state representations
    lr_state = lr_agent.get_state_representation()
    wd_state = wd_agent.get_state_representation()
    cw_state = cw_agent.get_state_representation()
    
    # Get enhanced metrics
    enhanced_features = state_manager.get_enhanced_state_features(["loss", "val_loss", "dice_score"])
    overfitting_signals = state_manager.get_overfitting_signals()
    
    # Print results
    print("\n=== Enhanced State Representation Test ===")
    print(f"Learning Rate Agent State Shape: {lr_state.shape}")
    print(f"Weight Decay Agent State Shape: {wd_state.shape}")
    print(f"Class Weights Agent State Shape: {cw_state.shape}")
    
    print("\n=== Enhanced Metrics ===")
    for metric, value in enhanced_features.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n=== Overfitting Signals ===")
    for signal, value in overfitting_signals.items():
        print(f"{signal}: {value:.4f}")
    
    # Plot metrics and enhanced features
    plt.figure(figsize=(15, 10))
    
    # Plot original metrics
    plt.subplot(2, 1, 1)
    plt.plot(metrics["loss"], label="Training Loss")
    plt.plot(metrics["val_loss"], label="Validation Loss")
    plt.plot(metrics["dice_score"], label="Dice Score")
    plt.plot(metrics["val_dice_score"], label="Validation Dice")
    plt.title("Original Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    # Plot enhanced features
    plt.subplot(2, 1, 2)
    
    # Extract enhanced features for each epoch
    enhanced_history = {
        "loss_short_trend": [],
        "loss_medium_trend": [],
        "loss_long_trend": [],
        "val_loss_short_trend": [],
        "val_loss_medium_trend": [],
        "val_loss_long_trend": [],
        "overfitting_signal": []
    }
    
    # Recreate state manager and record metrics again to collect history
    state_manager = SharedStateManager(
        history_size=100,
        verbose=False,
        total_epochs=50,
        enable_enhanced_metrics=True,
        short_window=5,
        medium_window=20,
        long_window=50
    )
    
    for epoch in range(len(metrics["loss"])):
        epoch_metrics = {key: values[epoch] for key, values in metrics.items()}
        state_manager.record_metrics(epoch, epoch_metrics)
        
        if epoch >= 5:  # Need some history for trends
            features = state_manager.get_enhanced_state_features(["loss", "val_loss"])
            signals = state_manager.get_overfitting_signals()
            
            enhanced_history["loss_short_trend"].append(features.get("loss_short_trend", 0))
            enhanced_history["loss_medium_trend"].append(features.get("loss_medium_trend", 0))
            enhanced_history["loss_long_trend"].append(features.get("loss_long_trend", 0))
            enhanced_history["val_loss_short_trend"].append(features.get("val_loss_short_trend", 0))
            enhanced_history["val_loss_medium_trend"].append(features.get("val_loss_medium_trend", 0))
            enhanced_history["val_loss_long_trend"].append(features.get("val_loss_long_trend", 0))
            enhanced_history["overfitting_signal"].append(signals.get("overfitting_signal", 0))
    
    # Plot enhanced features
    epochs_range = list(range(5, len(metrics["loss"])))
    plt.plot(epochs_range, enhanced_history["loss_short_trend"], label="Loss Short Trend")
    plt.plot(epochs_range, enhanced_history["loss_medium_trend"], label="Loss Medium Trend")
    plt.plot(epochs_range, enhanced_history["val_loss_short_trend"], label="Val Loss Short Trend")
    plt.plot(epochs_range, enhanced_history["overfitting_signal"], label="Overfitting Signal", linewidth=2)
    
    plt.title("Enhanced Features")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("enhanced_state_test.png")
    plt.show()

if __name__ == "__main__":
    test_enhanced_state()



