#!/usr/bin/env python
# Multi-Objective Reward Calculator for Segmentation Agents

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from monai.metrics import compute_hausdorff_distance, compute_dice

from .adaptive_weight_manager import AdaptiveWeightManager
from .reward_statistics import RewardStatisticsTracker

class MultiObjectiveRewardCalculator:
    """
    Calculates multi-objective rewards for segmentation agents.
    
    This class implements the comprehensive reward function with components for:
    1. Regional Overlap (Dice Score)
    2. Boundary Accuracy (Hausdorff Distance)
    3. Precision-Recall Balance (F1-Score for Cell Detection)
    4. Compactness & Shape Regularization
    5. Foreground-Background Balance
    
    Attributes:
        device (torch.device): Device to use for computation
        weights (dict): Weights for each reward component
        training_phase (str): Current training phase (exploration, exploitation, fine-tuning)
        adaptive_weight_manager (AdaptiveWeightManager): Manager for adaptive weights
        reward_stats_tracker (RewardStatisticsTracker): Tracker for reward statistics
        use_adaptive_weights (bool): Whether to use adaptive weights
        use_reward_normalization (bool): Whether to normalize rewards
        verbose (bool): Whether to print verbose output
    """
    
    def __init__(
        self,
        device: torch.device = None,
        initial_weights: Dict[str, float] = None,
        use_adaptive_weights: bool = True,
        use_reward_normalization: bool = True,
        reward_window_size: int = 100,
        reward_clip_range: Tuple[float, float] = (-10.0, 10.0),
        max_epochs: int = 100,
        exploration_ratio: float = 0.3,
        exploitation_ratio: float = 0.5,
        verbose: bool = False
    ):
        """
        Initialize the multi-objective reward calculator.
        
        Args:
            device: Device to use for computation
            initial_weights: Initial weights for each reward component
            use_adaptive_weights: Whether to use adaptive weights
            use_reward_normalization: Whether to normalize rewards
            reward_window_size: Size of the sliding window for reward statistics
            reward_clip_range: Range for clipping rewards
            max_epochs: Maximum number of epochs for training
            exploration_ratio: Ratio of epochs for exploration phase
            exploitation_ratio: Ratio of epochs for exploitation phase
            verbose: Whether to print verbose output
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_adaptive_weights = use_adaptive_weights
        self.use_reward_normalization = use_reward_normalization
        self.verbose = verbose
        
        # Set default weights if not provided
        if initial_weights is None:
            self.weights = {
                "dice": 1.0,
                "boundary": 0.5,
                "object_f1": 0.8,
                "shape": 0.3,
                "fg_balance": 0.4
            }
        else:
            self.weights = initial_weights
        
        # Initialize training phase
        self.training_phase = "exploration"
        
        # Initialize adaptive weight manager
        self.adaptive_weight_manager = AdaptiveWeightManager(
            initial_weights=self.weights,
            max_epochs=max_epochs,
            exploration_ratio=exploration_ratio,
            exploitation_ratio=exploitation_ratio,
            phase_detection_enabled=use_adaptive_weights,
            verbose=verbose
        )
        
        # Initialize reward statistics tracker
        self.reward_stats_tracker = RewardStatisticsTracker(
            window_size=reward_window_size,
            reward_clip_range=reward_clip_range,
            use_z_score_normalization=use_reward_normalization,
            verbose=verbose
        )
        
        # Set up logging
        self.logger = logging.getLogger("MultiObjectiveReward")
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        if self.verbose:
            self.logger.info("Initialized Multi-Objective Reward Calculator")
            self.logger.info(f"Initial weights: {self.weights}")
            self.logger.info(f"Using adaptive weights: {use_adaptive_weights}")
            self.logger.info(f"Using reward normalization: {use_reward_normalization}")
    
    def update_training_phase(self, phase: str):
        """
        Update the training phase and adjust weights if adaptive.
        
        Args:
            phase: New training phase (exploration, exploitation, fine-tuning)
        """
        self.training_phase = phase
        
        if self.use_adaptive_weights:
            # Update phase in adaptive weight manager
            self.adaptive_weight_manager.set_phase(phase)
            
            # Get updated weights
            self.weights = self.adaptive_weight_manager.get_current_weights()
            
            if self.verbose:
                self.logger.info(f"Updated training phase to {phase}")
                self.logger.info(f"Updated weights: {self.weights}")
    
    def update_weights_from_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Update weights based on current epoch and metrics.
        
        Args:
            epoch: Current epoch
            metrics: Dictionary of current metrics
            
        Returns:
            Updated weights
        """
        if self.use_adaptive_weights:
            # Update weights in adaptive weight manager
            self.weights = self.adaptive_weight_manager.update_weights(epoch, metrics)
            
            if self.verbose:
                self.logger.info(f"Updated weights based on metrics at epoch {epoch}: {self.weights}")
        
        return self.weights
    
    def calculate_dice_reward(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> float:
        """
        Calculate the Dice score reward component.
        
        R_{\text{Dice}} = \frac{2 \times |P \cap G|}{|P| + |G|}
        
        Args:
            prediction: Predicted segmentation mask
            ground_truth: Ground truth segmentation mask
            
        Returns:
            Dice score reward
        """
        # Ensure tensors are on the correct device
        prediction = prediction.to(self.device)
        ground_truth = ground_truth.to(self.device)
        
        # Calculate Dice score
        dice_score = compute_dice(prediction, ground_truth)
        
        # Convert to scalar if it's a tensor
        if isinstance(dice_score, torch.Tensor):
            dice_score = dice_score.item()
        
        return dice_score
    
    def calculate_boundary_reward(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> float:
        """
        Calculate the boundary accuracy reward component using Hausdorff distance.
        
        R_{\text{Boundary}} = -\text{Hausdorff}(P, G)
        
        Args:
            prediction: Predicted segmentation mask
            ground_truth: Ground truth segmentation mask
            
        Returns:
            Boundary accuracy reward (negative Hausdorff distance)
        """
        # Ensure tensors are on the correct device
        prediction = prediction.to(self.device)
        ground_truth = ground_truth.to(self.device)
        
        # Calculate Hausdorff distance
        hausdorff_dist = compute_hausdorff_distance(prediction, ground_truth)
        
        # Convert to scalar if it's a tensor
        if isinstance(hausdorff_dist, torch.Tensor):
            hausdorff_dist = hausdorff_dist.item()
        
        # Return negative distance as reward (lower distance is better)
        return -hausdorff_dist
    
    def calculate_detailed_boundary_metrics(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate detailed boundary metrics for specialized boundary agents.
        
        This method provides a comprehensive set of boundary-related metrics:
        1. Hausdorff Distance (HD): Maximum distance from a point in one boundary to the closest point in the other
        2. Average Surface Distance (ASD): Average distance between boundaries
        3. Boundary Dice Coefficient: Dice score calculated only on boundary pixels
        4. Boundary Precision: Precision of boundary pixels
        5. Boundary Recall: Recall of boundary pixels
        
        Args:
            prediction: Predicted segmentation mask
            ground_truth: Ground truth segmentation mask
            
        Returns:
            Dictionary of boundary metrics
        """
        # Ensure tensors are on the correct device
        prediction = prediction.to(self.device)
        ground_truth = ground_truth.to(self.device)
        
        # Convert to binary masks
        pred_binary = (prediction > 0.5).float()
        gt_binary = (ground_truth > 0.5).float()
        
        # Extract boundaries using gradient operators
        # Sobel operators for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(self.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(self.device)
        
        # Reshape for conv2d
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        # Apply Sobel operators
        pred_grad_x = F.conv2d(pred_binary.unsqueeze(1), sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_binary.unsqueeze(1), sobel_y, padding=1)
        gt_grad_x = F.conv2d(gt_binary.unsqueeze(1), sobel_x, padding=1)
        gt_grad_y = F.conv2d(gt_binary.unsqueeze(1), sobel_y, padding=1)
        
        # Calculate gradient magnitude
        pred_boundary = torch.sqrt(pred_grad_x**2 + pred_grad_y**2).squeeze(1)
        gt_boundary = torch.sqrt(gt_grad_x**2 + gt_grad_y**2).squeeze(1)
        
        # Threshold to get binary boundaries
        pred_boundary = (pred_boundary > 0.1).float()
        gt_boundary = (gt_boundary > 0.1).float()
        
        # Calculate Hausdorff distance
        hausdorff_dist = compute_hausdorff_distance(pred_boundary.unsqueeze(0), gt_boundary.unsqueeze(0))
        if isinstance(hausdorff_dist, torch.Tensor):
            hausdorff_dist = hausdorff_dist.item()
        
        # Calculate Average Surface Distance (ASD)
        # Convert to numpy for distance transform
        pred_boundary_np = pred_boundary.cpu().numpy()
        gt_boundary_np = gt_boundary.cpu().numpy()
        
        # Distance transforms
        pred_to_gt_dist = distance_transform_edt(1 - gt_boundary_np)
        gt_to_pred_dist = distance_transform_edt(1 - pred_boundary_np)
        
        # Calculate ASD
        pred_boundary_points = np.sum(pred_boundary_np)
        gt_boundary_points = np.sum(gt_boundary_np)
        
        if pred_boundary_points > 0 and gt_boundary_points > 0:
            # Average distance from prediction to ground truth
            avg_pred_to_gt = np.sum(pred_boundary_np * pred_to_gt_dist) / pred_boundary_points
            # Average distance from ground truth to prediction
            avg_gt_to_pred = np.sum(gt_boundary_np * gt_to_pred_dist) / gt_boundary_points
            # Symmetric average surface distance
            asd = (avg_pred_to_gt + avg_gt_to_pred) / 2
        else:
            asd = 100.0  # Large value if no boundary points
        
        # Calculate Boundary Dice
        intersection = torch.sum(pred_boundary * gt_boundary)
        pred_sum = torch.sum(pred_boundary)
        gt_sum = torch.sum(gt_boundary)
        
        if pred_sum + gt_sum > 0:
            boundary_dice = (2.0 * intersection) / (pred_sum + gt_sum)
        else:
            boundary_dice = 0.0
        
        # Calculate Boundary Precision and Recall
        if pred_sum > 0:
            boundary_precision = intersection / pred_sum
        else:
            boundary_precision = 0.0
        
        if gt_sum > 0:
            boundary_recall = intersection / gt_sum
        else:
            boundary_recall = 0.0
        
        # Convert tensor values to scalars
        if isinstance(boundary_dice, torch.Tensor):
            boundary_dice = boundary_dice.item()
        if isinstance(boundary_precision, torch.Tensor):
            boundary_precision = boundary_precision.item()
        if isinstance(boundary_recall, torch.Tensor):
            boundary_recall = boundary_recall.item()
        
        # Create metrics dictionary
        boundary_metrics = {
            "hausdorff_distance": hausdorff_dist,
            "average_surface_distance": asd,
            "boundary_dice": boundary_dice,
            "boundary_precision": boundary_precision,
            "boundary_recall": boundary_recall,
            "boundary_f1": 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall + 1e-6)
        }
        
        return boundary_metrics
    
    def calculate_object_f1_reward(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> float:
        """
        Calculate the object-level F1 score reward component.
        
        R_{\text{ObjF1}} = 2 \times \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}}
        
        Args:
            prediction: Predicted segmentation mask
            ground_truth: Ground truth segmentation mask
            
        Returns:
            Object-level F1 score reward
        """
        # Convert tensors to numpy arrays
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.detach().cpu().numpy()
        
        # Label connected components in prediction and ground truth
        pred_labels = label(prediction > 0.5)
        gt_labels = label(ground_truth > 0.5)
        
        # Get region properties
        pred_regions = regionprops(pred_labels)
        gt_regions = regionprops(gt_labels)
        
        # Count objects
        n_pred = len(pred_regions)
        n_gt = len(gt_regions)
        
        if n_gt == 0:
            if n_pred == 0:
                # No objects in either prediction or ground truth
                return 1.0
            else:
                # False positives only
                return 0.0
        
        if n_pred == 0:
            # False negatives only
            return 0.0
        
        # Calculate IoU for each pair of predicted and ground truth objects
        matches = 0
        iou_threshold = 0.5  # IoU threshold for considering a match
        
        for gt_region in gt_regions:
            gt_mask = gt_labels == gt_region.label
            best_iou = 0
            
            for pred_region in pred_regions:
                pred_mask = pred_labels == pred_region.label
                
                # Calculate intersection and union
                intersection = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()
                
                if union > 0:
                    iou = intersection / union
                    best_iou = max(best_iou, iou)
            
            if best_iou >= iou_threshold:
                matches += 1
        
        # Calculate precision and recall
        precision = matches / n_pred if n_pred > 0 else 0
        recall = matches / n_gt if n_gt > 0 else 0
        
        # Calculate F1 score
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0
        
        return f1_score
    
    def calculate_detailed_object_metrics(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate detailed object detection metrics for specialized object detection agents.
        
        This method provides a comprehensive set of object-level metrics:
        1. Object Count Accuracy: Ratio of predicted objects to ground truth objects
        2. Object Precision: Precision of object detection
        3. Object Recall: Recall of object detection
        4. Object F1 Score: F1 score for object detection
        5. Mean IoU: Mean IoU of matched objects
        6. Size Distribution Error: Error in the distribution of object sizes
        
        Args:
            prediction: Predicted segmentation mask
            ground_truth: Ground truth segmentation mask
            
        Returns:
            Dictionary of object detection metrics
        """
        # Convert tensors to numpy arrays
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.detach().cpu().numpy()
        
        # Label connected components in prediction and ground truth
        pred_labels = label(prediction > 0.5)
        gt_labels = label(ground_truth > 0.5)
        
        # Get region properties
        pred_regions = regionprops(pred_labels)
        gt_regions = regionprops(gt_labels)
        
        # Count objects
        n_pred = len(pred_regions)
        n_gt = len(gt_regions)
        
        # Initialize metrics
        object_metrics = {
            "object_count_accuracy": 0.0,
            "object_precision": 0.0,
            "object_recall": 0.0,
            "object_f1": 0.0,
            "mean_iou": 0.0,
            "size_distribution_error": 0.0
        }
        
        # Handle edge cases
        if n_gt == 0:
            if n_pred == 0:
                # No objects in either prediction or ground truth
                object_metrics["object_count_accuracy"] = 1.0
                object_metrics["object_precision"] = 1.0
                object_metrics["object_recall"] = 1.0
                object_metrics["object_f1"] = 1.0
                return object_metrics
            else:
                # False positives only
                object_metrics["object_count_accuracy"] = 0.0
                return object_metrics
        
        if n_pred == 0:
            # False negatives only
            object_metrics["object_count_accuracy"] = 0.0
            return object_metrics
        
        # Calculate object count accuracy (1 - absolute error ratio)
        count_diff = abs(n_pred - n_gt)
        object_metrics["object_count_accuracy"] = max(0, 1 - (count_diff / max(n_pred, n_gt)))
        
        # Calculate IoU for each pair of predicted and ground truth objects
        matches = 0
        matched_ious = []
        iou_threshold = 0.5  # IoU threshold for considering a match
        
        # Track matched objects to avoid double counting
        matched_gt_indices = set()
        matched_pred_indices = set()
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((n_gt, n_pred))
        
        for i, gt_region in enumerate(gt_regions):
            gt_mask = gt_labels == gt_region.label
            
            for j, pred_region in enumerate(pred_regions):
                pred_mask = pred_labels == pred_region.label
                
                # Calculate intersection and union
                intersection = np.logical_and(gt_mask, pred_mask).sum()
                union = np.logical_or(gt_mask, pred_mask).sum()
                
                if union > 0:
                    iou_matrix[i, j] = intersection / union
        
        # Match objects greedily by IoU
        while True:
            # Find highest IoU
            if len(matched_gt_indices) == n_gt or len(matched_pred_indices) == n_pred:
                break
                
            # Create mask for unmatched objects
            mask = np.ones_like(iou_matrix, dtype=bool)
            for i in matched_gt_indices:
                mask[i, :] = False
            for j in matched_pred_indices:
                mask[:, j] = False
            
            # Apply mask and find highest IoU
            masked_iou = np.where(mask, iou_matrix, 0)
            if np.max(masked_iou) < iou_threshold:
                break
                
            # Get indices of highest IoU
            i, j = np.unravel_index(np.argmax(masked_iou), iou_matrix.shape)
            
            # Add to matches
            matched_gt_indices.add(i)
            matched_pred_indices.add(j)
            matched_ious.append(iou_matrix[i, j])
            matches += 1
        
        # Calculate precision and recall
        object_metrics["object_precision"] = matches / n_pred if n_pred > 0 else 0
        object_metrics["object_recall"] = matches / n_gt if n_gt > 0 else 0
        
        # Calculate F1 score
        if object_metrics["object_precision"] + object_metrics["object_recall"] > 0:
            object_metrics["object_f1"] = (
                2 * object_metrics["object_precision"] * object_metrics["object_recall"] / 
                (object_metrics["object_precision"] + object_metrics["object_recall"])
            )
        
        # Calculate mean IoU of matched objects
        if matched_ious:
            object_metrics["mean_iou"] = np.mean(matched_ious)
        
        # Calculate size distribution error
        if n_gt > 0 and n_pred > 0:
            # Get areas of objects
            gt_areas = np.array([r.area for r in gt_regions])
            pred_areas = np.array([r.area for r in pred_regions])
            
            # Normalize areas
            gt_areas = gt_areas / np.sum(gt_areas)
            pred_areas = pred_areas / np.sum(pred_areas)
            
            # Calculate size distribution error using Earth Mover's Distance
            try:
                # Use SciPy's Wasserstein distance (Earth Mover's Distance)
                from scipy.stats import wasserstein_distance
                
                # Calculate EMD between the two distributions
                emd = wasserstein_distance(gt_areas, pred_areas)
                
                # Normalize to [0, 1] range and convert to similarity score
                # Assuming maximum EMD is 1.0 for normalized distributions
                object_metrics["size_distribution_error"] = 1 - min(1, emd)
            except ImportError:
                # Fall back to simple approximation if SciPy is not available
                # Sort areas for approximate EMD calculation
                gt_areas_sorted = np.sort(gt_areas)
                pred_areas_sorted = np.sort(pred_areas)
                
                # Pad the shorter array
                if n_gt < n_pred:
                    gt_areas_sorted = np.pad(gt_areas_sorted, (0, n_pred - n_gt))
                elif n_pred < n_gt:
                    pred_areas_sorted = np.pad(pred_areas_sorted, (0, n_gt - n_pred))
                
                # Calculate absolute difference
                size_diff = np.abs(gt_areas_sorted - pred_areas_sorted).sum()
                object_metrics["size_distribution_error"] = 1 - min(1, size_diff)
        
        return object_metrics
        
    def calculate_shape_reward(
        self,
        prediction: torch.Tensor
    ) -> float:
        """
        Calculate the shape regularization reward component.
        
        R_{\text{Shape}} = -\sum_{i} \left( \frac{P_i}{4\pi A_i} - 1 \right)^2
        
        Args:
            prediction: Predicted segmentation mask
            
        Returns:
            Shape regularization reward
        """
        # Convert tensor to numpy array
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        
        # Label connected components
        pred_labels = label(prediction > 0.5)
        
        # Get region properties
        regions = regionprops(pred_labels)
        
        if not regions:
            return 0.0
        
        # Calculate circularity for each region
        circularity_errors = []
        
        for region in regions:
            # Calculate perimeter and area
            perimeter = region.perimeter
            area = region.area
            
            if area > 0 and perimeter > 0:
                # Calculate circularity error (0 for perfect circle)
                circularity = (perimeter**2) / (4 * np.pi * area)
                circularity_error = (circularity - 1)**2
                circularity_errors.append(circularity_error)
        
        if not circularity_errors:
            return 0.0
        
        # Calculate average circularity error
        avg_circularity_error = np.mean(circularity_errors)
        
        # Return negative error as reward (lower error is better)
        return -avg_circularity_error
    
    def calculate_fg_balance_reward(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> float:
        """
        Calculate the foreground-background balance reward component.
        
        R_{\text{FG}} = -D_{KL}(P_{\text{FG}} || G_{\text{FG}})
        
        Args:
            prediction: Predicted segmentation mask
            ground_truth: Ground truth segmentation mask
            
        Returns:
            Foreground-background balance reward
        """
        # Ensure tensors are on the correct device
        prediction = prediction.to(self.device)
        ground_truth = ground_truth.to(self.device)
        
        # Calculate foreground ratios
        pred_fg_ratio = torch.mean((prediction > 0.5).float())
        gt_fg_ratio = torch.mean((ground_truth > 0.5).float())
        
        # Calculate absolute difference in foreground ratios
        fg_diff = torch.abs(pred_fg_ratio - gt_fg_ratio)
        
        # Convert to scalar if it's a tensor
        if isinstance(fg_diff, torch.Tensor):
            fg_diff = fg_diff.item()
        
        # Return negative difference as reward (lower difference is better)
        return -fg_diff
    
    def calculate_detailed_fg_balance_metrics(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate detailed foreground-background balance metrics.
        
        This method provides a comprehensive set of metrics related to foreground-background balance:
        1. Foreground Ratio Difference: Absolute difference between predicted and ground truth foreground ratios
        2. KL Divergence: Kullback-Leibler divergence between predicted and ground truth class distributions
        3. Class-wise Precision: Precision for foreground and background classes
        4. Class-wise Recall: Recall for foreground and background classes
        5. Class-wise F1 Score: F1 score for foreground and background classes
        
        Args:
            prediction: Predicted segmentation mask
            ground_truth: Ground truth segmentation mask
            
        Returns:
            Dictionary of foreground-background balance metrics
        """
        # Ensure tensors are on the correct device
        prediction = prediction.to(self.device)
        ground_truth = ground_truth.to(self.device)
        
        # Calculate foreground ratios
        pred_fg_ratio = torch.mean((prediction > 0.5).float()).item()
        gt_fg_ratio = torch.mean((ground_truth > 0.5).float()).item()
        
        # Calculate background ratios
        pred_bg_ratio = 1.0 - pred_fg_ratio
        gt_bg_ratio = 1.0 - gt_fg_ratio
        
        # Calculate absolute difference in foreground ratios
        fg_ratio_diff = abs(pred_fg_ratio - gt_fg_ratio)
        
        # Calculate KL divergence between predicted and ground truth class distributions
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        pred_distribution = torch.tensor([pred_bg_ratio + epsilon, pred_fg_ratio + epsilon], device=self.device)
        gt_distribution = torch.tensor([gt_bg_ratio + epsilon, gt_fg_ratio + epsilon], device=self.device)
        
        # Normalize distributions to sum to 1
        pred_distribution = pred_distribution / torch.sum(pred_distribution)
        gt_distribution = gt_distribution / torch.sum(gt_distribution)
        
        # Calculate KL divergence: KL(gt || pred)
        kl_divergence = torch.sum(gt_distribution * torch.log(gt_distribution / pred_distribution)).item()
        
        # Calculate class-wise metrics
        pred_binary = (prediction > 0.5).float()
        gt_binary = (ground_truth > 0.5).float()
        
        # True positives, false positives, false negatives for foreground
        tp_fg = torch.sum(pred_binary * gt_binary).item()
        fp_fg = torch.sum(pred_binary * (1 - gt_binary)).item()
        fn_fg = torch.sum((1 - pred_binary) * gt_binary).item()
        tn_fg = torch.sum((1 - pred_binary) * (1 - gt_binary)).item()
        
        # Calculate precision, recall, and F1 score for foreground
        fg_precision = tp_fg / (tp_fg + fp_fg) if (tp_fg + fp_fg) > 0 else 0.0
        fg_recall = tp_fg / (tp_fg + fn_fg) if (tp_fg + fn_fg) > 0 else 0.0
        fg_f1 = 2 * fg_precision * fg_recall / (fg_precision + fg_recall) if (fg_precision + fg_recall) > 0 else 0.0
        
        # True positives, false positives, false negatives for background (inverse of foreground)
        tp_bg = tn_fg
        fp_bg = fn_fg
        fn_bg = fp_fg
        tn_bg = tp_fg
        
        # Calculate precision, recall, and F1 score for background
        bg_precision = tp_bg / (tp_bg + fp_bg) if (tp_bg + fp_bg) > 0 else 0.0
        bg_recall = tp_bg / (tp_bg + fn_bg) if (tp_bg + fn_bg) > 0 else 0.0
        bg_f1 = 2 * bg_precision * bg_recall / (bg_precision + bg_recall) if (bg_precision + bg_recall) > 0 else 0.0
        
        # Calculate class imbalance metrics
        fg_pixels = torch.sum(gt_binary).item()
        bg_pixels = torch.sum(1 - gt_binary).item()
        total_pixels = fg_pixels + bg_pixels
        
        fg_weight = bg_pixels / total_pixels if total_pixels > 0 else 0.5
        bg_weight = fg_pixels / total_pixels if total_pixels > 0 else 0.5
        
        # Calculate weighted average F1 score
        weighted_f1 = fg_weight * fg_f1 + bg_weight * bg_f1
        
        # Create metrics dictionary
        fg_balance_metrics = {
            "fg_ratio_diff": fg_ratio_diff,
            "kl_divergence": kl_divergence,
            "pred_fg_ratio": pred_fg_ratio,
            "gt_fg_ratio": gt_fg_ratio,
            "fg_precision": fg_precision,
            "fg_recall": fg_recall,
            "fg_f1": fg_f1,
            "bg_precision": bg_precision,
            "bg_recall": bg_recall,
            "bg_f1": bg_f1,
            "weighted_f1": weighted_f1,
            "fg_weight": fg_weight,
            "bg_weight": bg_weight
        }
        
        return fg_balance_metrics
    
    def calculate_reward(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        include_detailed_metrics: bool = False
    ) -> Dict[str, float]:
        """
        Calculate the multi-objective reward.
        
        Args:
            prediction: Predicted segmentation mask
            ground_truth: Ground truth segmentation mask
            include_detailed_metrics: Whether to include detailed metrics for each component
            
        Returns:
            Dictionary with total reward and individual components
        """
        # Calculate individual reward components
        dice_reward = self.calculate_dice_reward(prediction, ground_truth)
        boundary_reward = self.calculate_boundary_reward(prediction, ground_truth)
        object_f1_reward = self.calculate_object_f1_reward(prediction, ground_truth)
        shape_reward = self.calculate_shape_reward(prediction)
        fg_balance_reward = self.calculate_fg_balance_reward(prediction, ground_truth)
        
        # Create raw reward dictionary
        raw_rewards = {
            "dice": dice_reward,
            "boundary": boundary_reward,
            "object_f1": object_f1_reward,
            "shape": shape_reward,
            "fg_balance": fg_balance_reward
        }
        
        # Apply weights to components
        weighted_rewards = {
            "weighted_dice": self.weights["dice"] * dice_reward,
            "weighted_boundary": self.weights["boundary"] * boundary_reward,
            "weighted_object_f1": self.weights["object_f1"] * object_f1_reward,
            "weighted_shape": self.weights["shape"] * shape_reward,
            "weighted_fg_balance": self.weights["fg_balance"] * fg_balance_reward
        }
        
        # Calculate total reward
        total_reward = sum(weighted_rewards.values())
        
        # Create complete reward dictionary
        reward_dict = {
            "total": total_reward,
            **raw_rewards,
            **weighted_rewards
        }
        
        # Include detailed metrics if requested
        if include_detailed_metrics:
            detailed_metrics = {}
            
            # Add detailed boundary metrics
            detailed_metrics["boundary_details"] = self.calculate_detailed_boundary_metrics(prediction, ground_truth)
            
            # Add detailed foreground-background balance metrics
            detailed_metrics["fg_balance_details"] = self.calculate_detailed_fg_balance_metrics(prediction, ground_truth)
            
            # Add detailed object detection metrics
            detailed_metrics["object_details"] = self.calculate_detailed_object_metrics(prediction, ground_truth)
            
            # Add detailed metrics to reward dictionary
            reward_dict["detailed_metrics"] = detailed_metrics
        
        # Update reward statistics and normalize if enabled
        if self.use_reward_normalization:
            normalized_rewards = self.reward_stats_tracker.update(raw_rewards)
            
            # Add normalized rewards to dictionary
            reward_dict["normalized"] = {
                "dice": normalized_rewards["dice"],
                "boundary": normalized_rewards["boundary"],
                "object_f1": normalized_rewards["object_f1"],
                "shape": normalized_rewards["shape"],
                "fg_balance": normalized_rewards["fg_balance"]
            }
            
            # Calculate normalized total reward
            normalized_total = sum(weighted_reward * normalized_rewards[component.replace("weighted_", "")] 
                                  for component, weighted_reward in weighted_rewards.items())
            
            reward_dict["normalized_total"] = normalized_total
        
        if self.verbose:
            self.logger.debug(f"Reward components: {reward_dict}")
        
        return reward_dict
    
    def get_reward_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each reward component.
        
        Returns:
            Dictionary of component statistics
        """
        return self.reward_stats_tracker.get_component_statistics()
    
    def get_reward_trends(self) -> Dict[str, Dict[str, float]]:
        """
        Get trend information for each reward component.
        
        Returns:
            Dictionary of component trends
        """
        return self.reward_stats_tracker.get_reward_trends()
    
    def get_component_correlations(self) -> Dict[Tuple[str, str], float]:
        """
        Get correlations between reward components.
        
        Returns:
            Dictionary of component correlations
        """
        return self.reward_stats_tracker.get_component_correlations()
    
    def set_reward_window_size(self, window_size: int) -> None:
        """
        Set the window size for reward statistics.
        
        Args:
            window_size: New window size
        """
        self.reward_stats_tracker.set_window_size(window_size)
    
    def set_reward_clip_range(self, clip_range: Tuple[float, float]) -> None:
        """
        Set the range for clipping rewards.
        
        Args:
            clip_range: New clip range
        """
        self.reward_stats_tracker.set_reward_clip_range(clip_range)
    
    def enable_reward_normalization(self) -> None:
        """
        Enable reward normalization.
        """
        self.use_reward_normalization = True
        self.reward_stats_tracker.enable_z_score_normalization()
    
    def disable_reward_normalization(self) -> None:
        """
        Disable reward normalization.
        """
        self.use_reward_normalization = False
        self.reward_stats_tracker.disable_z_score_normalization()
    
    def reset_statistics(self) -> None:
        """
        Reset all reward statistics.
        """
        self.reward_stats_tracker.reset()
