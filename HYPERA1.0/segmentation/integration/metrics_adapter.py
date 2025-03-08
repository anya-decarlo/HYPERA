#!/usr/bin/env python
# Metrics Adapter - Adapts metrics from different frameworks for use in HYPERA

import os
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import importlib
from collections import defaultdict

class MetricsAdapter:
    """
    Adapts metrics from different frameworks for use in HYPERA.
    
    This class provides a unified interface for accessing metrics from different
    frameworks such as MONAI, PyTorch, and custom implementations. It handles
    conversion between different formats and ensures consistent naming and scaling.
    
    Attributes:
        verbose (bool): Whether to print verbose output
        available_frameworks (dict): Dictionary of available frameworks and their metrics
        metric_mappings (dict): Mappings between framework-specific and unified metric names
        metric_scalings (dict): Scaling factors for different metrics
        registered_metrics (dict): Dictionary of registered metrics and their sources
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the metrics adapter.
        
        Args:
            verbose: Whether to print verbose output
        """
        self.verbose = verbose
        self.logger = logging.getLogger("MetricsAdapter")
        
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        # Initialize available frameworks
        self.available_frameworks = {}
        self._detect_available_frameworks()
        
        # Initialize metric mappings (framework-specific to unified)
        self.metric_mappings = {
            "monai": {
                "dice": "dice_score",
                "dice_metric": "dice_score",
                "hausdorff_distance": "hausdorff_distance",
                "hausdorff_distance_95": "hausdorff_distance_95",
                "surface_dice": "surface_dice",
                "precision": "precision",
                "recall": "recall",
                "f1_score": "f1_score"
            },
            "pytorch": {
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1_score"
            },
            "sklearn": {
                "accuracy_score": "accuracy",
                "precision_score": "precision",
                "recall_score": "recall",
                "f1_score": "f1_score",
                "jaccard_score": "jaccard_index"
            },
            "custom": {
                "dice_score": "dice_score",
                "hausdorff_distance": "hausdorff_distance",
                "boundary_iou": "boundary_iou",
                "shape_score": "shape_score",
                "fg_balance": "fg_balance"
            }
        }
        
        # Initialize metric scalings (to ensure all metrics are in [0, 1] range)
        self.metric_scalings = {
            "hausdorff_distance": lambda x: 1.0 / (1.0 + x),  # Convert to [0, 1] where 1 is best
            "hausdorff_distance_95": lambda x: 1.0 / (1.0 + x),
            "mean_surface_distance": lambda x: 1.0 / (1.0 + x)
        }
        
        # Initialize registered metrics
        self.registered_metrics = {}
        
        # Register default metrics
        self._register_default_metrics()
        
        if self.verbose:
            self.logger.info(f"Initialized MetricsAdapter with frameworks: {list(self.available_frameworks.keys())}")
    
    def _detect_available_frameworks(self):
        """
        Detect available metric frameworks.
        """
        # Check for MONAI
        try:
            import monai.metrics
            self.available_frameworks["monai"] = {
                "module": monai.metrics,
                "metrics": self._get_monai_metrics()
            }
            if self.verbose:
                self.logger.info("Detected MONAI framework")
        except ImportError:
            if self.verbose:
                self.logger.info("MONAI framework not available")
        
        # Check for PyTorch
        try:
            import torchmetrics
            self.available_frameworks["pytorch"] = {
                "module": torchmetrics,
                "metrics": self._get_pytorch_metrics()
            }
            if self.verbose:
                self.logger.info("Detected PyTorch Metrics framework")
        except ImportError:
            if self.verbose:
                self.logger.info("PyTorch Metrics framework not available")
        
        # Check for scikit-learn
        try:
            import sklearn.metrics
            self.available_frameworks["sklearn"] = {
                "module": sklearn.metrics,
                "metrics": self._get_sklearn_metrics()
            }
            if self.verbose:
                self.logger.info("Detected scikit-learn framework")
        except ImportError:
            if self.verbose:
                self.logger.info("scikit-learn framework not available")
        
        # Always register custom metrics
        self.available_frameworks["custom"] = {
            "module": None,
            "metrics": {}
        }
    
    def _get_monai_metrics(self):
        """
        Get available MONAI metrics.
        
        Returns:
            Dictionary of available MONAI metrics
        """
        import monai.metrics
        
        metrics = {}
        
        # Add common MONAI metrics
        metrics["dice"] = monai.metrics.DiceMetric
        metrics["hausdorff_distance"] = monai.metrics.HausdorffDistanceMetric
        metrics["surface_dice"] = monai.metrics.SurfaceDiceMetric
        metrics["mean_surface_distance"] = monai.metrics.MeanSurfaceDistanceMetric
        
        return metrics
    
    def _get_pytorch_metrics(self):
        """
        Get available PyTorch metrics.
        
        Returns:
            Dictionary of available PyTorch metrics
        """
        import torchmetrics
        
        metrics = {}
        
        # Add common PyTorch metrics
        metrics["accuracy"] = torchmetrics.Accuracy
        metrics["precision"] = torchmetrics.Precision
        metrics["recall"] = torchmetrics.Recall
        metrics["f1"] = torchmetrics.F1Score
        
        return metrics
    
    def _get_sklearn_metrics(self):
        """
        Get available scikit-learn metrics.
        
        Returns:
            Dictionary of available scikit-learn metrics
        """
        import sklearn.metrics
        
        metrics = {}
        
        # Add common scikit-learn metrics
        metrics["accuracy_score"] = sklearn.metrics.accuracy_score
        metrics["precision_score"] = sklearn.metrics.precision_score
        metrics["recall_score"] = sklearn.metrics.recall_score
        metrics["f1_score"] = sklearn.metrics.f1_score
        metrics["jaccard_score"] = sklearn.metrics.jaccard_score
        
        return metrics
    
    def _register_default_metrics(self):
        """
        Register default metrics.
        """
        # Register metrics from available frameworks
        for framework, framework_info in self.available_frameworks.items():
            for metric_name, metric_func in framework_info["metrics"].items():
                unified_name = self.metric_mappings.get(framework, {}).get(metric_name, metric_name)
                self.register_metric(unified_name, metric_func, framework)
    
    def register_metric(self, name: str, metric_func: Any, source: str = "custom") -> bool:
        """
        Register a new metric.
        
        Args:
            name: Name of the metric
            metric_func: Metric function or class
            source: Source framework of the metric
            
        Returns:
            Whether the registration was successful
        """
        if name in self.registered_metrics and self.verbose:
            self.logger.warning(f"Overriding existing metric: {name}")
        
        self.registered_metrics[name] = {
            "func": metric_func,
            "source": source
        }
        
        if source == "custom" and source in self.available_frameworks:
            self.available_frameworks[source]["metrics"][name] = metric_func
        
        if self.verbose:
            self.logger.info(f"Registered metric: {name} from {source}")
        
        return True
    
    def calculate_metric(self, name: str, y_pred: Union[torch.Tensor, np.ndarray], 
                         y_true: Union[torch.Tensor, np.ndarray], **kwargs) -> float:
        """
        Calculate a metric.
        
        Args:
            name: Name of the metric
            y_pred: Predicted values
            y_true: True values
            **kwargs: Additional arguments for the metric
            
        Returns:
            Metric value
        """
        if name not in self.registered_metrics:
            if self.verbose:
                self.logger.warning(f"Metric not registered: {name}")
            return None
        
        # Get metric info
        metric_info = self.registered_metrics[name]
        metric_func = metric_info["func"]
        source = metric_info["source"]
        
        # Convert inputs to appropriate format
        y_pred_converted, y_true_converted = self._convert_inputs(y_pred, y_true, source)
        
        # Calculate metric
        try:
            if source == "monai":
                # MONAI metrics are typically classes
                if isinstance(metric_func, type):
                    metric_instance = metric_func(**kwargs)
                    metric_instance(y_pred_converted, y_true_converted)
                    result = metric_instance.aggregate().item()
                else:
                    result = metric_func(y_pred_converted, y_true_converted, **kwargs)
                    if isinstance(result, torch.Tensor):
                        result = result.item()
            
            elif source == "pytorch":
                # PyTorch metrics are typically classes
                if isinstance(metric_func, type):
                    metric_instance = metric_func(**kwargs)
                    result = metric_instance(y_pred_converted, y_true_converted)
                    if isinstance(result, torch.Tensor):
                        result = result.item()
                else:
                    result = metric_func(y_pred_converted, y_true_converted, **kwargs)
                    if isinstance(result, torch.Tensor):
                        result = result.item()
            
            elif source == "sklearn":
                # scikit-learn metrics are typically functions
                result = metric_func(y_true_converted, y_pred_converted, **kwargs)
            
            else:  # custom
                result = metric_func(y_pred_converted, y_true_converted, **kwargs)
                if isinstance(result, torch.Tensor):
                    result = result.item()
            
            # Apply scaling if needed
            if name in self.metric_scalings:
                result = self.metric_scalings[name](result)
            
            return result
        
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error calculating metric {name}: {str(e)}")
            return None
    
    def _convert_inputs(self, y_pred: Union[torch.Tensor, np.ndarray], 
                        y_true: Union[torch.Tensor, np.ndarray], 
                        framework: str) -> Tuple[Any, Any]:
        """
        Convert inputs to the appropriate format for a framework.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            framework: Target framework
            
        Returns:
            Tuple of converted predicted and true values
        """
        # Convert numpy arrays to torch tensors if needed
        if framework in ["monai", "pytorch"]:
            if isinstance(y_pred, np.ndarray):
                y_pred = torch.from_numpy(y_pred)
            if isinstance(y_true, np.ndarray):
                y_true = torch.from_numpy(y_true)
        
        # Convert torch tensors to numpy arrays if needed
        elif framework == "sklearn":
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()
            if isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()
        
        return y_pred, y_true
    
    def calculate_all_metrics(self, y_pred: Union[torch.Tensor, np.ndarray], 
                              y_true: Union[torch.Tensor, np.ndarray], 
                              metrics: List[str] = None, **kwargs) -> Dict[str, float]:
        """
        Calculate multiple metrics.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            metrics: List of metrics to calculate (None for all registered metrics)
            **kwargs: Additional arguments for the metrics
            
        Returns:
            Dictionary of metric names and values
        """
        if metrics is None:
            metrics = list(self.registered_metrics.keys())
        
        results = {}
        for metric_name in metrics:
            result = self.calculate_metric(metric_name, y_pred, y_true, **kwargs)
            if result is not None:
                results[metric_name] = result
        
        return results
    
    def get_available_metrics(self) -> Dict[str, List[str]]:
        """
        Get available metrics by framework.
        
        Returns:
            Dictionary of frameworks and their available metrics
        """
        metrics_by_framework = {}
        
        for framework, framework_info in self.available_frameworks.items():
            metrics_by_framework[framework] = list(framework_info["metrics"].keys())
        
        return metrics_by_framework
    
    def get_unified_metrics(self) -> List[str]:
        """
        Get list of all unified metric names.
        
        Returns:
            List of unified metric names
        """
        return list(self.registered_metrics.keys())
    
    def register_custom_metric(self, name: str, func: Callable, 
                              scaling_func: Optional[Callable] = None) -> bool:
        """
        Register a custom metric.
        
        Args:
            name: Name of the metric
            func: Metric function
            scaling_func: Optional scaling function for the metric
            
        Returns:
            Whether the registration was successful
        """
        # Register the metric
        success = self.register_metric(name, func, "custom")
        
        # Register scaling function if provided
        if scaling_func is not None and success:
            self.metric_scalings[name] = scaling_func
        
        return success
    
    def batch_convert(self, metrics_dict: Dict[str, Any], source_framework: str) -> Dict[str, float]:
        """
        Convert a dictionary of metrics from a source framework to unified names.
        
        Args:
            metrics_dict: Dictionary of metrics from source framework
            source_framework: Source framework name
            
        Returns:
            Dictionary of metrics with unified names
        """
        unified_metrics = {}
        
        for metric_name, metric_value in metrics_dict.items():
            # Get unified name if available
            unified_name = self.metric_mappings.get(source_framework, {}).get(metric_name, metric_name)
            
            # Convert tensor to float if needed
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.item()
            
            # Apply scaling if needed
            if unified_name in self.metric_scalings:
                metric_value = self.metric_scalings[unified_name](metric_value)
            
            unified_metrics[unified_name] = metric_value
        
        return unified_metrics
    
    def save_state(self, path: str) -> bool:
        """
        Save the adapter state to a file.
        
        Args:
            path: Path to save the state
            
        Returns:
            Whether the save was successful
        """
        try:
            # Create state dictionary (excluding non-serializable objects)
            state = {
                "metric_mappings": self.metric_mappings,
                "metric_scalings": {k: None for k in self.metric_scalings},  # Can't serialize functions
                "registered_metrics": {k: {"source": v["source"]} for k, v in self.registered_metrics.items()}
            }
            
            # Save state
            torch.save(state, path)
            
            if self.verbose:
                self.logger.info(f"Saved MetricsAdapter state to {path}")
            
            return True
        
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error saving MetricsAdapter state: {str(e)}")
            
            return False
    
    def load_state(self, path: str) -> bool:
        """
        Load the adapter state from a file.
        
        Args:
            path: Path to load the state from
            
        Returns:
            Whether the load was successful
        """
        try:
            # Load state
            state = torch.load(path)
            
            # Update state
            self.metric_mappings = state["metric_mappings"]
            
            # Re-register metrics
            for metric_name, metric_info in state["registered_metrics"].items():
                source = metric_info["source"]
                
                if source in self.available_frameworks and source != "custom":
                    # Get metric function from framework
                    framework_metrics = self.available_frameworks[source]["metrics"]
                    
                    # Find original metric name
                    original_name = None
                    for orig, unified in self.metric_mappings.get(source, {}).items():
                        if unified == metric_name:
                            original_name = orig
                            break
                    
                    if original_name and original_name in framework_metrics:
                        metric_func = framework_metrics[original_name]
                        self.register_metric(metric_name, metric_func, source)
            
            if self.verbose:
                self.logger.info(f"Loaded MetricsAdapter state from {path}")
            
            return True
        
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error loading MetricsAdapter state: {str(e)}")
            
            return False
