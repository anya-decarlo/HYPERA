#!/usr/bin/env python
# Metric Processor for Enhanced State Representation

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import deque
import logging
from scipy import stats
from scipy.signal import savgol_filter
import ruptures as rpt  # For changepoint detection
from statsmodels.tsa.seasonal import STL  # For seasonal-trend decomposition
from statsmodels.tsa.stattools import adfuller  # For stationarity tests
from statsmodels.nonparametric.smoothers_lowess import lowess  # For LOWESS smoothing
from statsmodels.stats.diagnostic import acorr_ljungbox  # For autocorrelation tests
import warnings

class MetricProcessor:
    """
    Processes training metrics to extract meaningful signals for agent state representation.
    
    This class enhances metric processing by:
    1. Tracking metrics at multiple time scales
    2. Detecting trends and changepoints
    3. Identifying overfitting signals
    4. Computing statistical features of metric histories
    """
    
    def __init__(
        self,
        short_window: int = 5,
        medium_window: int = 20,
        long_window: int = 50,
        ema_alpha_short: float = 0.3,
        ema_alpha_long: float = 0.05,
        max_history_size: int = 1000,
        verbose: bool = False
    ):
        """
        Initialize the metric processor.
        
        Args:
            short_window: Size of short-term window for trend analysis
            medium_window: Size of medium-term window for trend analysis
            long_window: Size of long-term window for trend analysis
            ema_alpha_short: Alpha parameter for short-term exponential moving average
            ema_alpha_long: Alpha parameter for long-term exponential moving average
            max_history_size: Maximum size of raw metric history to store
            verbose: Whether to print verbose output
        """
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.ema_alpha_short = ema_alpha_short
        self.ema_alpha_long = ema_alpha_long
        self.max_history_size = max_history_size
        self.verbose = verbose
        
        # Initialize metric storage
        self.raw_metrics = {}  # Full history of raw metrics (limited by max_history_size)
        self.ema_short = {}    # Short-term exponential moving averages
        self.ema_long = {}     # Long-term exponential moving averages
        self.critical_points = {}  # Store critical points (local minima/maxima)
        self.changepoints = {}  # Store detected changepoints
        
        # Initialize overfitting signals
        self.overfitting_signals = {
            "early_overfitting": 0.0,
            "late_overfitting": 0.0,
            "train_val_divergence": 0.0
        }
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)
    
    def update(self, metrics: Dict[str, float]) -> None:
        """
        Update all metric statistics with new values.
        
        Args:
            metrics: Dictionary of new metric values
        """
        # Update raw metrics
        for metric_name, value in metrics.items():
            if metric_name not in self.raw_metrics:
                self.raw_metrics[metric_name] = deque(maxlen=self.max_history_size)
                self.ema_short[metric_name] = value
                self.ema_long[metric_name] = value
                self.critical_points[metric_name] = []
                self.changepoints[metric_name] = []
            
            # Add new value to history
            self.raw_metrics[metric_name].append(value)
            
            # Update EMAs
            self.ema_short[metric_name] = (self.ema_alpha_short * value + 
                                          (1 - self.ema_alpha_short) * self.ema_short[metric_name])
            self.ema_long[metric_name] = (self.ema_alpha_long * value + 
                                         (1 - self.ema_alpha_long) * self.ema_long[metric_name])
            
            # Update critical points if needed
            self._update_critical_points(metric_name)
        
        # Update changepoints for key metrics
        for metric_name in ["val_loss", "loss", "dice_score"]:
            if metric_name in self.raw_metrics and len(self.raw_metrics[metric_name]) >= self.medium_window:
                self._detect_changepoints(metric_name)
        
        # Update overfitting signals
        self._update_overfitting_signals()
    
    def _update_critical_points(self, metric_name: str) -> None:
        """
        Update critical points (local minima/maxima) for a metric.
        
        Args:
            metric_name: Name of the metric
        """
        history = list(self.raw_metrics[metric_name])
        if len(history) < 3:
            return
        
        # Check if the second-to-last point is a local minimum or maximum
        if len(history) >= 3:
            i = len(history) - 2
            if history[i-1] > history[i] < history[i+1]:  # Local minimum
                self.critical_points[metric_name].append(("min", i, history[i]))
                if self.verbose:
                    self.logger.info(f"Local minimum detected for {metric_name} at position {i}: {history[i]}")
            
            elif history[i-1] < history[i] > history[i+1]:  # Local maximum
                self.critical_points[metric_name].append(("max", i, history[i]))
                if self.verbose:
                    self.logger.info(f"Local maximum detected for {metric_name} at position {i}: {history[i]}")
        
        # Limit number of critical points stored
        if len(self.critical_points[metric_name]) > 10:
            self.critical_points[metric_name] = self.critical_points[metric_name][-10:]
    
    def _detect_changepoints(self, metric_name: str) -> None:
        """
        Detect changepoints in a metric's history.
        
        Args:
            metric_name: Name of the metric
        """
        history = list(self.raw_metrics[metric_name])
        if len(history) < self.medium_window:
            return
        
        try:
            # Use recent history for changepoint detection
            recent_history = np.array(history[-self.medium_window:])
            
            # Use improved changepoint detection
            changepoints = self._improved_detect_changepoints(recent_history, penalty=1.0)
            
            # Convert local indices to global indices
            global_indices = [len(history) - self.medium_window + idx for idx in changepoints]
            
            # Store only new changepoints
            existing_indices = [cp[1] for cp in self.changepoints[metric_name]]
            for idx in global_indices:
                if idx not in existing_indices:
                    self.changepoints[metric_name].append(("change", idx, history[idx]))
                    if self.verbose:
                        self.logger.info(f"Changepoint detected for {metric_name} at position {idx}: {history[idx]}")
            
            # Limit number of changepoints stored
            if len(self.changepoints[metric_name]) > 5:
                self.changepoints[metric_name] = self.changepoints[metric_name][-5:]
        
        except Exception as e:
            self.logger.warning(f"Error in changepoint detection for {metric_name}: {str(e)}")
    
    def _update_overfitting_signals(self) -> None:
        """
        Update overfitting signals based on training and validation metrics.
        """
        # Check if we have both training and validation loss
        if "loss" in self.raw_metrics and "val_loss" in self.raw_metrics:
            train_loss = list(self.raw_metrics["loss"])
            val_loss = list(self.raw_metrics["val_loss"])
            
            # Need enough history to detect overfitting
            if len(train_loss) < self.short_window or len(val_loss) < self.short_window:
                return
            
            # Use advanced trend analysis
            train_trend_short = self._calculate_advanced_trend(train_loss[-self.short_window:])
            val_trend_short = self._calculate_advanced_trend(val_loss[-self.short_window:])
            
            # Calculate medium-term trends if possible
            if len(train_loss) >= self.medium_window and len(val_loss) >= self.medium_window:
                train_trend_medium = self._calculate_advanced_trend(train_loss[-self.medium_window:])
                val_trend_medium = self._calculate_advanced_trend(val_loss[-self.medium_window:])
            else:
                train_trend_medium = train_trend_short
                val_trend_medium = val_trend_short
            
            # Check for seasonality in validation loss
            seasonality = self._detect_seasonality(val_loss[-min(len(val_loss), 30):])
            
            # Early overfitting: training loss decreasing but validation loss plateauing
            # Use more sophisticated detection with statistical significance
            if (train_trend_short["direction"] < 0 and train_trend_short["is_significant"] and 
                abs(val_trend_short["slope"]) < 0.005):
                self.overfitting_signals["early_overfitting"] = min(1.0, self.overfitting_signals["early_overfitting"] + 0.2)
            else:
                self.overfitting_signals["early_overfitting"] = max(0.0, self.overfitting_signals["early_overfitting"] - 0.1)
            
            # Late overfitting: training loss decreasing but validation loss increasing
            # Use more sophisticated detection with statistical significance
            if (train_trend_medium["direction"] < 0 and train_trend_medium["is_significant"] and 
                val_trend_medium["direction"] > 0 and val_trend_medium["is_significant"]):
                self.overfitting_signals["late_overfitting"] = min(1.0, self.overfitting_signals["late_overfitting"] + 0.2)
            else:
                self.overfitting_signals["late_overfitting"] = max(0.0, self.overfitting_signals["late_overfitting"] - 0.1)
            
            # Train-val divergence: gap between training and validation loss increasing
            recent_train = train_loss[-self.short_window:]
            recent_val = val_loss[-self.short_window:]
            gap = [v - t for t, v in zip(recent_train, recent_val)]
            gap_trend = self._calculate_advanced_trend(gap)
            
            if gap_trend["direction"] > 0 and gap_trend["is_significant"]:
                self.overfitting_signals["train_val_divergence"] = min(1.0, self.overfitting_signals["train_val_divergence"] + 0.2)
            else:
                self.overfitting_signals["train_val_divergence"] = max(0.0, self.overfitting_signals["train_val_divergence"] - 0.1)
            
            # Add seasonality-aware overfitting detection
            if seasonality["has_seasonality"] and seasonality["strength"] > 0.3:
                # If strong seasonality is detected, adjust overfitting signals
                # to avoid false positives during seasonal fluctuations
                period = seasonality["period"]
                if self.verbose:
                    self.logger.info(f"Detected seasonality in validation loss with period {period} and strength {seasonality['strength']:.2f}")
                
                # Decompose validation loss
                if len(val_loss) >= max(3 * period, 15):
                    stl_result = self._perform_stl_decomposition(val_loss, period=period)
                    # Use trend component for overfitting detection
                    trend_only = stl_result["trend"]
                    trend_analysis = self._calculate_advanced_trend(trend_only[-self.short_window:])
                    
                    # Update overfitting signals based on trend component
                    if trend_analysis["direction"] > 0 and trend_analysis["is_significant"]:
                        self.overfitting_signals["late_overfitting"] = min(1.0, self.overfitting_signals["late_overfitting"] + 0.1)
            
            if self.verbose and any(v > 0.5 for v in self.overfitting_signals.values()):
                self.logger.info(f"Overfitting signals: {self.overfitting_signals}")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate the trend in a list of values using linear regression.
        
        Args:
            values: List of values
            
        Returns:
            Slope of the trend line
        """
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope
    
    def _calculate_advanced_trend(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate advanced trend metrics using multiple methods.
        
        This method applies several trend analysis techniques:
        1. Linear regression (simple slope)
        2. LOWESS smoothing (non-parametric trend)
        3. Mann-Kendall test (non-parametric trend test)
        4. Statistical significance of the trend
        
        Args:
            values: List of values
            
        Returns:
            Dictionary of trend metrics
        """
        if len(values) < 5:
            # Fall back to simple linear regression for short sequences
            slope = self._calculate_trend(values)
            return {
                "slope": slope,
                "significance": 0.0,
                "is_significant": False,
                "direction": np.sign(slope),
                "strength": abs(slope)
            }
        
        result = {}
        
        # 1. Linear regression
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        result["slope"] = slope
        result["r_squared"] = r_value**2
        result["p_value"] = p_value
        result["std_err"] = std_err
        result["is_significant"] = p_value < 0.05
        result["significance"] = 1.0 - min(1.0, p_value * 10)  # Scale p-value for easier interpretation
        
        # 2. LOWESS smoothing for non-linear trend
        try:
            # Apply LOWESS smoothing
            smoothed = lowess(values, x, frac=0.3, return_sorted=False)
            
            # Calculate trend direction from smoothed data
            smoothed_slope = self._calculate_trend(smoothed)
            result["nonlinear_slope"] = smoothed_slope
            
            # Calculate trend strength as max deviation from linear trend
            linear_trend = intercept + slope * x
            max_deviation = np.max(np.abs(smoothed - linear_trend))
            result["nonlinearity"] = max_deviation / (np.max(values) - np.min(values)) if np.max(values) != np.min(values) else 0.0
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Error in LOWESS smoothing: {str(e)}")
            result["nonlinear_slope"] = slope
            result["nonlinearity"] = 0.0
        
        # 3. Mann-Kendall test (non-parametric trend test)
        try:
            # Calculate Mann-Kendall statistic
            n = len(values)
            s = 0
            for i in range(n-1):
                for j in range(i+1, n):
                    s += np.sign(values[j] - values[i])
            
            # Calculate variance
            unique_values = set(values)
            g = len(unique_values)
            if n == g:  # No ties
                var_s = (n * (n - 1) * (2 * n + 5)) / 18
            else:  # Handle ties
                tp = np.zeros(len(unique_values))
                for i, v in enumerate(unique_values):
                    tp[i] = sum(1 for val in values if val == v)
                var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18
            
            # Calculate z-score
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0
            
            # Calculate p-value
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            
            result["mk_statistic"] = s
            result["mk_p_value"] = p
            result["mk_is_significant"] = p < 0.05
            result["mk_direction"] = np.sign(s)
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Error in Mann-Kendall test: {str(e)}")
            result["mk_statistic"] = 0
            result["mk_p_value"] = 1.0
            result["mk_is_significant"] = False
            result["mk_direction"] = 0
        
        # Combine results for overall trend assessment
        result["direction"] = np.sign(slope) if p_value < 0.05 else (
            result["mk_direction"] if result.get("mk_p_value", 1.0) < 0.05 else 0)
        result["strength"] = abs(slope) * result["significance"]
        
        return result
    
    def _perform_stl_decomposition(self, values: List[float], period: int = 5) -> Dict[str, np.ndarray]:
        """
        Perform Seasonal-Trend decomposition using LOESS (STL).
        
        This decomposes a time series into trend, seasonal, and residual components.
        
        Args:
            values: List of values
            period: Period for seasonal component
            
        Returns:
            Dictionary with trend, seasonal, and residual components
        """
        if len(values) < max(2 * period, 12):
            # Not enough data for meaningful decomposition
            return {
                "trend": np.array(values),
                "seasonal": np.zeros(len(values)),
                "residual": np.zeros(len(values))
            }
        
        try:
            # Convert to numpy array
            data = np.array(values)
            
            # Apply STL decomposition
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stl = STL(data, period=period, robust=True)
                result = stl.fit()
            
            return {
                "trend": result.trend,
                "seasonal": result.seasonal,
                "residual": result.resid
            }
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Error in STL decomposition: {str(e)}")
            return {
                "trend": np.array(values),
                "seasonal": np.zeros(len(values)),
                "residual": np.zeros(len(values))
            }
    
    def _improved_detect_changepoints(self, values: List[float], penalty: float = 1.0) -> List[int]:
        """
        Improved changepoint detection with multiple algorithms and statistical validation.
        
        This method:
        1. Applies multiple changepoint detection algorithms
        2. Validates changepoints using statistical tests
        3. Combines results for more robust detection
        
        Args:
            values: List of values
            penalty: Penalty parameter for changepoint detection
            
        Returns:
            List of detected changepoint indices
        """
        if len(values) < 10:
            return []
        
        try:
            # Convert to numpy array
            data = np.array(values).reshape(-1, 1)
            
            # Apply smoothing to reduce noise
            window_length = min(5, len(values) - (len(values) % 2) - 1)
            if window_length >= 3:
                smoothed_data = savgol_filter(data.flatten(), window_length, 1).reshape(-1, 1)
            else:
                smoothed_data = data
            
            # 1. PELT algorithm with RBF kernel
            algo_pelt = rpt.Pelt(model="rbf").fit(smoothed_data)
            result_pelt = algo_pelt.predict(pen=penalty)
            
            # 2. Binary Segmentation with L2 cost
            algo_binseg = rpt.Binseg(model="l2").fit(smoothed_data)
            result_binseg = algo_binseg.predict(pen=penalty)
            
            # 3. Bottom-Up segmentation
            algo_bottomup = rpt.BottomUp(model="l2").fit(smoothed_data)
            result_bottomup = algo_bottomup.predict(pen=penalty * 2)  # Higher penalty for Bottom-Up
            
            # Combine results (excluding the last point which is always included)
            all_changepoints = set(result_pelt[:-1] + result_binseg[:-1] + result_bottomup[:-1])
            
            # Validate changepoints using statistical tests
            validated_changepoints = []
            for cp in sorted(all_changepoints):
                if cp < 5 or cp > len(values) - 5:
                    continue  # Skip points too close to the edges
                
                # Split data at changepoint
                before = values[max(0, cp-10):cp]
                after = values[cp:min(len(values), cp+10)]
                
                if len(before) < 5 or len(after) < 5:
                    continue
                
                # Test for difference in means
                t_stat, p_value = stats.ttest_ind(before, after, equal_var=False)
                
                # Test for difference in variance
                f_stat = np.var(before, ddof=1) / np.var(after, ddof=1)
                f_p_value = 1 - stats.f.cdf(f_stat, len(before)-1, len(after)-1)
                if f_stat < 1:
                    f_p_value = stats.f.cdf(1/f_stat, len(after)-1, len(before)-1)
                
                # Accept changepoint if either test is significant
                if p_value < 0.1 or f_p_value < 0.1:
                    validated_changepoints.append(cp)
            
            return validated_changepoints
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Error in improved changepoint detection: {str(e)}")
            return []
    
    def _detect_seasonality(self, values: List[float], max_lag: int = 20) -> Dict[str, float]:
        """
        Detect seasonality in time series data.
        
        This method:
        1. Performs autocorrelation analysis
        2. Tests for significant seasonal patterns
        3. Estimates the dominant period
        
        Args:
            values: List of values
            max_lag: Maximum lag to consider for autocorrelation
            
        Returns:
            Dictionary with seasonality metrics
        """
        if len(values) < max_lag + 5:
            return {
                "has_seasonality": False,
                "period": 0,
                "strength": 0.0
            }
        
        try:
            # Calculate autocorrelation
            data = np.array(values)
            n = len(data)
            mean = np.mean(data)
            var = np.var(data)
            
            if var == 0:
                return {
                    "has_seasonality": False,
                    "period": 0,
                    "strength": 0.0
                }
            
            # Calculate autocorrelation for different lags
            autocorr = np.zeros(max_lag + 1)
            for lag in range(max_lag + 1):
                if lag == 0:
                    autocorr[lag] = 1.0
                else:
                    autocorr[lag] = np.sum((data[lag:] - mean) * (data[:-lag] - mean)) / ((n - lag) * var)
            
            # Find peaks in autocorrelation
            peaks = []
            for i in range(2, max_lag):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.2:
                    peaks.append((i, autocorr[i]))
            
            # Test for significant autocorrelation using Ljung-Box test
            lb_stat, lb_p_value = acorr_ljungbox(data, lags=[max_lag], return_df=False)
            has_autocorr = lb_p_value[0] < 0.05
            
            if not peaks or not has_autocorr:
                return {
                    "has_seasonality": False,
                    "period": 0,
                    "strength": 0.0
                }
            
            # Find dominant period
            dominant_period, strength = max(peaks, key=lambda x: x[1])
            
            # Perform STL decomposition to validate seasonality
            stl_result = self._perform_stl_decomposition(values, period=dominant_period)
            seasonal_strength = np.var(stl_result["seasonal"]) / (np.var(stl_result["seasonal"]) + np.var(stl_result["residual"]))
            
            return {
                "has_seasonality": seasonal_strength > 0.1,
                "period": dominant_period,
                "strength": seasonal_strength
            }
            
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Error in seasonality detection: {str(e)}")
            return {
                "has_seasonality": False,
                "period": 0,
                "strength": 0.0
            }
    
    def get_processed_metrics(self, metric_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Get processed metrics for state representation.
        
        Args:
            metric_names: List of metric names to process
            
        Returns:
            Dictionary of processed metrics for each requested metric
        """
        result = {}
        
        for metric_name in metric_names:
            if metric_name not in self.raw_metrics:
                continue
            
            metric_data = {}
            
            # Get raw history
            history = list(self.raw_metrics[metric_name])
            
            # Current value
            metric_data["current"] = history[-1] if history else 0.0
            
            # EMAs
            metric_data["ema_short"] = self.ema_short.get(metric_name, 0.0)
            metric_data["ema_long"] = self.ema_long.get(metric_name, 0.0)
            
            # Use advanced trend analysis for short window
            if len(history) >= self.short_window:
                short_trend = self._calculate_advanced_trend(history[-self.short_window:])
                metric_data["trend_short"] = short_trend["slope"]
                metric_data["trend_short_significant"] = float(short_trend["is_significant"])
                metric_data["trend_short_strength"] = short_trend["strength"]
                metric_data["trend_short_nonlinearity"] = short_trend.get("nonlinearity", 0.0)
            else:
                metric_data["trend_short"] = 0.0
                metric_data["trend_short_significant"] = 0.0
                metric_data["trend_short_strength"] = 0.0
                metric_data["trend_short_nonlinearity"] = 0.0
            
            # Use advanced trend analysis for medium window
            if len(history) >= self.medium_window:
                medium_trend = self._calculate_advanced_trend(history[-self.medium_window:])
                metric_data["trend_medium"] = medium_trend["slope"]
                metric_data["trend_medium_significant"] = float(medium_trend["is_significant"])
                metric_data["trend_medium_strength"] = medium_trend["strength"]
                metric_data["trend_medium_nonlinearity"] = medium_trend.get("nonlinearity", 0.0)
            else:
                metric_data["trend_medium"] = metric_data["trend_short"]
                metric_data["trend_medium_significant"] = metric_data["trend_short_significant"]
                metric_data["trend_medium_strength"] = metric_data["trend_short_strength"]
                metric_data["trend_medium_nonlinearity"] = metric_data["trend_short_nonlinearity"]
            
            # Use advanced trend analysis for long window
            if len(history) >= self.long_window:
                long_trend = self._calculate_advanced_trend(history[-self.long_window:])
                metric_data["trend_long"] = long_trend["slope"]
                metric_data["trend_long_significant"] = float(long_trend["is_significant"])
                metric_data["trend_long_strength"] = long_trend["strength"]
                metric_data["trend_long_nonlinearity"] = long_trend.get("nonlinearity", 0.0)
            else:
                metric_data["trend_long"] = metric_data["trend_medium"]
                metric_data["trend_long_significant"] = metric_data["trend_medium_significant"]
                metric_data["trend_long_strength"] = metric_data["trend_medium_strength"]
                metric_data["trend_long_nonlinearity"] = metric_data["trend_medium_nonlinearity"]
            
            # Check for seasonality
            if len(history) >= 20:
                seasonality = self._detect_seasonality(history[-min(len(history), 50):])
                metric_data["has_seasonality"] = float(seasonality["has_seasonality"])
                metric_data["seasonality_period"] = float(seasonality["period"])
                metric_data["seasonality_strength"] = seasonality["strength"]
            else:
                metric_data["has_seasonality"] = 0.0
                metric_data["seasonality_period"] = 0.0
                metric_data["seasonality_strength"] = 0.0
            
            # Volatility (standard deviation)
            if len(history) >= self.short_window:
                metric_data["volatility_short"] = np.std(history[-self.short_window:])
            else:
                metric_data["volatility_short"] = 0.0
            
            if len(history) >= self.medium_window:
                metric_data["volatility_medium"] = np.std(history[-self.medium_window:])
            else:
                metric_data["volatility_medium"] = metric_data["volatility_short"]
            
            if len(history) >= self.long_window:
                metric_data["volatility_long"] = np.std(history[-self.long_window:])
            else:
                metric_data["volatility_long"] = metric_data["volatility_medium"]
            
            result[metric_name] = metric_data
        
        return result
    
    def get_overfitting_signals(self) -> Dict[str, float]:
        """
        Get current overfitting signals.
        
        Returns:
            Dictionary of overfitting signals
        """
        return self.overfitting_signals.copy()
    
    def get_enhanced_state_features(self, metric_names: List[str]) -> Dict[str, float]:
        """
        Get enhanced state features for agent state representation.
        
        This method combines processed metrics and overfitting signals into
        a single dictionary of features for state representation.
        
        Args:
            metric_names: List of metric names to include
            
        Returns:
            Dictionary of state features
        """
        features = {}
        
        # Add processed metrics
        processed_metrics = self.get_processed_metrics(metric_names)
        for metric_name, metric_data in processed_metrics.items():
            for feature_name, value in metric_data.items():
                features[f"{metric_name}_{feature_name}"] = value
        
        # Add overfitting signals
        for signal_name, value in self.overfitting_signals.items():
            features[signal_name] = value
        
        # Add relationships between metrics if available
        if "loss" in processed_metrics and "val_loss" in processed_metrics:
            # Train/val ratio
            train_val_ratio = (processed_metrics["val_loss"]["current"] / 
                              processed_metrics["loss"]["current"] 
                              if processed_metrics["loss"]["current"] > 0 else 1.0)
            features["train_val_ratio"] = min(train_val_ratio, 5.0)  # Cap at 5.0 to avoid extreme values
            
            # Trend alignment (are train and val trends aligned?)
            train_trend = processed_metrics["loss"]["trend_medium"]
            val_trend = processed_metrics["val_loss"]["trend_medium"]
            features["trend_alignment"] = np.sign(train_trend) == np.sign(val_trend)
        
        return features
    
    def get_enhanced_state_vector(self, metric_names: List[str], feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Get enhanced state vector for agent state representation.
        
        This method converts the enhanced state features into a numpy array
        for direct use in agent state representation.
        
        Args:
            metric_names: List of metric names to include
            feature_names: List of specific features to include (if None, include all)
            
        Returns:
            Numpy array of state features
        """
        features = self.get_enhanced_state_features(metric_names)
        
        if feature_names is not None:
            # Filter to only requested features
            filtered_features = {k: features.get(k, 0.0) for k in feature_names}
            return np.array(list(filtered_features.values()), dtype=np.float32)
        else:
            # Use all features
            return np.array(list(features.values()), dtype=np.float32)


