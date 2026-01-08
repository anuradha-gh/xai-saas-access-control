"""
Counterfactual Explainer for Anomaly Detection (C1)
Shows what changes would make an anomalous sample normal
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.spatial.distance import euclidean


class CounterfactualExplainer:
    """
    Counterfactual explanations for anomaly detection
    
    Shows:
    - What would need to change to make this sample normal
    - Nearest normal sample for comparison
    - Feature-wise differences
    - Actionable recommendations
    """
    
    def __init__(self, model, background_data: np.ndarray, feature_names: List[str] = None):
        """
        Initialize counterfactual explainer
        
        Args:
            model: Isolation Forest model
            background_data: Normal samples for comparison
            feature_names: Names of features
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names or [f"feature_{i}" for i in range(background_data.shape[1])]
        
        # Pre-compute scores for background data
        self.background_scores = model.decision_function(background_data)
        
        # Separate normal and anomalous samples
        self.normal_samples = background_data[self.background_scores > 0]
        self.normal_scores = self.background_scores[self.background_scores > 0]
    
    def find_counterfactual(self, anomalous_sample: np.ndarray, 
                           top_k_features: int = 5) -> Dict[str, Any]:
        """
        Find counterfactual explanation
        
        Args:
            anomalous_sample: Anomalous feature vector
            top_k_features: Number of top features to highlight
        
        Returns:
            Dict with counterfactual analysis
        """
        # Get anomaly score
        anomaly_score = self.model.decision_function(anomalous_sample.reshape(1, -1))[0]
        
        # Find nearest normal sample
        nearest_normal, normal_score, distance = self._find_nearest_normal(anomalous_sample)
        
        # Calculate feature-wise differences
        diff = anomalous_sample - nearest_normal
        abs_diff = np.abs(diff)
        
        # Get top differing features
        top_indices = np.argsort(abs_diff)[::-1][:top_k_features]
        
        feature_changes = []
        for idx in top_indices:
            feature_changes.append({
                'feature': self.feature_names[idx],
                'current_value': float(anomalous_sample[idx]),
                'normal_value': float(nearest_normal[idx]),
                'difference': float(diff[idx]),
                'absolute_difference': float(abs_diff[idx]),
                'change_needed': self._interpret_change(
                    self.feature_names[idx],
                    anomalous_sample[idx],
                    nearest_normal[idx]
                )
            })
        
        # Calculate how much score would improve
        score_improvement = normal_score - anomaly_score
        
        return {
            'is_anomaly': anomaly_score < 0,
            'anomaly_score': float(anomaly_score),
            'nearest_normal_score': float(normal_score),
            'score_improvement_needed': float(score_improvement),
            'distance_to_normal': float(distance),
            'top_feature_changes': feature_changes,
            'summary': self._generate_summary(anomaly_score, feature_changes)
        }
    
    def _find_nearest_normal(self, sample: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Find nearest normal sample"""
        if len(self.normal_samples) == 0:
            # No normal samples, use closest in background
            distances = np.linalg.norm(self.background_data - sample, axis=1)
            nearest_idx = np.argmin(distances)
            return (
                self.background_data[nearest_idx],
                self.background_scores[nearest_idx],
                distances[nearest_idx]
            )
        
        # Find closest normal sample
        distances = np.linalg.norm(self.normal_samples - sample, axis=1)
        nearest_idx = np.argmin(distances)
        
        return (
            self.normal_samples[nearest_idx],
            self.normal_scores[nearest_idx],
            distances[nearest_idx]
        )
    
    def _interpret_change(self, feature_name: str, current: float, normal: float) -> str:
        """Generate human-readable change recommendation"""
        diff = current - normal
        
        if 'mse' in feature_name.lower():
            if diff > 0:
                return f"Reduce reconstruction error by {abs(diff):.3f} (make behavior more typical)"
            else:
                return f"Current reconstruction error is already low"
        
        elif 'latent' in feature_name.lower():
            if diff > 0:
                return f"Decrease latent activation by {abs(diff):.3f}"
            else:
                return f"Increase latent activation by {abs(diff):.3f}"
        
        else:
            if diff > 0:
                return f"Decrease from {current:.3f} to {normal:.3f}"
            else:
                return f"Increase from {current:.3f} to {normal:.3f}"
    
    def _generate_summary(self, anomaly_score: float, feature_changes: List[Dict]) -> str:
        """Generate human-readable summary"""
        if anomaly_score >= 0:
            return "This sample is classified as NORMAL. No changes needed."
        
        top_change = feature_changes[0] if feature_changes else None
        
        if top_change:
            summary = f"To make this normal, primarily: {top_change['change_needed']}. "
            summary += f"This would reduce the anomaly score from {anomaly_score:.3f} to approximately 0."
        else:
            summary = f"Anomaly detected (score: {anomaly_score:.3f}). Consider reviewing the activity."
        
        return summary
    
    def visualize(self, counterfactual: Dict[str, Any]) -> str:
        """Create text visualization of counterfactual"""
        viz = f"\n{'='*70}\n"
        viz += f"COUNTERFACTUAL EXPLANATION\n"
        viz += f"{'='*70}\n\n"
        
        viz += f"Current Status: {'ANOMALY' if counterfactual['is_anomaly'] else 'NORMAL'}\n"
        viz += f"Anomaly Score: {counterfactual['anomaly_score']:.4f}\n"
        viz += f"Distance to Nearest Normal: {counterfactual['distance_to_normal']:.4f}\n\n"
        
        if counterfactual['is_anomaly']:
            viz += f"💡 TO MAKE THIS NORMAL:\n"
            viz += f"{'-'*70}\n"
            
            for i, change in enumerate(counterfactual['top_feature_changes'], 1):
                viz += f"\n{i}. {change['feature']}:\n"
                viz += f"   Current:  {change['current_value']:.4f}\n"
                viz += f"   Normal:   {change['normal_value']:.4f}\n"
                viz += f"   Change:   {change['change_needed']}\n"
            
            viz += f"\n{'-'*70}\n"
            viz += f"Summary: {counterfactual['summary']}\n"
        else:
            viz += f"✓ This sample is already classified as normal.\n"
        
        viz += f"{'='*70}\n"
        
        return viz


# Example usage:
"""
from xai_counterfactual import CounterfactualExplainer

# Initialize
cf_explainer = CounterfactualExplainer(
    model=isolation_forest,
    background_data=normal_samples,
    feature_names=['latent_0', 'latent_1', ..., 'mse']
)

# Find counterfactual
counterfactual = cf_explainer.find_counterfactual(anomalous_features)

# Display
print(cf_explainer.visualize(counterfactual))

# Or access programmatically
for change in counterfactual['top_feature_changes']:
    print(f"{change['feature']}: {change['change_needed']}")
"""
