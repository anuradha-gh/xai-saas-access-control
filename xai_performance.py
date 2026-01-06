"""
Performance Optimization Module for XAI
Adds caching and batch processing to speed up SHAP explanations
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List
import numpy as np


class ExplanationCache:
    """Cache for XAI explanations to avoid recomputing"""
    
    def __init__(self, cache_dir: str = ".xai_cache", max_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.memory_cache = {}  # In-memory cache for speed
        
    def _get_hash(self, data: Any) -> str:
        """Generate hash for data"""
        if isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        elif isinstance(data, dict):
            data_bytes = json.dumps(data, sort_keys=True).encode()
        elif isinstance(data, str):
            data_bytes = data.encode()
        else:
            data_bytes = str(data).encode()
            
        return hashlib.md5(data_bytes).hexdigest()
    
    def get(self, component: str, data: Any) -> Dict[str, Any]:
        """Get cached explanation"""
        key = f"{component}_{self._get_hash(data)}"
        
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
                self.memory_cache[key] = result  # Add to memory
                return result
        
        return None
    
    def set(self, component: str, data: Any, explanation: Dict[str, Any]):
        """Cache explanation"""
        key = f"{component}_{self._get_hash(data)}"
        
        # Add to memory cache
        self.memory_cache[key] = explanation
        
        # Limit memory cache size
        if len(self.memory_cache) > self.max_size:
            # Remove oldest (first) item
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        # Save to disk
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(explanation, f)
    
    def clear(self):
        """Clear all caches"""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()


class BatchExplainer:
    """Batch processing for multiple explanations"""
    
    def __init__(self, xai_factory, cache: ExplanationCache = None):
        self.xai_factory = xai_factory
        self.cache = cache or ExplanationCache()
    
    def explain_batch_c1(self, features_list: List[np.ndarray], 
                         original_features_list: List[Dict]) -> List[Dict]:
        """
        Explain multiple C1 instances efficiently
        
        Args:
            features_list: List of feature arrays
            original_features_list: List of original feature dicts
        
        Returns:
            List of explanations
        """
        results = []
        uncached_indices = []
        uncached_features = []
        
        # Check cache first
        for i, features in enumerate(features_list):
            cached = self.cache.get('c1', features)
            if cached:
                results.append(cached)
            else:
                results.append(None)  # Placeholder
                uncached_indices.append(i)
                uncached_features.append(features)
        
        # Compute uncached explanations
        if uncached_features:
            # Batch SHAP computation (if explainer supports it)
            for idx, features in zip(uncached_indices, uncached_features):
                explanation = self.xai_factory.explain_c1(
                    features, 
                    original_features_list[idx]
                )
                results[idx] = explanation
                self.cache.set('c1', features, explanation)
        
        return results
    
    def explain_batch_c2(self, texts: List[str]) -> List[Dict]:
        """Explain multiple C2 instances"""
        results = []
        
        for text in texts:
            cached = self.cache.get('c2', text)
            if cached:
                results.append(cached)
            else:
                explanation = self.xai_factory.explain_c2(text)
                self.cache.set('c2', text, explanation)
                results.append(explanation)
        
        return results
    
    def explain_batch_c3(self, texts: List[str], 
                         embeddings: List[np.ndarray]) -> List[Dict]:
        """Explain multiple C3 instances"""
        results = []
        
        for text, embedding in zip(texts, embeddings):
            cached = self.cache.get('c3', embedding)
            if cached:
                results.append(cached)
            else:
                explanation = self.xai_factory.explain_c3(text, embedding)
                self.cache.set('c3', embedding, explanation)
                results.append(explanation)
        
        return results


# Example usage:
"""
from xai_performance import ExplanationCache, BatchExplainer

# In XAI_enhanced.py load_system():
state['xai_cache'] = ExplanationCache(cache_dir=".xai_cache", max_size=500)
state['batch_explainer'] = BatchExplainer(state['xai_explainer'], state['xai_cache'])

# When explaining:
# Single explanation (with cache):
cached = state['xai_cache'].get('c1', features)
if cached:
    explanation = cached
else:
    explanation = state['xai_explainer'].explain_c1(features, original_log)
    state['xai_cache'].set('c1', features, explanation)

# Batch explanation:
explanations = state['batch_explainer'].explain_batch_c1(
    features_list=[feat1, feat2, feat3],
    original_features_list=[log1, log2, log3]
)
"""
