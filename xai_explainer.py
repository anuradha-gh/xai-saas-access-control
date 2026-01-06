"""
XAI Explainer Module
Provides interpretability techniques for the three classification models:
- C1: Autoencoder + Isolation Forest (Anomaly Detection)
- C2: BERT (Role Classification)
- C3: Sentence-BERT + Isolation Forest (Access Decision)
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Any, Optional
import json

# Suppress SHAP warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not installed. Run: pip install shap")

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME not installed. Run: pip install lime")


class SHAPExplainer:
    """SHAP explanations for Isolation Forest models (C1, C3)"""
    
    def __init__(self, model, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
    def initialize(self, background_data: np.ndarray):
        """Initialize SHAP explainer with background data"""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for this explainer")
        
        # Use KernelExplainer for Isolation Forest
        # We create a wrapper to get consistent output
        def model_predict(X):
            return self.model.decision_function(X)
        
        # Use a sample of background data for efficiency
        if len(background_data) > 100:
            background_sample = shap.sample(background_data, 100)
        else:
            background_sample = background_data
            
        self.explainer = shap.KernelExplainer(model_predict, background_sample)
        
    def explain(self, instance: np.ndarray, num_features: int = 10) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single instance
        
        Returns:
            Dict with 'shap_values', 'feature_importance', 'base_value'
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize() first.")
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(instance, nsamples=100)
        
        # Reshape if needed
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
        
        # Get feature importance (absolute SHAP values)
        importance = np.abs(shap_values[0])
        
        # Get top features
        top_indices = np.argsort(importance)[::-1][:num_features]
        
        feature_importance = []
        for idx in top_indices:
            feature_name = self.feature_names[idx] if self.feature_names else f"feature_{idx}"
            feature_importance.append({
                "feature": feature_name,
                "shap_value": float(shap_values[0][idx]),
                "importance": float(importance[idx]),
                "feature_value": float(instance.flatten()[idx]) if instance.size > idx else 0.0
            })
        
        return {
            "shap_values": shap_values[0].tolist(),
            "feature_importance": feature_importance,
            "base_value": float(self.explainer.expected_value)
        }


class ReconstructionExplainer:
    """Reconstruction error analysis for Autoencoder (C1)"""
    
    def __init__(self, autoencoder, preprocessor, feature_names: List[str]):
        self.autoencoder = autoencoder
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        
    def explain(self, instance: np.ndarray, original_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze which features have highest reconstruction error
        
        Args:
            instance: Scaled feature vector
            original_features: Original categorical features (before encoding)
        
        Returns:
            Dict with reconstruction errors per original feature
        """
        # Get reconstruction
        reconstruction = self.autoencoder.predict(instance, verbose=0)
        
        # Compute element-wise reconstruction error
        errors = np.abs(instance - reconstruction).flatten()
        
        # Map back to original features
        # This is tricky because of one-hot encoding
        # We'll aggregate errors by original feature
        
        feature_errors = {}
        
        # Get one-hot encoder info
        try:
            onehot_encoder = self.preprocessor.named_transformers_['cat']
            categories = onehot_encoder.categories_
            
            idx = 0
            for feat_idx, feature_name in enumerate(self.feature_names):
                # Get the size of one-hot encoding for this feature
                n_categories = len(categories[feat_idx])
                
                # Sum errors across all one-hot columns for this feature
                feature_error = np.sum(errors[idx:idx + n_categories])
                
                # Normalize by number of categories
                feature_errors[feature_name] = float(feature_error / n_categories)
                
                idx += n_categories
                
        except Exception as e:
            # Fallback: just use feature names directly
            for i, name in enumerate(self.feature_names):
                if i < len(errors):
                    feature_errors[name] = float(errors[i])
        
        # Sort by error magnitude
        sorted_errors = sorted(feature_errors.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_mse": float(np.mean(errors ** 2)),
            "feature_errors": [
                {
                    "feature": feat,
                    "error": err,
                    "original_value": original_features.get(feat, "unknown")
                }
                for feat, err in sorted_errors
            ]
        }


class LIMETextExplainer:
    """LIME explanations for BERT text classifier (C2)"""
    
    def __init__(self, model_pipeline, class_names: List[str]):
        self.model_pipeline = model_pipeline
        self.class_names = class_names
        self.explainer = None
        
        if LIME_AVAILABLE:
            self.explainer = LimeTextExplainer(class_names=class_names)
        
    def explain(self, text: str, num_features: int = 10, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Generate LIME explanation for text classification
        
        Returns:
            Dict with token importance for each predicted class
        """
        if not LIME_AVAILABLE:
            return {"error": "LIME not available"}
        
        # Define prediction function for LIME
        def predict_proba(texts):
            results = []
            for t in texts:
                preds = self.model_pipeline(t)[0]
                # Convert to list of probabilities in order of class_names
                probs = [0.0] * len(self.class_names)
                for pred in preds:
                    label = pred['label']
                    if label in self.class_names:
                        idx = self.class_names.index(label)
                        probs[idx] = pred['score']
                results.append(probs)
            return np.array(results)
        
        # Generate explanation
        exp = self.explainer.explain_instance(
            text, 
            predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Get predictions
        original_probs = predict_proba([text])[0]
        top_class_idx = np.argmax(original_probs)
        
        # Extract word importance for top class
        word_importance = []
        for word, weight in exp.as_list():
            word_importance.append({
                "word": word,
                "importance": float(weight),
                "direction": "positive" if weight > 0 else "negative"
            })
        
        return {
            "text": text,
            "predicted_class": self.class_names[top_class_idx],
            "confidence": float(original_probs[top_class_idx]),
            "word_importance": word_importance,
            "all_class_probs": {
                self.class_names[i]: float(original_probs[i]) 
                for i in range(len(self.class_names))
            }
        }


class AttentionExplainer:
    """BERT attention weights extraction (C2)"""
    
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        
    def explain(self, text: str, layer: int = -1) -> Dict[str, Any]:
        """
        Extract attention weights from BERT
        
        Args:
            text: Input text
            layer: Which layer to extract (-1 for last layer)
        
        Returns:
            Dict with tokens and attention weights
        """
        import torch
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Get attention from specified layer
        # attentions shape: (batch, num_heads, seq_len, seq_len)
        attention = outputs.attentions[layer][0]  # Remove batch dimension
        
        # Average across attention heads
        avg_attention = attention.mean(dim=0)  # (seq_len, seq_len)
        
        # Get attention to [CLS] token (index 0) - represents overall importance
        cls_attention = avg_attention[0, :].cpu().numpy()
        
        # Create token-attention pairs
        token_attention = []
        for i, (token, attn) in enumerate(zip(tokens, cls_attention)):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                token_attention.append({
                    "token": token,
                    "attention": float(attn),
                    "position": i
                })
        
        # Sort by attention weight
        token_attention = sorted(token_attention, key=lambda x: x['attention'], reverse=True)
        
        return {
            "text": text,
            "tokens": tokens,
            "token_attention": token_attention,
            "num_heads": attention.shape[0],
            "sequence_length": len(tokens)
        }


class EmbeddingExplainer:
    """Sentence-BERT embedding interpretation (C3)"""
    
    def __init__(self, sbert_model, isolation_forest, reference_embeddings: np.ndarray = None, reference_texts: List[str] = None):
        self.sbert_model = sbert_model
        self.isolation_forest = isolation_forest
        self.reference_embeddings = reference_embeddings
        self.reference_texts = reference_texts
        
    def explain(self, text: str, top_k_neighbors: int = 5) -> Dict[str, Any]:
        """
        Explain embedding-based decision
        
        Returns:
            Dict with semantic neighbors and isolation score breakdown
        """
        # Get embedding
        embedding = self.sbert_model.encode([text])[0]
        
        # Get isolation score
        iso_score = self.isolation_forest.decision_function([embedding])[0]
        
        explanation = {
            "text": text,
            "isolation_score": float(iso_score),
            "embedding_dim": len(embedding),
            "embedding_norm": float(np.linalg.norm(embedding))
        }
        
        # Find nearest neighbors if reference data available
        if self.reference_embeddings is not None and self.reference_texts is not None:
            distances = np.linalg.norm(self.reference_embeddings - embedding, axis=1)
            nearest_indices = np.argsort(distances)[:top_k_neighbors]
            
            neighbors = []
            for idx in nearest_indices:
                neighbors.append({
                    "text": self.reference_texts[idx],
                    "distance": float(distances[idx]),
                    "similarity": float(1 / (1 + distances[idx]))  # Convert to similarity
                })
            
            explanation["nearest_neighbors"] = neighbors
        
        # Identify most important embedding dimensions
        # Use gradient approximation or perturbation
        dimension_importance = []
        for dim in range(min(10, len(embedding))):  # Top 10 dimensions
            # Perturb this dimension
            perturbed = embedding.copy()
            perturbed[dim] = 0  # Zero out dimension
            
            perturbed_score = self.isolation_forest.decision_function([perturbed])[0]
            importance = abs(iso_score - perturbed_score)
            
            dimension_importance.append({
                "dimension": dim,
                "value": float(embedding[dim]),
                "importance": float(importance)
            })
        
        # Sort by importance
        dimension_importance = sorted(dimension_importance, key=lambda x: x['importance'], reverse=True)
        explanation["important_dimensions"] = dimension_importance[:5]
        
        return explanation


class XAIExplainerFactory:
    """Factory to manage all explainers"""
    
    def __init__(self, models: Dict[str, Any], preprocessors: Dict[str, Any] = None):
        self.models = models
        self.preprocessors = preprocessors or {}
        self.explainers = {}
        self.initialized = False
        
    def initialize(self, background_data: Dict[str, Any] = None):
        """Initialize all explainers"""
        print("\n🔧 Initializing XAI Explainers...")
        
        # C1 SHAP Explainer
        if 'c1_if' in self.models and SHAP_AVAILABLE:
            feature_names = ['eventName', 'eventSource', 'userIdentityType', 'awsRegion']
            self.explainers['c1_shap'] = SHAPExplainer(self.models['c1_if'], feature_names)
            if background_data and 'c1_features' in background_data:
                self.explainers['c1_shap'].initialize(background_data['c1_features'])
                print("  ✓ C1 SHAP Explainer initialized")
        
        # C1 Reconstruction Explainer
        if 'c1_ae' in self.models and 'c1_prep' in self.preprocessors:
            feature_names = ['eventName', 'eventSource', 'userIdentityType', 'awsRegion']
            self.explainers['c1_recon'] = ReconstructionExplainer(
                self.models['c1_ae'],
                self.preprocessors['c1_prep'],
                feature_names
            )
            print("  ✓ C1 Reconstruction Explainer initialized")
        
        # C2 LIME Explainer
        if 'c2_pipe' in self.models and LIME_AVAILABLE:
            class_names = ['LABEL_0', 'LABEL_1', 'LABEL_2']  # Update with actual labels
            self.explainers['c2_lime'] = LIMETextExplainer(self.models['c2_pipe'], class_names)
            print("  ✓ C2 LIME Explainer initialized")
        
        # C2 Attention Explainer
        if 'c2_pipe' in self.models:
            try:
                tokenizer = self.models['c2_pipe'].tokenizer
                model = self.models['c2_pipe'].model
                self.explainers['c2_attention'] = AttentionExplainer(tokenizer, model)
                print("  ✓ C2 Attention Explainer initialized")
            except Exception as e:
                print(f"  ⚠️ C2 Attention Explainer failed: {e}")
        
        # C3 Embedding Explainer
        if 'c3_sbert' in self.models and 'c3_if' in self.models:
            self.explainers['c3_embedding'] = EmbeddingExplainer(
                self.models['c3_sbert'],
                self.models['c3_if']
            )
            print("  ✓ C3 Embedding Explainer initialized")
        
        # C3 SHAP Explainer
        if 'c3_if' in self.models and SHAP_AVAILABLE:
            self.explainers['c3_shap'] = SHAPExplainer(self.models['c3_if'])
            if background_data and 'c3_embeddings' in background_data:
                self.explainers['c3_shap'].initialize(background_data['c3_embeddings'])
                print("  ✓ C3 SHAP Explainer initialized")
        
        self.initialized = True
        print("✅ All XAI Explainers Ready\n")
    
    def get_explainer(self, name: str):
        """Get a specific explainer"""
        return self.explainers.get(name)
    
    def explain_c1(self, scaled_instance: np.ndarray, original_features: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive C1 explanation"""
        explanations = {}
        
        if 'c1_shap' in self.explainers:
            try:
                explanations['shap'] = self.explainers['c1_shap'].explain(scaled_instance)
            except Exception as e:
                explanations['shap'] = {"error": str(e)}
        
        if 'c1_recon' in self.explainers:
            try:
                explanations['reconstruction'] = self.explainers['c1_recon'].explain(
                    scaled_instance, original_features
                )
            except Exception as e:
                explanations['reconstruction'] = {"error": str(e)}
        
        return explanations
    
    def explain_c2(self, text: str) -> Dict[str, Any]:
        """Get comprehensive C2 explanation"""
        explanations = {}
        
        if 'c2_lime' in self.explainers:
            try:
                explanations['lime'] = self.explainers['c2_lime'].explain(text, num_features=8)
            except Exception as e:
                explanations['lime'] = {"error": str(e)}
        
        if 'c2_attention' in self.explainers:
            try:
                explanations['attention'] = self.explainers['c2_attention'].explain(text)
            except Exception as e:
                explanations['attention'] = {"error": str(e)}
        
        return explanations
    
    def explain_c3(self, text: str, embedding: np.ndarray) -> Dict[str, Any]:
        """Get comprehensive C3 explanation"""
        explanations = {}
        
        if 'c3_embedding' in self.explainers:
            try:
                explanations['embedding'] = self.explainers['c3_embedding'].explain(text)
            except Exception as e:
                explanations['embedding'] = {"error": str(e)}
        
        if 'c3_shap' in self.explainers:
            try:
                explanations['shap'] = self.explainers['c3_shap'].explain(embedding.reshape(1, -1))
            except Exception as e:
                explanations['shap'] = {"error": str(e)}
        
        return explanations
