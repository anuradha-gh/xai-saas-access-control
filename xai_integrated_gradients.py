"""
Integrated Gradients Explainer for BERT (C2)
Replaces LIME with gradient-based attribution for faster, more accurate explanations
"""

import torch
import numpy as np
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

try:
    from captum.attr import IntegratedGradients, LayerIntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("⚠️ Captum not installed. Run: pip install captum")


class IntegratedGradientsExplainer:
    """
    Integrated Gradients for BERT text classification
    
    Advantages over LIME:
    - 5-10x faster (no sampling needed)
    - More accurate (gradient-based)
    - Better for transformers (direct attribution)
    - Theoretically grounded (satisfies sensitivity and implementation invariance)
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize Integrated Gradients explainer
        
        Args:
            model: HuggingFace BERT model
            tokenizer: HuggingFace tokenizer
        """
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum required for Integrated Gradients")
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Set model to eval mode
        self.model.eval()
        
        # Create Integrated Gradients instance
        self.ig = IntegratedGradients(self.forward_func)
        
    def forward_func(self, input_ids, attention_mask=None):
        """
        Forward function for IG
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
        
        Returns:
            Model logits
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def explain(self, text: str, target_class: int = None, n_steps: int = 50) -> Dict[str, Any]:
        """
        Generate Integrated Gradients explanation
        
        Args:
            text: Input text
            target_class: Target class index (None = predicted class)
            n_steps: Number of IG steps (default: 50)
        
        Returns:
            Dict with token attributions and predictions
        """
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
        
        # Use predicted class if target not specified
        if target_class is None:
            target_class = predicted_class
        
        # Create baseline (all PAD tokens)
        baseline_ids = torch.zeros_like(input_ids)
        baseline_ids[:] = self.tokenizer.pad_token_id
        
        # Compute attributions
        attributions = self.ig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            target=target_class,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
            return_convergence_delta=False
        )
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Sum attributions across embedding dimension
        attr_values = attributions.sum(dim=-1).squeeze().cpu().numpy()
        
        # Create token-attribution pairs (exclude special tokens)
        token_attributions = []
        for i, (token, attr) in enumerate(zip(tokens, attr_values)):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                token_attributions.append({
                    'token': token,
                    'attribution': float(attr),
                    'position': i,
                    'absolute_attribution': float(abs(attr))
                })
        
        # Sort by absolute attribution
        token_attributions.sort(key=lambda x: x['absolute_attribution'], reverse=True)
        
        return {
            'text': text,
            'predicted_class': int(predicted_class),
            'confidence': float(probs[0][predicted_class]),
            'all_class_probs': probs[0].cpu().numpy().tolist(),
            'token_attributions': token_attributions,
            'method': 'Integrated Gradients',
            'n_steps': n_steps
        }
    
    def visualize_attributions(self, explanation: Dict[str, Any], top_k: int = 10) -> str:
        """
        Create text visualization of attributions
        
        Args:
            explanation: Output from explain()
            top_k: Number of top tokens to show
        
        Returns:
            Formatted string visualization
        """
        token_attrs = explanation['token_attributions'][:top_k]
        
        viz = f"\n{'='*60}\n"
        viz += f"Integrated Gradients Explanation\n"
        viz += f"{'='*60}\n\n"
        viz += f"Predicted Class: {explanation['predicted_class']}\n"
        viz += f"Confidence: {explanation['confidence']:.2%}\n\n"
        viz += f"Top {top_k} Contributing Tokens:\n"
        viz += f"{'-'*60}\n"
        
        for i, token_attr in enumerate(token_attrs, 1):
            token = token_attr['token']
            attr = token_attr['attribution']
            sign = '+' if attr > 0 else ''
            
            # Visual bar
            bar_length = int(abs(attr) * 50)
            bar = '█' * min(bar_length, 50)
            
            viz += f"{i:2d}. {token:20s} {sign}{attr:7.4f} {bar}\n"
        
        viz += f"{'='*60}\n"
        
        return viz


# Example usage:
"""
from xai_integrated_gradients import IntegratedGradientsExplainer

# Initialize
ig_explainer = IntegratedGradientsExplainer(bert_model, tokenizer)

# Explain
explanation = ig_explainer.explain("Root user performed DeleteBucket on s3")

# Top tokens
for token in explanation['token_attributions'][:5]:
    print(f"{token['token']}: {token['attribution']:.4f}")

# Visualize
print(ig_explainer.visualize_attributions(explanation, top_k=10))
"""
