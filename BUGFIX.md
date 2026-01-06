# Bug Fix: Feature Dimension Mismatch

## Issue
When initializing SHAP explainer, the system was passing 451-dimensional one-hot encoded features instead of the 9-dimensional latent+MSE features that the Isolation Forest expects.

## Error
```
ValueError: X has 451 features, but IsolationForest is expecting 9 features as input.
```

## Root Cause
The Isolation Forest in C1 is trained on:
- Latent features from autoencoder bottleneck (8 dimensions)  
- MSE (reconstruction error) (1 dimension)
- **Total: 9 features**

But we were passing the raw one-hot encoded features (451 dimensions) to SHAP.

## Fix Applied

### 1. XAI_enhanced.py (lines 37-42)
Extract latent + MSE features for SHAP background:
```python
# IMPORTANT: For C1 SHAP, we need latent features + MSE
latent = state['models']['c1_enc'].predict(X_scaled, verbose=0)
recon = state['models']['c1_ae'].predict(X_scaled, verbose=0)
mse = np.mean((X_scaled - recon)**2, axis=1).reshape(-1, 1)
c1_features_for_shap = np.hstack([latent, mse])
```

### 2. xai_explainer.py (lines 383-390)
Update feature names for latent space:
```python
# Feature names for latent features + MSE
latent_dim = background_data['c1_features'].shape[1] - 1
feature_names = [f'latent_{i}' for i in range(latent_dim)] + ['mse']
```

### 3. XAI_enhanced.py run_analysis_c1_with_xai (lines 87-95)
Pass correct features to SHAP:
```python
# Get latent features + MSE (same as Isolation Forest input)
latent = state['models']['c1_enc'].predict(scaled, verbose=0)
recon = state['models']['c1_ae'].predict(scaled, verbose=0)
mse = np.mean((scaled - recon)**2, axis=1).reshape(-1, 1)
features_for_if = np.hstack([latent, mse])

# Pass features_for_if to SHAP (not scaled)
explanations = state['xai_explainer'].explain_c1(features_for_if, p_log)
```

## Status
✅ Fixed in local files
⏳ Ready to commit to GitHub

## Testing
System should now start without the dimension mismatch error.
