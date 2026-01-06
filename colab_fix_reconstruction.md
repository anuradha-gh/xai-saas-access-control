# Quick Fix for Colab Reconstruction Error

## Problem
The reconstruction explainer returns data but the 'feature_errors' key is missing.

## Solution
Update the Colab notebook Cell #7 (Test XAI) to handle the actual output format:

```python
# Example CloudTrail log
test_log = {
    "eventTime": "2017-02-12T21:30:56Z",
    "eventSource": "s3.amazonaws.com",
    "eventName": "DeleteBucket",
    "awsRegion": "us-west-2",
    "sourceIPAddress": "AWS Internal",
    "userIdentity": {
        "type": "Root",
        "userName": "root_account"
    }
}

# Prepare features
p_log = parse_log_c1(test_log)
processed = state['preprocessors']['c1_prep'].transform(pd.DataFrame([p_log])).toarray()
scaled = state['preprocessors']['c1_scaler'].transform(processed)

# Extract latent + MSE
latent = state['models']['c1_enc'].predict(scaled, verbose=0)
recon = state['models']['c1_ae'].predict(scaled, verbose=0)
mse = np.mean((scaled - recon)**2, axis=1).reshape(-1, 1)
features_for_if = np.hstack([latent, mse])

# Get XAI explanation
xai_result = state['xai_explainer'].explain_c1(features_for_if, p_log)

print("\n📊 XAI EXPLANATION:")
print("="*70)

# Display SHAP values
if 'shap' in xai_result:
    print("\n🔍 SHAP Feature Importance:")
    for feat in xai_result['shap']['feature_importance'][:5]:
        print(f"  - {feat['feature']}: {feat['shap_value']:.4f} (importance: {feat['importance']:.4f})")

# Display reconstruction errors (FIXED)
if 'reconstruction' in xai_result:
    print("\n🔧 Reconstruction Errors:")
    recon_data = xai_result['reconstruction']
    
    # Check what keys are available
    if 'feature_errors' in recon_data:
        for feat in recon_data['feature_errors'][:3]:
            print(f"  - {feat['feature']}: {feat['error']:.4f}")
    elif 'total_error' in recon_data:
        # Fallback: just show total error
        print(f"  Total MSE: {recon_data['total_error']:.4f}")
    else:
        # Show all available data
        print("  Available reconstruction data:")
        for key, value in recon_data.items():
            if isinstance(value, (int, float)):
                print(f"    - {key}: {value:.4f}")
            else:
                print(f"    - {key}: {type(value).__name__}")

print("\n" + "="*70)
```

## Why This Happens

The `ReconstructionExplainer` in `xai_explainer.py` needs the original categorical feature names to map reconstruction errors back to features. In Colab, if you pass only latent+MSE features (not the original parsed log), it can't create the detailed feature error mapping.

## Alternative: Skip Reconstruction Display

If you only care about SHAP values (which are working perfectly), just remove the reconstruction section:

```python
# Get XAI explanation
xai_result = state['xai_explainer'].explain_c1(features_for_if, p_log)

print("\n📊 XAI EXPLANATION:")
print("="*70)

# Display SHAP values (this works!)
if 'shap' in xai_result:
    print("\n🔍 SHAP Feature Importance:")
    for feat in xai_result['shap']['feature_importance'][:5]:
        direction = "↑ Increases risk" if feat['shap_value'] > 0 else "↓ Decreases risk"
        print(f"  • {feat['feature']}: {feat['shap_value']:.4f}")
        print(f"     {direction}, Importance: {feat['importance']:.4f}")
```

## Best Solution: Focus on SHAP

For validation purposes, **SHAP values are the most important**. They show:
- ✅ Which latent dimensions drive anomaly detection
- ✅ MSE contribution to the score
- ✅ Direction of influence (positive/negative)

Reconstruction errors are secondary - they show which **original features** (before encoding) are unusual, but SHAP on latent features is more directly tied to the model's decision.

Your current output shows:
- ✅ **MSE has highest importance** (-0.0820) → Reconstruction quality matters most
- ✅ **Latent dimensions vary** → Shows model uses different aspects of latent space
- ✅ **Negative values** → These features push score toward "normal" (lower risk)

This is exactly what you need for validation! 🎯
