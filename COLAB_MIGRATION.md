# Google Colab Migration Guide

## Overview

This guide shows how to run the XAI pipeline in Google Colab **without Ollama/Gemma 3 1B**, focusing purely on XAI validation metrics.

## Why Colab Without LLM?

✅ **Advantages:**
- Free GPU/TPU for faster model inference
- No local setup needed
- Easy sharing and collaboration
- Focus on **quantitative XAI metrics** (SHAP, LIME scores)
- LLM explanations are optional - you still get numerical insights

## Migration Steps

### Step 1: Upload Models to Google Drive

Since model files are large, use Google Drive:

1. Upload these files to a Google Drive folder:
   - `autoencoder.h5` (~800 KB)
   - `isolation_forest.joblib` (~857 KB)  
   - `trained_role_classifier/` (BERT model)
   - `c3_unsupervised_aws_model/`
   - `flaws_cloudtrail00.json` (or a smaller subset)

2. Note the folder path (e.g., `/content/drive/MyDrive/SAAS_XAI/`)

### Step 2: Install Dependencies in Colab

```python
# Install XAI libraries
!pip install shap lime sentence-transformers transformers joblib scikit-learn scipy

# No Ollama needed!
```

### Step 3: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Set paths to your models
MODEL_PATH = '/content/drive/MyDrive/SAAS_XAI/'
```

### Step 4: Disable LLM in Code

Modify the configuration:

```python
# In your Colab notebook
USE_LLM = False  # Disable Ollama

XAI_CONFIG = {
    'enable_xai': True,
    'default_stakeholder': 'general',
    'enable_validation': True,  # Keep validation ON
    'num_shap_samples': 100,
    'num_lime_samples': 1000,
}
```

### Step 5: Use Template Explanations

Without LLM, the system automatically uses template-based explanations:

```python
# Fallback explanations (from llm_translator.py)
if not USE_LLM:
    return f"Risk Score: {risk_score}/100. Anomaly detected in {feature_name}."
```

These are **less natural** but still contain **all the numerical XAI data**.

## What You Get Without LLM

### ✅ Full XAI Capabilities:
- **SHAP values**: Feature importance for each prediction
- **LIME explanations**: Token-level importance for BERT
- **Attention weights**: BERT attention patterns
- **Reconstruction errors**: Autoencoder analysis
- **Embedding similarity**: Semantic neighbors

### ✅ Complete Validation:
- **Fidelity metrics**: Perturbation sensitivity, R² scores
- **Stability metrics**: Jaccard similarity, variance
- **Pass/fail thresholds**: Quantitative quality checks
- **Validation reports**: JSON + text output

### ❌ What You Lose:
- Natural language explanations (stakeholder-specific prose)
- LLM-powered chatbot
- Comparative explanations ("Why A instead of B?")

## Colab Notebook Structure

I'll create a complete notebook for you with these sections:

1. **Setup** - Install packages, mount Drive
2. **Load Models** - Import from Drive
3. **Initialize XAI** - Create explainers (SHAP, LIME)
4. **Run Analysis** - Analyze CloudTrail logs
5. **Validate XAI** - Run fidelity/stability tests
6. **Visualize Results** - Plot SHAP values, attention

## Alternative: Use Google's Gemini API

If you want LLM explanations in Colab, use **Google's Gemini API** (free tier available):

```python
import google.generativeai as genai

genai.configure(api_key='YOUR_API_KEY')
model = genai.GenerativeModel('gemini-pro')

# Use instead of Ollama
response = model.generate_content(prompt)
```

This gives you:
- ✅ Natural language explanations
- ✅ Free tier (60 requests/minute)
- ✅ No local model download
- ✅ Better quality than Gemma 3 1B

## Performance Comparison

| Aspect | Local (Ollama) | Colab (No LLM) | Colab (Gemini API) |
|--------|---------------|----------------|-------------------|
| Setup Time | Medium | Fast | Fast |
| XAI Speed | Fast | **Fastest** | Fast |
| Explanations | Natural | Template | Natural |
| Cost | Free | Free | Free (limits) |
| GPU Access | Local only | **Free GPU** | Free GPU |
| Validation | ✅ | ✅ | ✅ |

## Ready-Made Colab Notebook

I'll create `XAI_Colab.ipynb` with:
- One-click setup
- Pre-configured paths
- Validation-focused workflow
- Visualization cells
- No LLM dependencies

Would you like me to create this notebook now?
