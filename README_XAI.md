# XAI Interpretability Pipeline for SaaS Access Control

This directory contains a comprehensive Explainable AI (XAI) interpretability pipeline for three classification models used in AWS CloudTrail analysis.

## 🎯 Overview

The system provides:
1. **XAI Techniques**: SHAP, LIME, attention weights, reconstruction error analysis, and embedding explanations
2. **LLM Translation Layer**: Converts numerical XAI outputs to stakeholder-specific natural language
3. **Validation Framework**: Quantitative metrics for fidelity and stability of explanations

## 📁 Files

### Core Modules
- **`xai_explainer.py`**: Core XAI techniques (SHAP, LIME, attention, reconstruction, embedding)
- **`llm_translator.py`**: LLM-powered translation to natural language
- **`xai_validator.py`**: Fidelity and stability validation metrics
- **`XAI_enhanced.py`**: Enhanced wrapper integrating XAI with existing system

### Supporting Files
- **`requirements_xai.txt`**: Additional dependencies for XAI
- **`examples/`**: Example scripts demonstrating functionality
- **`XAI.py`**: Original system (unchanged)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_xai.txt
```

Required packages:
- `shap>=0.42.0` - For SHAP explanations
- `lime>=0.2.0.1` - For LIME text explanations  
- `scikit-learn>=1.3.0` - For validation metrics
- `scipy>=1.10.0` - For statistical calculations

### 2. Run Enhanced System

```bash
python XAI_enhanced.py
```

This loads all models and initializes the XAI pipeline. You can then:
- Analyze CloudTrail logs with XAI explanations
- Switch between stakeholder views (technical/executive/compliance/general)
- Run validation suite to check XAI quality

### 3. Run Examples

```bash
# C1 Anomaly Detection example
python examples/example_c1_explanation.py

# Validation suite example
python examples/example_validation.py
```

## 🔍 Components

### C1: Anomaly Detection (Autoencoder + Isolation Forest)

**XAI Techniques:**
- **SHAP**: Explains which latent features contribute to anomaly score
- **Reconstruction Error**: Shows which input features are most unusual

**Example Output:**
```
🔍 SHAP Feature Importance:
  ↑ eventName: 0.0234 (Importance: 0.0234)
  ↓ userIdentityType: -0.0156 (Importance: 0.0156)

🔧 Reconstruction Errors:
  • eventName: 0.3542 (DeleteBucket is rare)
  • awsRegion: 0.0823 (us-west-2 is common)
```

### C2: Role Classification (BERT)

**XAI Techniques:**
- **LIME**: Token-level importance for classification
- **Attention Weights**: BERT attention patterns

**Example Output:**
```
📝 Important Words (LIME):
  - 'DeleteBucket': 0.452 (positive)
  - 's3': 0.318 (positive)
  - 'Root': -0.281 (negative)
```

### C3: Access Decision (Sentence-BERT + Isolation Forest)

**XAI Techniques:**
- **Embedding Analysis**: Semantic similarity to known patterns
- **SHAP**: Explains which embedding dimensions drive decisions

**Example Output:**
```
🔍 Similar Past Activities:
  - "DeleteBucket by Developer in us-east-1" (similarity: 92%)
  - "DeleteBucket by Admin in us-west-2" (similarity: 88%)
```

## 👥 Stakeholder-Specific Explanations

The LLM translator provides tailored explanations:

### Technical (ML Engineers)
Focus on model internals, feature importance, statistical measures

### Executive (C-Suite)
High-level summaries, business impact, actionable decisions

### Compliance (Auditors)
Policy adherence, regulatory requirements, audit trails

### General (Security Team)
Balanced explanations accessible to non-experts

**Usage:**
```python
# In menu,select option 5 to change explanation style
# Or programmatically:
state['llm_translator'].translate("C1", xai_data, StakeholderType.EXECUTIVE)
```

## ✅ Validation Framework

### Fidelity Metrics

**Model Approximation Fidelity:**
- Measures how well XAI approximates actual model
- Metric: R² score between original and surrogate model
- Threshold: R² > 0.7

**Perturbation Fidelity:**
- Tests if perturbing important features changes predictions
- Metric: Mean prediction sensitivity
- Threshold: Sensitivity > 0.01

### Stability Metrics

**Similar Input Consistency:**
- Explanations should be stable for similar inputs
- Metric: Jaccard similarity of top-k features
- Threshold: Jaccard > 0.5

**Explanation Variance:**
- Measures robustness to input noise
- Metric: Variance of feature importance under perturbations
- Threshold: Variance < 0.1

### Running Validation

```python
# Enable in config
XAI_CONFIG['enable_validation'] = True

# Run suite
report = run_validation_suite()

# Results saved to validation_report.json
```

## 📊 Configuration

Edit `XAI_CONFIG` in `XAI_enhanced.py`:

```python
XAI_CONFIG = {
    'enable_xai': True,  # Enable/disable XAI
    'default_stakeholder': 'general',  # Default explanation style
    'enable_validation': False,  # Enable validation (slower)
    'num_shap_samples': 100,  # SHAP background samples
    'num_lime_samples': 1000,  # LIME perturbation samples
}
```

## 🎓 Technical Details

### SHAP (SHapley Additive exPlanations)
- Uses game-theoretic approach to attribute predictions to features
- Provides consistent, locally accurate feature attributions
- We use `KernelExplainer` for model-agnostic explanations

### LIME (Local Interpretable Model-agnostic Explanations)
- Trains local surrogate model around prediction
- For text, perturbs by masking tokens
- Fast and intuitive for stakeholders

### Attention Mechanisms
- Extracts BERT attention weights from final layer
- Averaged across attention heads
- Shows which tokens the model focuses on

### Validation Approach
- **Fidelity**: Do explanations accurately reflect model behavior?
- **Stability**: Are explanations consistent for similar inputs?
- Uses statistical tests (R², Spearman correlation, Jaccard similarity)

## 🐛 Troubleshooting

### "SHAP/LIME not installed"
```bash
pip install shap lime
```

### "Ollama not available"
- System falls back to template-based explanations
- To use LLM: Install Ollama and pull model:
```bash
ollama pull gemma3:1b
```

### Validation takes too long
- Reduce test samples in `run_validation_suite()`
- Disable validation: `XAI_CONFIG['enable_validation'] = False`

### Out of memory
- Reduce `num_shap_samples` and `num_lime_samples` in config
- Use smaller background dataset for SHAP

## 📚 References

- **SHAP**: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
- **LIME**: Ribeiro et al. (2016) "Why Should I Trust You?: Explaining the Predictions of Any Classifier"
- **Validation**: Robnik-Šikonja & Bohanec (2018) "Perturbation-Based Explanations of Prediction Models"

## 📝 Example Usage

```python
from XAI_enhanced import *

# Load system
load_system()

# Analyze log
log = {...}  # CloudTrail JSON
run_full_analysis_with_xai(json.dumps(log), stakeholder_type='executive')

# Run validation
XAI_CONFIG['enable_validation'] = True
report = run_validation_suite()
```

## 🤝 Contributing

To extend the XAI pipeline:

1. **Add new explainer**: Extend `xai_explainer.py` with new class
2. **Add validation metric**: Add to `FidelityValidator` or `StabilityValidator`
3. **Add stakeholder type**: Extend `StakeholderType` enum and add prompts

## 📄 License

Same as original XAI.py project

## 🙋 Support

For issues or questions about the XAI pipeline, please review:
1. This README
2. Example scripts in `examples/`
3. Implementation plan: `implementation_plan.md`
