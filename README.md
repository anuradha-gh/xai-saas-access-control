# XAI Interpretability Pipeline for SaaS Access Control

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Explainable AI (XAI) interpretability pipeline for AWS CloudTrail security analysis, featuring SHAP, LIME, attention visualization, LLM-powered explanations, and quantitative validation metrics.

## 🎯 Overview

This system provides interpretability for three classification models used in AI-powered granular access control:
- **C1**: Anomaly Detection (Autoencoder + Isolation Forest)
- **C2**: Role Classification (BERT)
- **C3**: Access Decision (Sentence-BERT + Isolation Forest)

### Key Features

✨ **Multiple XAI Techniques**
- SHAP values for tree-based models
- LIME text explanations for BERT
- Attention weight visualization
- Reconstruction error analysis
- Embedding-based semantic explanations

🤖 **LLM Translation Layer**
- Converts numerical XAI outputs to natural language
- Stakeholder-specific explanations (Technical, Executive, Compliance, General)
- Powered by Ollama (local LLM) with fallback to templates

✅ **Validation Framework**
- **Fidelity Metrics**: Model approximation, perturbation sensitivity
- **Stability Metrics**: Similar input consistency, explanation variance
- Quantitative pass/fail thresholds based on research best practices

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/xai-saas-access-control.git
cd xai-saas-access-control
```

### 2. Install Dependencies

**Core Dependencies:**
```bash
pip install numpy pandas scikit-learn scipy tensorflow transformers sentence-transformers joblib
```

**XAI Dependencies:**
```bash
pip install shap lime
```

**Optional (for LLM explanations):**
```bash
# Install Ollama from https://ollama.ai/
ollama pull gemma3:1b
pip install ollama
```

Or install everything:
```bash
pip install -r requirements_xai.txt
```

### 3. Download Models

> ⚠️ **Important**: Model files are not included in this repository due to size constraints.

You need to download or train the following models and place them in the root directory:
- `autoencoder.h5` - Autoencoder for C1
- `isolation_forest.joblib` - Isolation Forest for C1
- `trained_role_classifier/` - Fine-tuned BERT for C2
- `c3_unsupervised_aws_model/` - Sentence-BERT + Isolation Forest for C3
- `flaws_cloudtrail00.json` - CloudTrail training data

See [`MODEL_SETUP.md`](MODEL_SETUP.md) for details on model training/downloading.

## 🚀 Quick Start

### Basic Usage

```bash
python XAI_enhanced.py
```

This opens an interactive menu where you can:
1. Analyze CloudTrail logs with XAI explanations
2. Test with default example
3. Chat with AI about analysis results
4. Run XAI validation suite
5. Change explanation style (technical/executive/compliance/general)

### Programmatic Usage

```python
from XAI_enhanced import *

# Load system
load_system()

# Analyze a log
log = {
    "eventName": "DeleteBucket",
    "eventSource": "s3.amazonaws.com",
    "userIdentity": {"type": "Root", "userName": "root_account"},
    "awsRegion": "us-west-2"
}

# Get executive-level explanation
run_full_analysis_with_xai(json.dumps(log), stakeholder_type='executive')
```

## 📖 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Installation and setup guide
- **[README_XAI.md](README_XAI.md)** - Comprehensive XAI documentation
- **[examples/](examples/)** - Usage examples and demos

## 🏗️ Architecture

```
┌─────────────────┐
│ CloudTrail Logs │
└────────┬────────┘
         │
    ┌────▼─────┐
    │ C1: C2: C3│  (Three Classification Models)
    └────┬─────┘
         │
    ┌────▼─────────────┐
    │ XAI Explainers   │  (SHAP, LIME, Attention, etc.)
    │ xai_explainer.py │
    └────┬─────────────┘
         │
    ┌────▼──────────────┐
    │ LLM Translator    │  (Stakeholder-specific explanations)
    │ llm_translator.py │
    └────┬──────────────┘
         │
    ┌────▼────────────┐
    │ XAI Validator   │  (Fidelity & Stability metrics)
    │ xai_validator.py│
    └─────────────────┘
```

## 📂 Project Structure

```
xai-saas-access-control/
├── xai_explainer.py          # Core XAI techniques
├── llm_translator.py         # LLM translation layer
├── xai_validator.py          # Validation framework
├── XAI_enhanced.py           # Enhanced system with XAI
├── XAI.py                    # Original system (unmodified)
├── requirements_xai.txt      # Python dependencies
├── examples/
│   ├── example_c1_explanation.py
│   └── example_validation.py
├── README.md                 # This file
├── README_XAI.md            # Detailed XAI documentation
├── QUICKSTART.md            # Setup guide
└── .gitignore               # Git ignore patterns
```

## 🎓 XAI Techniques by Component

### C1: Anomaly Detection

**Techniques:**
- **SHAP**: Explains latent feature contributions to anomaly score
- **Reconstruction Error**: Identifies which input features are most unusual

**Example Output:**
```
🔍 SHAP Feature Importance:
  ↑ eventName: 0.0234 (DeleteBucket is highly anomalous)
  ↓ userIdentityType: -0.0156 (Root type expected)

🔧 Reconstruction Errors:
  • eventName: 0.3542 (Very high - action is rare)
```

### C2: Role Classification

**Techniques:**
- **LIME**: Token-level importance for text classification
- **Attention Weights**: BERT attention patterns

**Example Output:**
```
📝 Important Words (LIME):
  - 'DeleteBucket': 0.452 (strongly indicates Developer role)
  - 's3': 0.318 (service context)
  - 'Root': -0.281 (conflicts with Developer classification)
```

### C3: Access Decision

**Techniques:**
- **Embedding Analysis**: Semantic similarity to known patterns
- **SHAP**: Explains embedding dimension contributions

**Example Output:**
```
🔍 Similar Past Activities:
  - "DeleteBucket by Developer" (92% similar, was DENIED)
  - "DeleteBucket by Admin" (88% similar, was APPROVED)
  
Decision: ADMIN REVIEW (similarity conflict detected)
```

## ✅ Validation Metrics

### Fidelity (Do explanations reflect model behavior?)

- **Perturbation Sensitivity**: Threshold > 0.01
- **Model Approximation**: R² threshold > 0.7

### Stability (Are explanations consistent?)

- **Jaccard Similarity**: Threshold > 0.5
- **Explanation Variance**: Threshold < 0.1

Run validation:
```bash
python examples/example_validation.py
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- **SHAP**: [Lundberg & Lee (2017)](https://arxiv.org/abs/1705.07874)
- **LIME**: [Ribeiro et al. (2016)](https://arxiv.org/abs/1602.04938)
- **Attention Visualization**: [Clark et al. (2019)](https://arxiv.org/abs/1906.04341)
- **XAI Validation**: [Robnik-Šikonja & Bohanec (2018)](https://www.sciencedirect.com/science/article/pii/S0950705118300285)

## 🙋 Support

For questions or issues:
- 📧 Email: anuradhalakshmanbandara@proton.me
- 🐛 Issues: [GitHub Issues](https://github.com/YOUR_USERNAME/xai-saas-access-control/issues)
- 📖 Documentation: [README_XAI.md](README_XAI.md)

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Built with ❤️ for explainable AI in cybersecurity**
