# Model Setup Guide

This document explains how to set up the required models for the XAI pipeline.

## 📦 Required Models

The following models are needed but **not included in this repository** due to size constraints:

1. **`autoencoder.h5`** (~800 KB) - Autoencoder for C1 anomaly detection
2. **`isolation_forest.joblib`** (~857 KB) - Isolation Forest for C1
3. **`trained_role_classifier/`** - Fine-tuned BERT model for C2 role classification
4. **`c3_unsupervised_aws_model/`** - Sentence-BERT + Isolation Forest for C3
5. **`flaws_cloudtrail00.json`** (~100 MB) - CloudTrail training data

## 🔧 Option 1: Download Pre-trained Models

If you have access to pre-trained models:

1. Download models from your model repository or cloud storage
2. Place them in the project root directory:
   ```
   xai-saas-access-control/
   ├── autoencoder.h5
   ├── isolation_forest.joblib
   ├── trained_role_classifier/
   ├── c3_unsupervised_aws_model/
   └── flaws_cloudtrail00.json
   ```

3. Update paths in `XAI.py` if needed:
   ```python
   CONFIG = {
       'c1_autoencoder': r'autoencoder.h5',
       'c1_iso_forest': r'isolation_forest.joblib',
       # ... etc
   }
   ```

## 🏋️ Option 2: Train Your Own Models

### C1: Autoencoder + Isolation Forest

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.ensemble import IsolationForest
import joblib

# 1. Train Autoencoder
# ... (train on your CloudTrail data)
autoencoder.save('autoencoder.h5')

# 2. Train Isolation Forest on latent features
# ... (use autoencoder's latent space + reconstruction error)
joblib.dump(iso_forest, 'isolation_forest.joblib')
```

### C2: BERT Role Classifier

```python
from transformers import AutoModelForSequenceClassification, Trainer

# Fine-tune BERT on your labeled CloudTrail data
# ... (training code)

# Save model
model.save_pretrained('trained_role_classifier')
tokenizer.save_pretrained('trained_role_classifier')
```

### C3: Sentence-BERT + Isolation Forest

```python
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest

# 1. Use or fine-tune Sentence-BERT
sbert = SentenceTransformer('all-MiniLM-L6-v2')
# ... (optional fine-tuning)
sbert.save('c3_unsupervised_aws_model/sbert_model')

# 2. Train Isolation Forest on embeddings
# ... (training code)
joblib.dump(iso_forest, 'c3_unsupervised_aws_model/isolation_forest.joblib')
```

## 📊 Training Data

**CloudTrail Data Format:**
```json
{
  "Records": [
    {
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
  ]
}
```

You can:
- Use your own CloudTrail logs
- Use synthetic data generation
- Use publicly available datasets (e.g., CloudTrail from flaws.cloud)

## 🐙 Git LFS for Large Files (Recommended)

If you want to include models in your repository:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.h5"
git lfs track "*.joblib"
git lfs track "*.json"
git lfs track "trained_role_classifier/**"
git lfs track "c3_unsupervised_aws_model/**"

# Add .gitattributes
git add .gitattributes

# Commit and push
git add .
git commit -m "Add models via Git LFS"
git push
```

> ⚠️ **Note**: Git LFS may have storage/bandwidth limits on free plans.

## ✅ Verify Setup

After placing models, verify:

```bash
python -c "import os; print('✅ All models found!' if all(os.path.exists(f) for f in ['autoencoder.h5', 'isolation_forest.joblib']) else '❌ Missing models')"
```

Or run the system:
```bash
python XAI_enhanced.py
```

If models load successfully, you'll see:
```
🧠 Loading C1 Autoencoder...
🌲 Loading C1 IsolationForest...
🤖 Loading C2 BERT model...
📝 Loading C3 Sentence-BERT...
✅ SYSTEM READY!
```

## 🆘 Troubleshooting

**Error: "File not found"**
- Check file paths in `CONFIG` dictionary in `XAI.py`
- Ensure files are in the correct directory

**Error: "Model version mismatch"**
- Ensure TensorFlow/scikit-learn versions match training environment
- See `requirements_xai.txt` for compatible versions

**Error: "Out of memory"**
- Models may be too large for your system
- Consider using smaller model variants
- Use cloud compute for inference

## 📧 Need Help?

If you need assistance with model setup:
- Open an issue on GitHub
- Check the main README.md
- Review XAI.py configuration section
