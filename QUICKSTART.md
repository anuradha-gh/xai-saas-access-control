# XAI Pipeline - Quick Installation & Setup Guide

## ✅ Step 1: Verify Core Installation

The XAI pipeline modules are already installed. Test basic imports:

```bash
cd "c:\Users\Anuradha\Downloads\SAAS XAI"
python -c "import xai_explainer, llm_translator, xai_validator; print('Core modules OK!')"
```

## 📦 Step 2: Install Optional Dependencies

For full XAI functionality, install these packages:

```bash
pip install shap>=0.42.0 lime>=0.2.0.1
```

**Note**: If you encounter errors, you can still use the system - it will automatically fall back to basic explanations.

## 🚀 Step 3: Run the Enhanced System

```bash
python XAI_enhanced.py
```

Then choose option 2 to analyze the default log example.

## 🎓 Step 4: Test Different Explanation Styles

In the menu:
- Choose option 5 to select explanation style (technical/executive/compliance/general)
- Choose option 2 again to see explanations in the new style

## 📊 Step 5 (Optional): Run Validation Suite

**Only if SHAP and LIME are installed:**

In the menu:
- Choose option 4 to run XAI validation suite
- This will test fidelity and stability metrics
- Report saved to `validation_report.json`

## 📚 Step 6: Try Examples

```bash
# C1 Anomaly Detection example
python examples/example_c1_explanation.py

# Validation suite example (requires SHAP/LIME)
python examples/example_validation.py
```

## 🐛 Troubleshooting

### "Module not found: shap" or "Module not found: lime"
- **Solution**: Install optional dependencies (Step 2)
- **Alternative**: System will work with template-based explanations

### "Ollama not available"
- **Solution**: Install Ollama from https://ollama.ai/
- **Alternative**: System will use template-based explanations
- **Note**: LLM explanations are optional for enhanced quality

### Unicode encoding errors on Windows
- This only affects console output of emojis
- Does not affect functionality
- Explanations are still generated correctly

## ✨ What's Working Without Optional Dependencies

Even without SHAP/LIME/Ollama:
- ✅ All module imports
- ✅ System loading
- ✅ Basic analysis (anomaly detection, role classification, access decisions)
- ✅ Template-based explanations
- ✅ XAI framework structure

## 🎯 Full Feature Matrix

| Feature | Without Optional Deps | With SHAP/LIME | With SHAP/LIME/Ollama |
|---------|----------------------|----------------|-----------------------|
| Core Analysis | ✅ | ✅ | ✅ |
| SHAP Explanations | ❌ | ✅ | ✅ |
| LIME Explanations | ❌ | ✅ | ✅ |
| Attention Weights | ✅ | ✅ | ✅ |
| Template Explanations | ✅ | ✅ | ✅ |
| LLM Explanations | ❌ | ❌ | ✅ |
| Stakeholder Styles | ⚠️ (basic) | ⚠️ (basic) | ✅ |
| Validation Metrics | ❌ | ✅ | ✅ |

## 📖 Next Steps

1. Review [`README_XAI.md`](README_XAI.md) for comprehensive documentation
2. Check [`walkthrough.md`](file:///C:/Users/Anuradha/.gemini/antigravity/brain/a1cfd31f-2f35-4941-89a7-1849969f62ae/walkthrough.md) for implementation details
3. Explore [`examples/`](examples/) for usage patterns

## 💡 Recommendation

For best experience:
```bash
# Install all optional dependencies
pip install -r requirements_xai.txt

# Install Ollama for LLM-powered explanations
# Visit: https://ollama.ai/
# Then: ollama pull gemma3:1b
```

This gives you access to all XAI features!
