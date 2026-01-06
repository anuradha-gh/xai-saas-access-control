"""
Quick Test Script - Verify XAI Pipeline Installation
Tests basic functionality without running full analysis
"""

import sys

print("="*70)
print("XAI PIPELINE - INSTALLATION VERIFICATION")
print("="*70)

# Test 1: Import core modules
print("\n📦 Test 1: Checking imports...")
try:
    from xai_explainer import XAIExplainerFactory, SHAPExplainer, LIMETextExplainer
    print("  ✅ xai_explainer.py imported successfully")
except ImportError as e:
    print(f"  ❌ xai_explainer.py import failed: {e}")
    sys.exit(1)

try:
    from llm_translator import LLMTranslator, StakeholderType
    print("  ✅ llm_translator.py imported successfully")
except ImportError as e:
    print(f"  ❌ llm_translator.py import failed: {e}")
    sys.exit(1)

try:
    from xai_validator import XAIValidator, FidelityValidator, StabilityValidator
    print("  ✅ xai_validator.py imported successfully")
except ImportError as e:
    print(f"  ❌ xai_validator.py import failed: {e}")
    sys.exit(1)

# Test 2: Check dependencies
print("\n📚 Test 2: Checking dependencies...")
dependencies = {
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'sklearn': 'scikit-learn',
    'scipy': 'SciPy',
}

optional_dependencies = {
    'shap': 'SHAP',
    'lime': 'LIME',
    'ollama': 'Ollama (for LLM)'
}

for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"  ✅ {name} installed")
    except ImportError:
        print(f"  ❌ {name} NOT installed - required!")
        sys.exit(1)

for module, name in optional_dependencies.items():
    try:
        __import__(module)
        print(f"  ✅ {name} installed")
    except ImportError:
        print(f"  ⚠️  {name} NOT installed - some features may be disabled")

# Test 3: Test LLM Translator
print("\n🤖 Test 3: Testing LLM Translator...")
try:
    translator = LLMTranslator(use_ollama=False)  # Test without Ollama
    print("  ✅ LLM Translator initialized (template mode)")
    
    # Test translation
    test_data = {
        'risk_score': 75,
        'category': 'HIGH',
        'log_context': {'eventName': 'DeleteBucket'}
    }
    
    explanation = translator.translate("C1", test_data, StakeholderType.GENERAL)
    print(f"  ✅ Generated explanation: {explanation[:50]}...")
    
except Exception as e:
    print(f"  ❌ LLM Translator test failed: {e}")
    sys.exit(1)

# Test 4: Test Validator (without actual models)
print("\n🔬 Test 4: Testing Validator...")
try:
    validator = XAIValidator()
    print("  ✅ XAI Validator initialized")
except Exception as e:
    print(f"  ❌ Validator initialization failed: {e}")
    sys.exit(1)

# Test 5: Check stakeholder types
print("\n👥 Test 5: Checking stakeholder types...")
try:
    stakeholders = [StakeholderType.TECHNICAL, StakeholderType.EXECUTIVE, 
                   StakeholderType.COMPLIANCE, StakeholderType.GENERAL]
    print(f"  ✅ All stakeholder types available: {[s.value for s in stakeholders]}")
except Exception as e:
    print(f"  ❌ Stakeholder types error: {e}")

# Summary
print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nYour XAI pipeline is ready to use!")
print("\nNext steps:")
print("  1. Install optional dependencies if needed:")
print("     pip install shap lime")
print("  2. Install Ollama for LLM-powered explanations:")
print("     https://ollama.ai/")
print("  3. Run the enhanced system:")
print("     python XAI_enhanced.py")
print("  4. Try the examples:")
print("     python examples/example_c1_explanation.py")
print("\n" + "="*70)
