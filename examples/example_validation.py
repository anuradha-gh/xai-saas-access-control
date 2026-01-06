"""
Example: Running XAI Validation Suite
Demonstrates fidelity and stability metrics
"""

import json
import sys
sys.path.append('..')

from XAI_enhanced import *

def demo_validation():
    """Demo validation suite"""
    print("="*70)
    print("XAI VALIDATION SUITE DEMONSTRATION")
    print("="*70)
    
    # Enable validation in config
    XAI_CONFIG['enable_validation'] = True
    
    # Load system
    print("\nLoading system with validation enabled...")
    load_system()
    
    if 'xai_validator' not in state:
        print("❌ Validation not enabled. Check configuration.")
        return
    
    print("\n📊 Running comprehensive validation tests...")
    print("   This may take a few minutes...\n")
    
    # Run validation suite
    report = run_validation_suite()
    
    # Display detailed results
    print("\n" + "="*70)
    print("VALIDATION RESULTS SUMMARY")
    print("="*70)
    
    if 'summary' in report:
        summary = report['summary']
        print(f"\n✅ Components Passed: {summary.get('components_passed', 0)}/{summary.get('total_components_tested', 0)}")
        print(f"📈 Overall Pass Rate: {summary.get('overall_pass_rate', 0):.1%}")
    
    # Component-specific results
    for component, results in report.get('components', {}).items():
        print(f"\n{'='*70}")
        print(f"{component} - {results.get('explainer_type', 'Unknown')} Explainer")
        print(f"{'='*70}")
        
        # Fidelity
        print("\n🎯 FIDELITY METRICS:")
        for test_name, metrics in results.get('fidelity', {}).items():
            if 'error' not in metrics:
                print(f"\n  {test_name.upper()}:")
                for key, value in metrics.items():
                    if key not in ['pass', 'interpretation']:
                        print(f"    • {key}: {value}")
                print(f"    ➜ Interpretation: {metrics.get('interpretation', 'N/A')}")
                print(f"    ➜ Status: {'✅ PASS' if metrics.get('pass') else '❌ FAIL'}")
        
        # Stability
        print("\n🔄 STABILITY METRICS:")
        for test_name, metrics in results.get('stability', {}).items():
            if 'error' not in metrics:
                print(f"\n  {test_name.upper()}:")
                for key, value in metrics.items():
                    if key not in ['pass', 'interpretation']:
                        print(f"    • {key}: {value}")
                print(f"    ➜ Interpretation: {metrics.get('interpretation', 'N/A')}")
                print(f"    ➜ Status: {'✅ PASS' if metrics.get('pass') else '❌ FAIL'}")
        
        # Overall
        print(f"\n🏆 OVERALL STATUS: {'✅ PASS' if results.get('overall_pass') else '❌ FAIL'}")
    
    # Interpretation guide
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
FIDELITY METRICS:
  - Perturbation Sensitivity: Measures if perturbing important features changes predictions
    ➜ Higher is better (means XAI correctly identifies important features)
    ➜ Threshold: > 0.01 (1% prediction change)
  
STABILITY METRICS:
  - Similar Input Consistency: Measures if similar inputs get similar explanations
    ➜ Higher is better (means explanations are stable/reliable)
    ➜ Threshold: > 0.5 (50% feature overlap via Jaccard similarity)
  
  - Explanation Variance: Measures how much explanations change with small noise
    ➜ Lower is better (means explanations are robust)
    ➜ Threshold: < 0.1 (low variance)

OVERALL:
  - Both fidelity AND stability must pass for the explainer to be considered reliable
    """)
    
    print("="*70)
    
    # Save report to file
    output_file = "validation_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Full report saved to: {output_file}")

if __name__ == "__main__":
    demo_validation()
