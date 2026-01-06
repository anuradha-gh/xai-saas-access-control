"""
Example: C1 Anomaly Detection with XAI Explanations
Demonstrates SHAP and reconstruction error analysis
"""

import json
import sys
sys.path.append('..')

from XAI_enhanced import *

def demo_c1_explanation():
    """Demo C1 XAI explanations"""
    print("="*70)
    print("C1 ANOMALY DETECTION - XAI DEMONSTRATION")
    print("="*70)
    
    # Load system
    print("\nLoading system...")
    load_system()
    
    # Example suspicious log
    suspicious_log = {
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
    
    # Get explanation
    print("\nAnalyzing suspicious activity...")
    risk_score, category, color, xai_data = run_analysis_c1_with_xai(suspicious_log)
    
    print(f"\n{color} Risk Classification: {category}")
    print(f"📊 Risk Score: {risk_score:.2f}/100")
    
    # Display SHAP values
    if 'shap' in xai_data and 'feature_importance' in xai_data['shap']:
        print("\n🔍 SHAP Feature Importance:")
        print("  (Shows which features contribute most to the anomaly score)")
        for feat in xai_data['shap']['feature_importance'][:5]:
            direction = "↑" if feat['shap_value'] > 0 else "↓"
            print(f"  {direction} {feat['feature']}: {feat['shap_value']:.4f}")
            print(f"     (Value: {feat['feature_value']:.3f}, Importance: {feat['importance']:.4f})")
    
    # Display reconstruction errors
    if 'reconstruction' in xai_data and 'feature_errors' in xai_data['reconstruction']:
        print("\n🔧 Reconstruction Errors:")
        print("  (Higher error = more unusual for the trained model)")
        for feat in xai_data['reconstruction']['feature_errors'][:5]:
            print(f"  • {feat['feature']}: {feat['error']:.4f}")
            print(f"     Original value: {feat['original_value']}")
    
    # Get LLM explanation for different stakeholders
    print("\n" + "="*70)
    print("LLM EXPLANATIONS FOR DIFFERENT STAKEHOLDERS")
    print("="*70)
    
    xai_context = {
        'risk_score': risk_score,
        'category': category,
        'shap': xai_data.get('shap', {}),
        'reconstruction': xai_data.get('reconstruction', {}),
        'log_context': {
            'eventName': suspicious_log.get('eventName'),
            'eventSource': suspicious_log.get('eventSource'),
            'userName': suspicious_log.get('userIdentity', {}).get('userName', 'Unknown'),
            'awsRegion': suspicious_log.get('awsRegion')
        }
    }
    
    if 'llm_translator' in state:
        print("\n👔 EXECUTIVE SUMMARY:")
        print("-" * 70)
        exec_exp = state['llm_translator'].translate("C1", xai_context, StakeholderType.EXECUTIVE)
        print(exec_exp)
        
        print("\n🔬 TECHNICAL EXPLANATION:")
        print("-" * 70)
        tech_exp = state['llm_translator'].translate("C1", xai_context, StakeholderType.TECHNICAL)
        print(tech_exp)
        
        print("\n📋 COMPLIANCE VIEW:")
        print("-" * 70)
        comp_exp = state['llm_translator'].translate("C1", xai_context, StakeholderType.COMPLIANCE)
        print(comp_exp)
    
    print("\n" + "="*70)

if __name__ == "__main__":
    demo_c1_explanation()
