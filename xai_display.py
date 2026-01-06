"""
Enhanced XAI Display Formatter
Creates the prettified output shown in XAI_COMPARISON.md
"""

def display_xai_enhanced(risk_score, category, xai_c1, p_log):
    """
    Display XAI results in enhanced format (like XAI_COMPARISON.md example)
    
    Args:
        risk_score: Anomaly risk score (0-100)
        category: Risk category (LOW/MEDIUM/HIGH)
        xai_c1: XAI data from explain_c1
        p_log: Parsed log data
    """
    
    print("\n" + "="*70)
    print(f"Risk Score: {risk_score}/100 ({category})")
    print(f"Category: {category}")
    print(f"Decision: Flag for review" if risk_score > 70 else "Monitor")
    print("="*70)
    
    # SHAP Explanation
    if 'shap' in xai_c1 and 'feature_importance' in xai_c1['shap']:
        print("\n📊 SHAP EXPLANATION:")
        print("Top Contributing Factors:")
        
        for i, feat in enumerate(xai_c1['shap']['feature_importance'][:3], 1):
            shap_val = feat['shap_value']
            importance = feat['importance']
            feature_name = feat['feature']
            
            # Interpret the feature
            interpretation = interpret_shap_feature(feature_name, shap_val, importance)
            
            print(f"  {i}. {feature_name}: {shap_val:.3f} ({importance*100:.0f}% importance)")
            print(f"     → {interpretation}")
    
    # Reconstruction Analysis
    if 'reconstruction' in xai_c1:
        recon_data = xai_c1['reconstruction']
        
        print("\n🔧 RECONSTRUCTION ANALYSIS:")
        print("Unusual Features:")
        
        if 'feature_errors' in recon_data:
            for feat in recon_data['feature_errors'][:2]:
                feature_name = feat['feature']
                error = feat['error']
                value = feat.get('original_value', 'N/A')
                
                interpretation = interpret_reconstruction_error(feature_name, error, value, p_log)
                
                print(f"  - {feature_name} ({value}): {error:.3f} error")
                print(f"    → {interpretation}")
        elif 'total_mse' in recon_data:
            mse = recon_data['total_mse']
            print(f"  - Total MSE: {mse:.4f}")
            print(f"    → Overall reconstruction error indicates anomaly severity")
    
    # AI Explanation (if LLM available)
    print("\n💡 AI EXPLANATION:")
    if 'llm_explanation' in xai_c1:
        print(xai_c1['llm_explanation'])
    else:
        # Generate template explanation
        top_feature = xai_c1['shap']['feature_importance'][0] if 'shap' in xai_c1 else None
        if top_feature:
            print(f"The {category.lower()} risk score ({risk_score}/100) is primarily driven by")
            print(f"{top_feature['feature']} (importance: {top_feature['importance']*100:.0f}%),")
            print(f"indicating this activity pattern deviates from learned normal behavior.")
        else:
            print(f"Risk score of {risk_score}/100 indicates {category.lower()} risk activity.")
    
    print("\n" + "="*70)


def interpret_shap_feature(feature_name, shap_value, importance):
    """Generate human-readable interpretation of SHAP feature"""
    
    interpretations = {
        'mse': f"Activity pattern deviates significantly from normal behavior",
        'latent_0': f"Unusual user behavior pattern detected in latent space",
        'latent_1': f"Unusual temporal access pattern",
        'latent_2': f"Abnormal resource access sequence",
        'latent_3': f"Atypical API call frequency",
        'latent_4': f"Irregular service interaction pattern",
        'latent_5': f"Abnormal API call sequence",
        'latent_6': f"Unusual geographic access pattern",
        'latent_7': f"Atypical time-of-day activity",
    }
    
    return interpretations.get(feature_name, f"Feature contributes {importance*100:.0f}% to anomaly score")


def interpret_reconstruction_error(feature_name, error, value, p_log):
    """Generate human-readable interpretation of reconstruction errors"""
    
    if feature_name == 'eventName':
        return f"This specific action ({value}) is rare for this user type"
    elif feature_name == 'userIdentityType':
        return f"{value} user activity outside normal patterns"
    elif feature_name == 'eventSource':
        return f"Unusual interaction with {value} service"
    elif feature_name == 'awsRegion':
        return f"Uncommon access from {value} region"
    else:
        return f"Feature shows {error:.1%} deviation from normal"


def format_c2_xai_enhanced(predicted_role, confidence, xai_c2):
    """Format C2 (LIME) results in enhanced style"""
    
    print("\n" + "="*70)
    print(f"Predicted Role: {predicted_role} (confidence: {confidence:.1%})")
    print("="*70)
    
    # LIME word importance
    if 'lime' in xai_c2 and 'word_importance' in xai_c2['lime']:
        print("\n🔍 LIME WORD IMPORTANCE:")
        
        positive_words = [w for w in xai_c2['lime']['word_importance'] if w['importance'] > 0][:3]
        negative_words = [w for w in xai_c2['lime']['word_importance'] if w['importance'] < 0][:3]
        
        if positive_words:
            print("Words supporting prediction:")
            for i, word in enumerate(positive_words, 1):
                print(f"  {i}. '{word['word']}' (+{word['importance']:.2f}) - {word.get('direction', 'positive')} indicator")
        
        if negative_words:
            print("\nWords opposing prediction:")
            for i, word in enumerate(negative_words, 1):
                print(f"  {i}. '{word['word']}' ({word['importance']:.2f}) - suggests different role")
    
    # Attention weights
    if 'attention' in xai_c2 and 'token_attention' in xai_c2['attention']:
        print("\n👁️ ATTENTION ANALYSIS:")
        print("BERT focused most on:")
        
        # Get top 3 attention tokens
        tokens = xai_c2['attention']['token_attention'][:3]
        for i, token in enumerate(tokens, 1):
            print(f"  {i}. '{token['token']}' ({token['attention']*100:.1f}% attention)")
    
    print("\n" + "="*70)


# Example usage in XAI_enhanced.py:
"""
# Replace display_xai_details() call with:
from xai_display import display_xai_enhanced

# In run_analysis_c1_with_xai():
if show_enhanced:
    display_xai_enhanced(risk_score, category, xai_data, p_log)
else:
    # Use simple display
    display_xai_details(xai_c1, xai_c2, xai_c3)
"""
