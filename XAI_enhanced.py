"""
Enhanced XAI.py - Wrapper that integrates XAI explanations
Use this to run the system with full XAI capabilities
"""

# Import original system
from XAI import *
from xai_explainer import XAIExplainerFactory
from llm_translator import LLMTranslator, StakeholderType  
from xai_validator import XAIValidator

# Override the load_system function to add XAI initialization
original_load_system = load_system

def load_system_with_xai():
    """Enhanced system loader with XAI pipeline"""
    # Call original loader
    original_load_system()
    
    # Add XAI components
    if XAI_AVAILABLE and XAI_CONFIG.get('enable_xai', True):
        print("\n🔧 Initializing XAI Pipeline...")
        
        # Get preprocessed data for SHAP background
        print("📂 Loading CloudTrail data for XAI background...")
        with open(CONFIG['log_data'], 'r') as f:
            log_data = json.load(f)
        records = log_data.get('Records', [])
        df = pd.DataFrame([parse_log_c1(r) for r in records[:100]])  # Use subset
        cat_features = ['eventName', 'eventSource', 'userIdentityType', 'awsRegion']
        for col in cat_features:
            df[col] = df[col].fillna('Unknown')
        
        X_processed = state['preprocessors']['c1_prep'].transform(df).toarray()
        X_scaled = state['preprocessors']['c1_scaler'].transform(X_processed)
        
        # IMPORTANT: For C1 SHAP, we need latent features + MSE (same as Isolation Forest input)
        # NOT the raw one-hot encoded features
        latent = state['models']['c1_enc'].predict(X_scaled, verbose=0)
        recon = state['models']['c1_ae'].predict(X_scaled, verbose=0)
        mse = np.mean((X_scaled - recon)**2, axis=1).reshape(-1, 1)
        c1_features_for_shap = np.hstack([latent, mse])  # This matches Isolation Forest input
        
        # IMPROVEMENT #1: Prepare C3 reference embeddings for similar activities
        print("📝 Preparing C3 reference embeddings...")
        reference_texts = []
        for record in records[:500]:  # Use 500 samples as reference
            try:
                text = log_to_text(record)
                reference_texts.append(text)
            except:
                continue
        
        # Generate embeddings for reference texts
        if reference_texts:
            reference_embeddings = state['models']['c3_sbert'].encode(reference_texts, show_progress_bar=False)
            print(f"✓ Generated {len(reference_texts)} reference embeddings for C3")
        else:
            reference_embeddings = None
            reference_texts = None
            print("⚠️ No reference data for C3 - similar activities won't be shown")
        
        # Create explainer factory
        state['xai_explainer'] = XAIExplainerFactory(state['models'], state['preprocessors'])
        
        # Prepare background data for SHAP + reference data for C3
        background_data = {
            'c1_features': c1_features_for_shap,  # Latent + MSE features
            'c3_reference_embeddings': reference_embeddings,  # NEW!
            'c3_reference_texts': reference_texts  # NEW!
        }
        
        state['xai_explainer'].initialize(background_data)
        
        # Create LLM translator
        state['llm_translator'] = LLMTranslator(
            use_ollama=USE_LLM,
            model_name=LOCAL_MODEL_NAME
        )
        
        # Create validator (optional)
        if XAI_CONFIG.get('enable_validation', False):
            state['xai_validator'] = XAIValidator()
            print("✓ XAI Validator initialized")
        
        print("✅ XAI Pipeline Ready!")
    else:
        print("ℹ️ XAI Pipeline disabled")

# Replace the function
load_system = load_system_with_xai


# Enhanced analysis functions with XAI
def run_analysis_c1_with_xai(log_raw):
    """Component 1 with XAI explanations"""
    # Run original analysis
    risk_score, category, color = run_analysis_c1(log_raw)
    
    # Add XAI explanations if available
    xai_data = {}
    if 'xai_explainer' in state and state['xai_explainer'].initialized:
        try:
            # Prepare data - need to extract same features as Isolation Forest input
            p_log = parse_log_c1(log_raw)
            processed = state['preprocessors']['c1_prep'].transform(pd.DataFrame([p_log])).toarray()
            scaled = state['preprocessors']['c1_scaler'].transform(processed)
            
            # Get latent features + MSE (same as Isolation Forest input)
            latent = state['models']['c1_enc'].predict(scaled, verbose=0)
            recon = state['models']['c1_ae'].predict(scaled, verbose=0)
            mse = np.mean((scaled - recon)**2, axis=1).reshape(-1, 1)
            features_for_if = np.hstack([latent, mse])
            
            # Get XAI explanations
            # Pass features_for_if to SHAP (not scaled)
            explanations = state['xai_explainer'].explain_c1(features_for_if, p_log)
            xai_data = explanations
            
        except Exception as e:
            print(f"⚠️ XAI C1 error: {e}")
    
    return risk_score, category, color, xai_data


def run_analysis_c2_with_xai(log_raw):
    """Component 2 with XAI explanations"""
    # Run original analysis
    top_roles = run_analysis_c2(log_raw)
    
    # Add XAI explanations
    xai_data = {}
    if 'xai_explainer' in state and state['xai_explainer'].initialized:
        try:
            text = log_to_text(log_raw)
            explanations = state['xai_explainer'].explain_c2(text)
            xai_data = explanations
        except Exception as e:
            print(f"⚠️ XAI C2 error: {e}")
    
    return top_roles, xai_data


def run_analysis_c3_with_xai(log_raw, anomaly_risk_score):
    """Component 3 with XAI explanations"""
    # Run original analysis
    decision, confidence, emoji = run_analysis_c3(log_raw, anomaly_risk_score)
    
    # Add XAI explanations
    xai_data = {}
    if 'xai_explainer' in state and state['xai_explainer'].initialized:
        try:
            text = log_to_text(log_raw)
            emb = state['models']['c3_sbert'].encode([text])
            explanations = state['xai_explainer'].explain_c3(text, emb[0])
            xai_data = explanations
        except Exception as e:
            print(f"⚠️ XAI C3 error: {e}")
    
    return decision, confidence, emoji, xai_data


def run_full_analysis_with_xai(log_json_str, stakeholder_type='general'):
    """Run complete analysis with XAI explanations"""
    global current_log_context
    
    try:
        log_raw = json.loads(log_json_str)
    except json.JSONDecodeError as e:
        print(f"\n❌ Invalid JSON: {e}")
        return
    
    print("\n⚙️  Processing with AI models...")
    
    # Run all three components with XAI
    risk_score, risk_cat, risk_color, xai_c1 = run_analysis_c1_with_xai(log_raw)
    top_roles, xai_c2 = run_analysis_c2_with_xai(log_raw)
    decision, confidence, dec_emoji, xai_c3 = run_analysis_c3_with_xai(log_raw, risk_score)
    
    # Generate XAI-enhanced explanations
    print("🤖 Generating AI explanations with XAI insights...")
    
    stakeholder = StakeholderType(stakeholder_type) if stakeholder_type in ['technical', 'executive', 'compliance', 'general'] else StakeholderType.GENERAL
    
    if 'llm_translator' in state:
        # Prepare XAI data for translation
        c1_xai_data = {
            'risk_score': risk_score,
            'category': risk_cat,
            'shap': xai_c1.get('shap', {}),
            'reconstruction': xai_c1.get('reconstruction', {}),
            'log_context': {
                'eventName': log_raw.get('eventName'),
                'eventSource': log_raw.get('eventSource'),
                'userName': log_raw.get('userIdentity', {}).get('userName', 'Unknown'),
                'awsRegion': log_raw.get('awsRegion')
            }
        }
        
        c2_xai_data = {
            'predicted_role': top_roles[0]['label'] if top_roles else 'None',
            'confidence': top_roles[0]['score'] if top_roles else 0,
            'lime': xai_c2.get('lime', {}),
            'attention': xai_c2.get('attention', {}),
            'log_context': {
                'text': log_to_text(log_raw)
            }
        }
        
        c3_xai_data = {
            'decision': decision,
            'confidence': confidence,
            'embedding': xai_c3.get('embedding', {}),
            'log_context': {
                'text': log_to_text(log_raw)
            }
        }
        
        # Translate to natural language
        exp_c1 = state['llm_translator'].translate("C1", c1_xai_data, stakeholder)
        exp_c2 = state['llm_translator'].translate("C2", c2_xai_data, stakeholder)
        exp_c3 = state['llm_translator'].translate("C3", c3_xai_data, stakeholder)
    else:
        # Fallback to original explanations
        exp_c1 = generate_explanation(
            "Anomaly Detection",
            f"Risk Score: {risk_score} ({risk_cat}). Log: {log_raw.get('eventName')}"
        )
        exp_c2 = generate_explanation(
            "Role Classification",
            f"Top Prediction: {top_roles[0]['label'] if top_roles else 'None'}"
        )
        exp_c3 = generate_explanation(
            "Access Decision",
            f"Decision: {decision}, Confidence: {confidence}/10"
        )
    
    # Save context for chatbot
    current_log_context = {
        "log": log_raw,
        "anomaly": {"score": risk_score, "category": risk_cat, "xai": xai_c1},
        "roles": top_roles,
        "role_xai": xai_c2,
        "decision": decision,
        "decision_xai": xai_c3
    }
    
    # Display results
    print_banner()
    display_c1_results(risk_score, risk_cat, risk_color, exp_c1)
    display_c2_results(top_roles, exp_c2)
    display_c3_results(decision, confidence, dec_emoji, exp_c3)
    
    # Display XAI details if available
    if XAI_CONFIG.get('enable_xai', True):
        display_xai_details(xai_c1, xai_c2, xai_c3)
    
    print("\n" + "=" * 70)


def display_xai_details(xai_c1, xai_c2, xai_c3):
    """Display detailed XAI information"""
    print("\n" + "-" * 70)
    print("  🔬 XAI TECHNICAL DETAILS")
    print("-" * 70)
    
    # C1 XAI
    if xai_c1.get('shap', {}).get('feature_importance'):
        print("\n📊 C1 Feature Importance (SHAP):")
        for feat in xai_c1['shap']['feature_importance'][:3]:
            print(f"  - {feat['feature']}: {feat['shap_value']:.3f}")
    
    if xai_c1.get('reconstruction', {}).get('feature_errors'):
        print("\n🔧 C1 Reconstruction Errors:")
        for feat in xai_c1['reconstruction']['feature_errors'][:3]:
            print(f"  - {feat['feature']}: {feat['error']:.3f}")
    
    # C2 XAI
    if xai_c2.get('lime', {}).get('word_importance'):
        print("\n📝 C2 Important Words (LIME):")
        for word in xai_c2['lime']['word_importance'][:5]:
            print(f"  - '{word['word']}': {word['importance']:.3f}")
    
    # C3 XAI - ADDED!
    if xai_c3.get('embedding', {}).get('nearest_neighbors'):
        print("\n🔍 C3 Similar Activities (Semantic Similarity):")
        for neighbor in xai_c3['embedding']['nearest_neighbors'][:2]:
            print(f"  - {neighbor['text'][:60]}... (similarity: {neighbor['similarity']:.2%})")
    elif xai_c3.get('embedding'):
        # Fallback: show what C3 data exists
        print("\n🔍 C3 Embedding Analysis:")
        emb_data = xai_c3['embedding']
        if 'isolation_score' in emb_data:
            print(f"  - Isolation Score: {emb_data['isolation_score']:.4f}")
        if 'important_dimensions' in emb_data:
            print(f"  - Top Important Dimensions:")
            for dim in emb_data['important_dimensions'][:3]:
                print(f"    • Dimension {dim['dimension']}: {dim['importance']:.4f}")
    
    # C3 SHAP (if available)
    if xai_c3.get('shap', {}).get('feature_importance'):
        print("\n📊 C3 Embedding Dimensions (SHAP - Top 3):")
        for feat in xai_c3['shap']['feature_importance'][:3]:
            dim = feat['feature'].replace('dim_', '')
            print(f"  - Dimension {dim}: {feat['shap_value']:.4f}")


def run_validation_suite():
    """Run comprehensive XAI validation"""
    if 'xai_validator' not in state:
        print("⚠️ XAI Validator not initialized. Enable validation in config.")
        return
    
    print("\n" + "="*70)
    print("🔬 RUNNING XAI VALIDATION SUITE")
    print("="*70)
    
    # Prepare test data
    print("\n📂 Loading test data...")
    with open(CONFIG['log_data'], 'r') as f:
        log_data = json.load(f)
    records = log_data.get('Records', [])[:50]  # Use 50 samples
    
    df = pd.DataFrame([parse_log_c1(r) for r in records])
    cat_features = ['eventName', 'eventSource', 'userIdentityType', 'awsRegion']
    for col in cat_features:
        df[col] = df[col].fillna('Unknown')
    
    X_processed = state['preprocessors']['c1_prep'].transform(df).toarray()
    X_scaled = state['preprocessors']['c1_scaler'].transform(X_processed)
    
    test_data = {
        'c1': X_scaled
    }
    
    # Run validation
    report = state['xai_validator'].validate_all(
        state['models'],
        state['xai_explainer'],
        test_data
    )
    
    # Display report
    print(state['xai_validator'].generate_report())
    
    return report


def show_xai_menu():
    """Enhanced menu with XAI options"""
    print("\n" + "=" * 70)
    print("MAIN MENU")
    print("=" * 70)
    print("1. Analyze a CloudTrail log (paste JSON)")
    print("2. Analyze default example log")
    print("3. Open chatbot (ask questions about last analysis)")
    print("4. Run XAI Validation Suite (quantitative metrics)")
    print("5. Change explanation style (technical/executive/compliance/general)")
    print("6. Exit")
    print("-" * 70)


def main_with_xai():
    """Enhanced main program with XAI"""
    print_banner()
    
    # Load models and XAI
    load_system()
    
    # Default example log
    default_log = """{
  "eventTime": "2017-02-12T21:30:56Z",
  "eventSource": "s3.amazonaws.com",
  "eventName": "DeleteBucket",
  "awsRegion": "us-west-2",
  "sourceIPAddress": "AWS Internal",
  "userIdentity": {
    "type": "Root",
    "userName": "root_account"
  }
}"""
    
    current_stakeholder = 'general'
    
    while True:
        show_xai_menu()
        
        try:
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == '1':
                print("\n📝 Paste your CloudTrail log JSON (press Enter twice when done):")
                print("-" * 70)
                lines = []
                while True:
                    line = input()
                    if line == "":
                        if len(lines) > 0 and lines[-1] == "":
                            break
                    lines.append(line)
                
                log_json = "\n".join(lines[:-1])
                run_full_analysis_with_xai(log_json, current_stakeholder)
            
            elif choice == '2':
                print("\n📝 Using default example log:")
                print(default_log)
                run_full_analysis_with_xai(default_log, current_stakeholder)
            
            elif choice == '3':
                if not current_log_context:
                    print("\n⚠️ No analysis performed yet. Please analyze a log first.")
                else:
                    chat_loop()
            
            elif choice == '4':
                run_validation_suite()
            
            elif choice == '5':
                print("\nSelect explanation style:")
                print("  1. General (default)")
                print("  2. Technical (ML engineers)")
                print("  3. Executive (business leaders)")
                print("  4. Compliance (auditors)")
                style_choice = input("Enter choice (1-4): ").strip()
                
                styles = {'1': 'general', '2': 'technical', '3': 'executive', '4': 'compliance'}
                current_stakeholder = styles.get(style_choice, 'general')
                print(f"✓ Explanation style set to: {current_stakeholder}")
            
            elif choice == '6':
                print("\n👋 Goodbye!\n")
                break
            
            else:
                print("\n❌ Invalid choice. Please enter 1-6.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except EOFError:
            print("\n\n👋 Goodbye!\n")
            break


if __name__ == "__main__":
    main_with_xai()
