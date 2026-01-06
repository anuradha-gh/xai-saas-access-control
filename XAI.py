"""
🛡️ AI-Powered Granular Access Control for SaaS Applications  - Explainability Engine
"""

import json
import joblib
import pandas as pd
import numpy as np
import re
import os
import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer

# XAI Pipeline imports
try:
    from xai_explainer import XAIExplainerFactory
    from llm_translator import LLMTranslator, StakeholderType
    from xai_validator import XAIValidator
    XAI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ XAI modules not fully available: {e}")
    XAI_AVAILABLE = False

# ==========================================
# 0. CONFIGURATION
# ==========================================


# For Local Configuration:
CONFIG = {
    'c1_autoencoder': r'C:\Users\Anuradha\Downloads\SAAS XAI\autoencoder.h5',
    'c1_iso_forest':  r'C:\Users\Anuradha\Downloads\SAAS XAI\isolation_forest.joblib',
    'c2_bert_path':   r'C:\Users\Anuradha\Downloads\SAAS XAI\trained_role_classifier (1)\checkpoint-15000',
    'c3_sbert_path':  r'C:\Users\Anuradha\Downloads\SAAS XAI\c3_unsupervised_aws_model\sbert_model',
    'c3_iso_forest':  r'C:\Users\Anuradha\Downloads\SAAS XAI\c3_unsupervised_aws_model\isolation_forest.joblib',
    'log_data':       r'C:\Users\Anuradha\Downloads\SAAS XAI\flaws_cloudtrail00.json'
}

# LLM Configuration (Optional - set to None if Ollama not available)
USE_LLM = True  # Set to False to disable LLM explanations
LOCAL_MODEL_NAME = "gemma3:1b"

# XAI Configuration
XAI_CONFIG = {
    'enable_xai': True,  # Set to False to disable XAI explanations
    'default_stakeholder': 'general',  # 'technical', 'executive', 'compliance', 'general'
    'enable_validation': False,  # Set to True to enable validation metrics (slower)
    'num_shap_samples': 100,  # Number of samples for SHAP explanations
    'num_lime_samples': 1000,  # Number of samples for LIME explanations
}

if USE_LLM:
    try:
        import ollama
        print("✓ Ollama imported successfully")
    except ImportError:
        print("⚠️ Ollama not available - LLM explanations disabled")
        USE_LLM = False

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==========================================
# 1. GLOBAL STATE
# ==========================================

state = {'models': {}, 'preprocessors': {}}
chat_history = []
current_log_context = {}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================


def parse_log_c1(record):
    """Extract features for Component 1"""
    return {
        'eventName': record.get('eventName', 'Unknown'),
        'eventSource': record.get('eventSource', 'Unknown').split('.')[0],
        'userIdentityType': record.get('userIdentity', {}).get('type', 'Unknown'),
        'awsRegion': record.get('awsRegion', 'Unknown'),
    }


def log_to_text(row):
    """Convert log to natural language for BERT/SBERT"""
    user = "unknown"
    if 'userIdentity' in row:
        uid = row['userIdentity']
        user = uid.get('userName', uid.get('arn', uid.get('type', 'unknown')))
    user = re.sub(r'[^a-zA-Z0-9_.-]', '', str(user))
    return (f"User {user} of type {row.get('userIdentity',{}).get('type','')} "
            f"from IP {row.get('sourceIPAddress','')} performed {row.get('eventName','')} "
            f"on service {row.get('eventSource','')} in region {row.get('awsRegion','')}.")


def generate_explanation(component_name, data_context):
    """Generate explanation using Gemma LLM (if available)"""
    if not USE_LLM:
        return f"[LLM disabled] {component_name}: {data_context}"

    # Enhanced prompt with better structure and domain expertise
    prompt = f"""
    You are a Cybersecurity Expert. Explain the following analysis result in exactly 2-3 sentences (approx 50 words). Do not use markdown bolding.
    
    Component: {component_name}
    Data: {data_context}

Explanation:"""
    try:
        response = ollama.chat(model=LOCAL_MODEL_NAME, messages=[
                               {'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception as e:
        return f"[Explanation error: {str(e)}]"

# ==========================================
# 3. SYSTEM LOADER
# ==========================================


def load_system():
    """Load all AI models and preprocessors"""
    print("=" * 70)
    print("⏳ LOADING SYSTEM MODELS...")
    print("=" * 70)

    try:
        # Load preprocessing data
        print("📂 Loading CloudTrail data for preprocessing...")
        with open(CONFIG['log_data'], 'r') as f:
            log_data = json.load(f)
        records = log_data.get('Records', [])
        df = pd.DataFrame([parse_log_c1(r) for r in records])
        cat_features = ['eventName', 'eventSource',
                        'userIdentityType', 'awsRegion']
        for col in cat_features:
            df[col] = df[col].fillna('Unknown')

        # C1 Preprocessing
        print("🔧 Fitting C1 preprocessors...")
        preprocessor = ColumnTransformer([('cat', OneHotEncoder(
            handle_unknown='ignore'), cat_features)], remainder='passthrough')
        X_processed = preprocessor.fit_transform(df).toarray()
        scaler = StandardScaler()
        scaler.fit(X_processed)
        state['preprocessors']['c1_prep'] = preprocessor
        state['preprocessors']['c1_scaler'] = scaler

        # Load C1 Models
        print("🧠 Loading C1 Autoencoder...")
        state['models']['c1_ae'] = load_model(CONFIG['c1_autoencoder'])
        print("🌲 Loading C1 IsolationForest...")
        state['models']['c1_if'] = joblib.load(CONFIG['c1_iso_forest'])
        dense_layers = [l for l in state['models']
                        ['c1_ae'].layers if isinstance(l, tf.keras.layers.Dense)]
        bottleneck = min(dense_layers, key=lambda l: l.units)
        state['models']['c1_enc'] = tf.keras.models.Model(
            inputs=state['models']['c1_ae'].input, outputs=bottleneck.output)

        # Load C2 Models
        print("🤖 Loading C2 BERT model...")
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['c2_bert_path'])
        c2_model = AutoModelForSequenceClassification.from_pretrained(
            CONFIG['c2_bert_path'])
        state['models']['c2_pipe'] = pipeline(
            "text-classification", model=c2_model, tokenizer=tokenizer, return_all_scores=True)

        # Load C3 Models
        print("📝 Loading C3 Sentence-BERT...")
        state['models']['c3_sbert'] = SentenceTransformer(
            CONFIG['c3_sbert_path'])
        print("🌲 Loading C3 IsolationForest...")
        state['models']['c3_if'] = joblib.load(CONFIG['c3_iso_forest'])

        print("\n" + "=" * 70)
        print("✅ SYSTEM READY! All models loaded successfully.")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ LOAD ERROR: {e}\n")
        raise

# ==========================================
# 4. ANALYSIS FUNCTIONS
# ==========================================


def run_analysis_c1(log_raw):
    """Component 1: Anomaly Detection"""
    try:
        p_log = parse_log_c1(log_raw)
        processed = state['preprocessors']['c1_prep'].transform(
            pd.DataFrame([p_log])).toarray()
        scaled = state['preprocessors']['c1_scaler'].transform(processed)
        latent = state['models']['c1_enc'].predict(scaled, verbose=0)
        recon = state['models']['c1_ae'].predict(scaled, verbose=0)
        mse = np.mean((scaled - recon)**2, axis=1).reshape(-1, 1)
        features = np.hstack([latent, mse])

        raw_score = state['models']['c1_if'].decision_function(features)[0]
        risk_score = 100 - ((raw_score + 0.2) / 0.4 * 100)
        risk_score = max(0, min(100, risk_score))

        if risk_score > 75:
            category = "HIGH"
            color = "🔴"
        elif risk_score > 40:
            category = "MEDIUM"
            color = "🟠"
        else:
            category = "LOW"
            color = "🟢"

        return risk_score, category, color
    except Exception as e:
        print(f"Error in C1: {e}")
        return 0, "ERROR", "⚫"


def run_analysis_c2(log_raw):
    """Component 2: Role Classification"""
    try:
        text = log_to_text(log_raw)
        preds = state['models']['c2_pipe'](text)[0]
        sorted_preds = sorted(preds, key=lambda x: x['score'], reverse=True)
        return sorted_preds[:3]
    except Exception as e:
        print(f"Error in C2: {e}")
        return []


def run_analysis_c3(log_raw, anomaly_risk_score):
    """Component 3: Access Decision"""
    try:
        text = log_to_text(log_raw)
        emb = state['models']['c3_sbert'].encode([text])
        sem_score = state['models']['c3_if'].decision_function(emb)[0]

        raw_conf = abs(sem_score * 100)
        confidence = max(1, min(10, raw_conf))

        decision = "UNKNOWN"
        emoji = "⚪"

        if confidence >= 7:
            if sem_score > 0 and anomaly_risk_score < 50:
                decision = "AUTO-APPROVE"
                emoji = "🟢"
            else:
                decision = "ADMIN REVIEW"
                emoji = "🟣"
        elif confidence >= 4:
            decision = "TRIGGER MFA"
            emoji = "🟠"
        else:
            decision = "ADMIN REVIEW"
            emoji = "🟣"

        return decision, confidence, emoji
    except Exception as e:
        print(f"Error in C3: {e}")
        return "ERROR", 0, "⚫"

# ==========================================
# 5. TERMINAL UI FUNCTIONS
# ==========================================


def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 70)
    print("🛡️  AI-Powered Granular Access Control for SaaS Applications  - Explainability Engine")
    print("=" * 70)


def print_component_header(title):
    """Print component section header"""
    print("\n" + "-" * 70)
    print(f"  {title}")
    print("-" * 70)


def display_c1_results(risk_score, category, color, explanation):
    """Display Component 1 results"""
    print_component_header("COMPONENT 1: ANOMALY DETECTION")
    print(f"\n{color} Risk Category:  {category}")
    print(f"📊 Risk Score:     {risk_score:.2f} / 100")

    # ASCII progress bar
    bar_length = 50
    filled = int((risk_score / 100) * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"   [{bar}]")

    print(f"\n✨ AI Explanation:")
    print(f"   {explanation}")


def display_c2_results(top_roles, explanation):
    """Display Component 2 results"""
    print_component_header("COMPONENT 2: ROLE RECOMMENDATION")

    if not top_roles:
        print("\n⚠️ No role predictions available")
        return

    medals = ["🥇", "🥈", "🥉"]
    for i, role in enumerate(top_roles):
        pct = role['score'] * 100
        bar_length = 40
        filled = int((pct / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(f"\n{medals[i]} #{i+1} {role['label']}")
        print(f"   [{bar}] {pct:.1f}%")

    print(f"\n✨ AI Explanation:")
    print(f"   {explanation}")


def display_c3_results(decision, confidence, emoji, explanation):
    """Display Component 3 results"""
    print_component_header("COMPONENT 3: ACCESS CONTROL DECISION")
    print(f"\n{emoji} DECISION:  {decision}")
    print(f"📈 Confidence: {confidence:.2f} / 10")

    # Confidence bar
    bar_length = 40
    filled = int((confidence / 10) * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"   [{bar}]")

    print(f"\n✨ AI Explanation:")
    print(f"   {explanation}")


def run_full_analysis(log_json_str):
    """Run complete analysis pipeline"""
    global current_log_context

    try:
        log_raw = json.loads(log_json_str)
    except json.JSONDecodeError as e:
        print(f"\n❌ Invalid JSON: {e}")
        return

    print("\n⚙️  Processing with AI models...")

    # Run all three components
    risk_score, risk_cat, risk_color = run_analysis_c1(log_raw)
    top_roles = run_analysis_c2(log_raw)
    decision, confidence, dec_emoji = run_analysis_c3(log_raw, risk_score)

    # Generate explanations
    print("🤖 Generating AI explanations...")
    exp_c1 = generate_explanation(
        "Anomaly Detection",
        f"Risk Score: {risk_score} ({risk_cat}). Log: {log_raw.get('eventName')} by {log_raw.get('userIdentity', {}).get('userName')}"
    )
    exp_c2 = generate_explanation(
        "Role Classification",
        f"Top Prediction: {top_roles[0]['label'] if top_roles else 'None'}. Log Action: {log_raw.get('eventName')}"
    )
    exp_c3 = generate_explanation(
        "Access Decision",
        f"Decision: {decision}, Confidence: {confidence}/10"
    )

    # Save context for chatbot
    current_log_context = {
        "log": log_raw,
        "anomaly": {"score": risk_score, "category": risk_cat},
        "roles": top_roles,
        "decision": decision
    }

    # Display results
    print_banner()
    display_c1_results(risk_score, risk_cat, risk_color, exp_c1)
    display_c2_results(top_roles, exp_c2)
    display_c3_results(decision, confidence, dec_emoji, exp_c3)

    print("\n" + "=" * 70)

# ==========================================
# 6. CHATBOT FUNCTIONS
# ==========================================


def update_chat_ui():
    """Display chat history in terminal"""
    print("\n" + "-" * 70)
    print("CHAT HISTORY")
    print("-" * 70)
    for sender, msg in chat_history:
        icon = "👤" if sender == "You" else "🤖"
        print(f"\n{icon} {sender}: {msg}")
    print("-" * 70)


def on_chat_send(user_msg):
    """Process user chat message and get AI response"""
    global chat_history

    if not user_msg:
        return ""

    # 1. Update User Message
    chat_history.append(("You", user_msg))

    # 2. Get AI Response
    prompt = f"""
    You are a Security Assistant. Context: {json.dumps(current_log_context)}.
    User Question: {user_msg}.
    Keep answer short (under 50 words).
    """

    try:
        if USE_LLM:
            response = ollama.chat(model=LOCAL_MODEL_NAME, messages=[
                                   {'role': 'user', 'content': prompt}])
            ai_msg = response['message']['content']
        else:
            ai_msg = "[LLM disabled] Cannot generate response. Enable Ollama to use chatbot."
    except Exception as e:
        ai_msg = f"Error: {e}"

    chat_history.append(("AI", ai_msg))
    return ai_msg


def chat_loop():
    """Interactive chatbot in terminal"""
    global chat_history

    print("\n" + "=" * 70)
    print("💬 AI SECURITY ANALYST CHATBOT")
    print("=" * 70)
    print("Type your questions about the log analysis.")
    print("Commands: 'quit' or 'exit' to return, 'clear' to reset chat")
    print("-" * 70)

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n✓ Exiting chatbot\n")
                break

            if user_input.lower() == 'clear':
                chat_history = []
                print("\n✓ Chat history cleared\n")
                continue

            # Use the new on_chat_send method
            ai_response = on_chat_send(user_input)

            # Display only the new AI response
            print(f"\n🤖 AI: {ai_response}")

        except KeyboardInterrupt:
            print("\n\n✓ Exiting chatbot\n")
            break
        except EOFError:
            print("\n\n✓ Exiting chatbot\n")
            break

# ==========================================
# 7. MAIN INTERACTIVE LOOP
# ==========================================


def show_menu():
    """Show main menu"""
    print("\n" + "=" * 70)
    print("MAIN MENU")
    print("=" * 70)
    print("1. Analyze a CloudTrail log (paste JSON)")
    print("2. Analyze default example log")
    print("3. Open chatbot (ask questions about last analysis)")
    print("4. Exit")
    print("-" * 70)


def main():
    """Main program loop"""
    print_banner()

    # Load models
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

    while True:
        show_menu()

        try:
            choice = input("\nEnter choice (1-4): ").strip()

            if choice == '1':
                print(
                    "\n📝 Paste your CloudTrail log JSON (press Enter twice when done):")
                print("-" * 70)
                lines = []
                while True:
                    line = input()
                    if line == "":
                        if len(lines) > 0 and lines[-1] == "":
                            break
                    lines.append(line)

                log_json = "\n".join(lines[:-1])  # Remove last empty line
                run_full_analysis(log_json)

            elif choice == '2':
                print("\n📝 Using default example log:")
                print(default_log)
                run_full_analysis(default_log)

            elif choice == '3':
                if not current_log_context:
                    print("\n⚠️ No analysis performed yet. Please analyze a log first.")
                else:
                    chat_loop()

            elif choice == '4':
                print("\n👋 Goodbye!\n")
                break

            else:
                print("\n❌ Invalid choice. Please enter 1-4.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except EOFError:
            print("\n\n👋 Goodbye!\n")
            break


# ==========================================
# RUN MAIN PROGRAM
# ==========================================
if __name__ == "__main__":
    main()
