# COLAB FIX: Dimension Mismatch Error
# Replace Cell #5 in XAI_Colab.ipynb with this code

import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

state = {'models': {}, 'preprocessors': {}}

print("📂 Loading CloudTrail data for preprocessing...")
with open(CONFIG['log_data'], 'r') as f:
    log_data = json.load(f)

# Helper functions (from XAI.py)
def parse_log_c1(record):
    return {
        'eventName': record.get('eventName', 'Unknown'),
        'eventSource': record.get('eventSource', 'Unknown').split('.')[0],
        'userIdentityType': record.get('userIdentity', {}).get('type', 'Unknown'),
        'awsRegion': record.get('awsRegion', 'Unknown'),
    }

# CRITICAL FIX: Use ALL records for fitting preprocessor
# This ensures same dimensions (451 features) as original model training
all_records = log_data.get('Records', [])
print(f"Total records available: {len(all_records)}")

# Use ALL records to fit preprocessor (not a subset!)
df = pd.DataFrame([parse_log_c1(r) for r in all_records])
cat_features = ['eventName', 'eventSource', 'userIdentityType', 'awsRegion']
for col in cat_features:
    df[col] = df[col].fillna('Unknown')

# C1 Preprocessing
print("🔧 Fitting C1 preprocessors...")
print(f"  DataFrame shape: {df.shape}")

preprocessor = ColumnTransformer(
    [('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)], 
    remainder='passthrough'
)
X_processed = preprocessor.fit_transform(df).toarray()
print(f"  Processed shape: {X_processed.shape[1]} features")
print(f"  Expected: 451 features (from original training)")

# Check dimensions
if X_processed.shape[1] != 451:
    print(f"\n⚠️ WARNING: Got {X_processed.shape[1]} features, expected 451")
    print("  Make sure you're using the FULL CloudTrail dataset!")
    print("  Subset data will have fewer unique categorical values.")

scaler = StandardScaler()
scaler.fit(X_processed)
state['preprocessors']['c1_prep'] = preprocessor
state['preprocessors']['c1_scaler'] = scaler

# Load C1 Models
print("\n🧠 Loading C1 Autoencoder...")
state['models']['c1_ae'] = load_model(CONFIG['c1_autoencoder'])
print("🌲 Loading C1 IsolationForest...")
state['models']['c1_if'] = joblib.load(CONFIG['c1_iso_forest'])

# Extract encoder (bottleneck layer)
dense_layers = [l for l in state['models']['c1_ae'].layers 
                if isinstance(l, tf.keras.layers.Dense)]
bottleneck = min(dense_layers, key=lambda l: l.units)
state['models']['c1_enc'] = tf.keras.models.Model(
    inputs=state['models']['c1_ae'].input, 
    outputs=bottleneck.output
)

# Verify model input shape
expected_shape = state['models']['c1_ae'].input_shape[1]
print(f"\n✅ Model expects {expected_shape} features")
print(f"✅ Preprocessor produces {X_processed.shape[1]} features")

if X_processed.shape[1] == expected_shape:
    print("✅ DIMENSIONS MATCH!")
else:
    print(f"❌ DIMENSION MISMATCH: {X_processed.shape[1]} != {expected_shape}")
    raise ValueError("Feature dimension mismatch. Use full dataset for preprocessing.")

# Load C2 Models
print("\n🤖 Loading C2 BERT model...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG['c2_bert_path'])
c2_model = AutoModelForSequenceClassification.from_pretrained(CONFIG['c2_bert_path'])
state['models']['c2_pipe'] = pipeline(
    "text-classification", 
    model=c2_model, 
    tokenizer=tokenizer, 
    return_all_scores=True
)

# Load C3 Models
print("📝 Loading C3 Sentence-BERT...")
state['models']['c3_sbert'] = SentenceTransformer(CONFIG['c3_sbert_path'])
print("🌲 Loading C3 IsolationForest...")
state['models']['c3_if'] = joblib.load(CONFIG['c3_iso_forest'])

print("\n" + "="*70)
print("✅ ALL MODELS LOADED SUCCESSFULLY!")
print("="*70)
