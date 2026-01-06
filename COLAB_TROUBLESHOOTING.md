# Colab Troubleshooting Guide

## Error: Dimension Mismatch (451 vs 102)

**Problem:**
```
ValueError: expected shape=(None, 451), found shape=(32, 102)
```

**Root Cause:**
The preprocessor is fit on a **subset of data** (1000 records) which has fewer unique categorical values, creating only 102 features instead of 451.

**Solution:**
Use the **FULL dataset** for fitting the preprocessor.

## Quick Fix

In your Colab notebook, **replace Cell #5** (Load Models) with the code from `colab_fix_dimensions.py`.

**Key Changes:**
```python
# BEFORE (Wrong):
records = log_data.get('Records', [])[:1000]  # Subset!
df = pd.DataFrame([parse_log_c1(r) for r in records])

# AFTER (Correct):
all_records = log_data.get('Records', [])  # ALL records
df = pd.DataFrame([parse_log_c1(r) for r in all_records])
```

## Why This Happens

**OneHotEncoder** creates one column per unique categorical value:
- **eventName**: 250+ unique values (PutObject, GetObject, DeleteBucket, etc.)
- **eventSource**: 50+ unique values (s3, ec2, iam, etc.)
- **userIdentityType**: 5+ unique values (Root, IAMUser, AssumedRole, etc.)
- **awsRegion**: 20+ unique values (us-east-1, us-west-2, etc.)

**Total**: ~451 features (varies by dataset)

If you use only 1000 records:
- Fewer unique eventNames are seen
- Fewer unique eventSources are seen
- Result: Only ~102 features created
- Model expects 451 → **Error!**

## Verification Steps

After applying the fix:

1. **Check preprocessor output:**
```python
print(f"Preprocessor creates {X_processed.shape[1]} features")
# Should output: "Preprocessor creates 451 features"
```

2. **Check model input:**
```python
print(f"Model expects {state['models']['c1_ae'].input_shape[1]} features")
# Should output: "Model expects 451 features"
```

3. **They should match!**

## Alternative: Smaller Dataset

If you can't load the full dataset in Colab (memory issues), you need to:

1. **Retrain models** on the smaller dataset
2. **OR** Use a **pre-fitted preprocessor** from the original training

You can't mix:
- ❌ Models trained on 451 features
- ❌ Preprocessor fit on 102 features

## Memory Issues?

If full dataset is too large:

**Option 1: Sample Smartly**
```python
# Get ALL unique categorical values first
unique_events = df['eventName'].unique()
unique_sources = df['eventSource'].unique()
# etc.

# Then sample while ensuring all categories are represented
sampled_df = df.groupby(['eventName', 'eventSource']).sample(n=1)
```

**Option 2: Use Colab Pro**
- More RAM (25GB+ vs 12GB)
- Faster GPU
- $10/month

**Option 3: Use Subset for Inference Only**
```python
# Fit on FULL data
preprocessor.fit(full_df)

# But use subset for testing/validation
test_subset = full_df.sample(1000)
```

## Success Indicators

You'll know it's fixed when:
- ✅ No ValueError about shapes
- ✅ `X_processed.shape[1] == 451`
- ✅ Models load and predict successfully
- ✅ SHAP explanations work

## Still Having Issues?

Check:
1. Are you using the COMPLETE `flaws_cloudtrail00.json` file?
2. Did you upload the full file to Drive (not a sample)?
3. Is the file path correct in `CONFIG`?

The file should be ~100MB with 80,000+ records.
