import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. Paths
DATA_DIR = os.path.join("data", "Dataset_Final")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, "train")
TEST_AUDIO_DIR = os.path.join(DATA_DIR, "test")
SUBMISSION_DIR = "submissions"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# 2. Setup Device & Pretrained Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(model_name)
w2v_model = Wav2Vec2Model.from_pretrained(model_name).to(device)
w2v_model.eval()

def extract_w2v_embedding(file_path):
    try:
        # Wav2Vec expects 16kHz mono audio
        y, sr = librosa.load(file_path, sr=16000)
        if len(y) == 0:
            return np.zeros(768)
        
        # Limit to first 20 seconds to manage memory/speed
        max_len = 16000 * 20
        if len(y) > max_len:
            y = y[:max_len]

        inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            outputs = w2v_model(input_values)
            # Mean pool over temporal dimension: (batch, seq_len, 768) -> (768,)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
        return emb
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return np.zeros(768)

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Extracting Wav2Vec 2.0 embeddings for training data (this may take a few minutes)...")
X_train_w2v = np.array([extract_w2v_embedding(os.path.join(TRAIN_AUDIO_DIR, f)) for f in train_df['filename']])
y_train = train_df['label'].values

print("Extracting Wav2Vec 2.0 embeddings for testing data...")
X_test_w2v = np.array([extract_w2v_embedding(os.path.join(TEST_AUDIO_DIR, f)) for f in test_df['filename']])

# 3. 5-Fold Cross-Validation on Deep Embeddings
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_ridge = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
oof_lgb = np.zeros(len(train_df))

test_ridge = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))
test_lgb = np.zeros(len(test_df))

print("\n--- Training Models on Speech Embeddings ---")
for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train_w2v, y_train)):
    X_tr, y_tr = X_train_w2v[trn_idx], y_train[trn_idx]
    X_val, y_val = X_train_w2v[val_idx], y_train[val_idx]
    
    # 1. Ridge Regressor (often performs exceptionally well on dense 768-dim embeddings)
    ridge = Ridge(alpha=50.0)
    ridge.fit(X_tr, y_tr)
    oof_ridge[val_idx] = ridge.predict(X_val)
    test_ridge += ridge.predict(X_test_w2v) / kf.n_splits
    
    # 2. CatBoost
    cat = CatBoostRegressor(iterations=600, learning_rate=0.03, depth=4, random_seed=42 + fold, verbose=False)
    cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=40)
    oof_cat[val_idx] = cat.predict(X_val)
    test_cat += cat.predict(X_test_w2v) / kf.n_splits
    
    # 3. LightGBM
    lgb_m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, random_state=42 + fold, verbosity=-1)
    lgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(40, verbose=False)])
    oof_lgb[val_idx] = lgb_m.predict(X_val)
    test_lgb += lgb_m.predict(X_test_w2v) / kf.n_splits

# 4. Ensemble
oof_w2v = (0.50 * oof_ridge) + (0.30 * oof_cat) + (0.20 * oof_lgb)
test_w2v_pred = (0.50 * test_ridge) + (0.30 * test_cat) + (0.20 * test_lgb)

w2v_rmse = np.sqrt(mean_squared_error(y_train, oof_w2v))
w2v_mae = mean_absolute_error(y_train, oof_w2v)

print("\n==========================================")
print(f"Wav2Vec 2.0 Ensemble CV RMSE: {w2v_rmse:.4f}")
print(f"Wav2Vec 2.0 Ensemble CV MAE:  {w2v_mae:.4f}")
print("==========================================")

# 5. Save Model Output
sub = pd.DataFrame({'filename': test_df['filename'], 'label': np.clip(test_w2v_pred, 0.0, 5.0)})
out_path = os.path.join(SUBMISSION_DIR, "wav2vec_ensemble.csv")
sub.to_csv(out_path, index=False)
print(f"\nSaved Wav2Vec predictions to {out_path}!")

# 6. Combined Super-Blend (Acoustic Handcrafted + Deep Embeddings)
# Blend with previous advanced ensemble if available
advanced_path = os.path.join(SUBMISSION_DIR, "ensemble_advanced.csv")
if os.path.exists(advanced_path):
    adv_sub = pd.read_csv(advanced_path)
    super_blend = (0.60 * sub['label']) + (0.40 * adv_sub['label'])
    super_blend_sub = pd.DataFrame({'filename': test_df['filename'], 'label': np.clip(super_blend, 0.0, 5.0)})
    super_path = os.path.join(SUBMISSION_DIR, "super_blend.csv")
    super_blend_sub.to_csv(super_path, index=False)
    print(f"Saved Super-Blend (Acoustic + Wav2Vec) to {super_path}!")