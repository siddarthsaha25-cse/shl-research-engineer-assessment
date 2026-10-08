import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. Directories
DATA_DIR = os.path.join("data", "Dataset_Final")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, "train")
TEST_AUDIO_DIR = os.path.join(DATA_DIR, "test")
SUBMISSION_DIR = "submissions"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# 2. Comprehensive Acoustic Feature Extraction
def extract_advanced_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) == 0:
            return np.zeros(160)
        
        # Duration & speech energy
        duration = len(y) / sr
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        
        # Pauses / speech activity ratio (silence detection)
        non_silent_intervals = librosa.effects.split(y, top_db=25)
        non_silent_duration = sum([(end - start) for start, end in non_silent_intervals]) / sr
        speech_ratio = non_silent_duration / (duration + 1e-6)
        
        # Spectral descriptors
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=y)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # MFCCs + Deltas
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Chroma & Tonnetz
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        
        # Mel spectrogram aggregates (40 bands)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        mel_mean = np.mean(mel, axis=1)
        mel_std = np.std(mel, axis=1)

        feature_vector = np.hstack([
            duration,
            speech_ratio,
            np.mean(rms), np.std(rms),
            np.mean(zcr), np.std(zcr),
            np.mean(centroid), np.std(centroid),
            np.mean(bandwidth), np.std(bandwidth),
            np.mean(rolloff), np.std(rolloff),
            np.mean(flatness), np.std(flatness),
            np.mean(contrast, axis=1), np.std(contrast, axis=1), # 7 + 7
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),          # 20 + 20
            np.mean(mfcc_delta, axis=1), np.std(mfcc_delta, axis=1), # 20 + 20
            np.mean(mfcc_delta2, axis=1),                        # 20
            np.mean(chroma, axis=1), np.std(chroma, axis=1),     # 12 + 12
            np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1),   # 6 + 6
            mel_mean, mel_std                                    # 40 + 40
        ])
        return feature_vector
    except Exception as e:
        print(f"Error {audio_path}: {e}")
        return np.zeros(160)

print("Loading metadata...")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Extracting training features (approx. 2-3 mins)...")
X_train = np.array([extract_advanced_features(os.path.join(TRAIN_AUDIO_DIR, f)) for f in train_df['filename']])
y_train = train_df['label'].values

print("Extracting testing features...")
X_test = np.array([extract_advanced_features(os.path.join(TEST_AUDIO_DIR, f)) for f in test_df['filename']])

# Replace any NaN or inf values with 0
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

# 3. Ensemble Model Training with 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
oof_ridge = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))
test_ridge = np.zeros(len(test_df))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Training LightGBM, CatBoost, and Ridge Ensembles ---")
for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train[trn_idx], y_train[trn_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    
    # Model 1: LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=450,
        learning_rate=0.02,
        num_leaves=25,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1
    )
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(35, verbose=False)])
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    test_lgb += lgb_model.predict(X_test) / kf.n_splits
    
    # Model 2: CatBoost
    cat_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.03,
        depth=5,
        random_seed=42 + fold,
        verbose=False
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=35)
    oof_cat[val_idx] = cat_model.predict(X_val)
    test_cat += cat_model.predict(X_test) / kf.n_splits
    
    # Model 3: Regularized Ridge Linear Model
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_scaled[trn_idx], y_tr)
    oof_ridge[val_idx] = ridge.predict(X_train_scaled[val_idx])
    test_ridge += ridge.predict(X_test_scaled) / kf.n_splits

# 4. Ensemble Optimization (Weighted Average)
oof_ensemble = (0.45 * oof_lgb) + (0.45 * oof_cat) + (0.10 * oof_ridge)
final_test_pred = (0.45 * test_lgb) + (0.45 * test_cat) + (0.10 * test_ridge)
final_test_pred = np.clip(final_test_pred, 0.0, 5.0)

rmse_baseline_comp = 0.7526
rmse_ensemble = np.sqrt(mean_squared_error(y_train, oof_ensemble))
mae_ensemble = mean_absolute_error(y_train, oof_ensemble)

print("\n==========================================")
print(f"Previous Baseline CV RMSE: {rmse_baseline_comp:.4f}")
print(f"Ensemble Model CV RMSE:    {rmse_ensemble:.4f}")
print(f"Ensemble Model CV MAE:     {mae_ensemble:.4f}")
print("==========================================")

# 5. Save Output
sub = pd.DataFrame({'filename': test_df['filename'], 'label': final_test_pred})
out_path = os.path.join(SUBMISSION_DIR, "ensemble_advanced.csv")
sub.to_csv(out_path, index=False)
print(f"\nSaved new submission to {out_path}!")