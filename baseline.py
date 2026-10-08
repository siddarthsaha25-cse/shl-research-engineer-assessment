import os
import glob
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# 1. Paths Setup
DATA_DIR = os.path.join("data", "Dataset_Final")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, "train")
TEST_AUDIO_DIR = os.path.join(DATA_DIR, "test")
SUBMISSION_DIR = "submissions"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# 2. Acoustic Feature Extraction Function
def extract_features(audio_path):
    try:
        # Load audio (downsampled to 16kHz for uniform processing)
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) == 0:
            return np.zeros(68)
        
        # Audio length & energy
        duration = len(y) / sr
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # MFCCs (20 coefficients + mean and std)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Compile vector
        features = np.hstack([
            duration,
            np.mean(rms), np.std(rms),
            np.mean(zcr), np.std(zcr),
            np.mean(spectral_centroid), np.std(spectral_centroid),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
            np.mean(spectral_rolloff), np.std(spectral_rolloff),
            mfcc_mean,
            mfcc_std,
            chroma_mean
        ])
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return np.zeros(68)

print("Loading CSV files...")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# 3. Extract Features
print("Extracting acoustic features from train audios (this may take 1-2 minutes)...")
X_train = np.array([
    extract_features(os.path.join(TRAIN_AUDIO_DIR, fname))
    for fname in train_df['filename']
])
y_train = train_df['label'].values

print("Extracting acoustic features from test audios...")
X_test = np.array([
    extract_features(os.path.join(TEST_AUDIO_DIR, fname))
    for fname in test_df['filename']
])

# 4. 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(train_df))
test_predictions = np.zeros(len(test_df))

print("\n--- Starting 5-Fold Training (LightGBM Regressor) ---")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbosity=-1
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    oof_predictions[val_idx] = model.predict(X_val)
    test_predictions += model.predict(X_test) / kf.n_splits

# 5. Evaluate CV Performance
rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))
mae = mean_absolute_error(y_train, oof_predictions)

print("\n==========================================")
print(f"Local 5-Fold CV RMSE: {rmse:.4f}")
print(f"Local 5-Fold CV MAE:  {mae:.4f}")
print("==========================================")

# Clip predictions to valid label bounds [0.0, 5.0]
test_predictions = np.clip(test_predictions, 0.0, 5.0)

# 6. Generate Submission
sub = pd.DataFrame({
    'filename': test_df['filename'],
    'label': test_predictions
})

out_path = os.path.join(SUBMISSION_DIR, "baseline_lgbm.csv")
sub.to_csv(out_path, index=False)
print(f"\nSaved submission to {out_path}!")
print(sub.head())