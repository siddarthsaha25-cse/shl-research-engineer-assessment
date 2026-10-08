import os
import numpy as np
import pandas as pd
import librosa
from scipy.stats import skew, kurtosis
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join("data", "Dataset_Final")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, "train")
TEST_AUDIO_DIR = os.path.join(DATA_DIR, "test")
SUBMISSION_DIR = "submissions"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

def stats(arr, axis=1):
    m = np.mean(arr, axis=axis)
    s = np.std(arr, axis=axis)
    sk = skew(arr, axis=axis)
    kt = kurtosis(arr, axis=axis)
    return np.hstack([m, s, sk, kt])

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) == 0:
            return np.zeros(210)
        
        duration = len(y) / sr
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        
        # Speech activity ratio
        intervals = librosa.effects.split(y, top_db=25)
        speech_time = sum([end - start for start, end in intervals]) / sr
        speech_ratio = speech_time / (duration + 1e-6)
        
        # Spectral features
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        sr_off = librosa.feature.spectral_rolloff(y=y, sr=sr)
        sf = librosa.feature.spectral_flatness(y=y)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # MFCCs + Delta + Delta-Delta
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        d_mfcc = librosa.feature.delta(mfcc)
        d2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Chroma & Tonnetz
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        
        feats = np.hstack([
            duration, speech_ratio,
            np.mean(rms), np.std(rms), skew(rms.squeeze()), kurtosis(rms.squeeze()),
            np.mean(zcr), np.std(zcr), skew(zcr.squeeze()), kurtosis(zcr.squeeze()),
            np.mean(sc), np.std(sc), np.mean(sb), np.std(sb),
            np.mean(sr_off), np.std(sr_off), np.mean(sf), np.std(sf),
            stats(contrast),
            stats(mfcc),
            np.mean(d_mfcc, axis=1), np.std(d_mfcc, axis=1),
            np.mean(d2_mfcc, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1)
        ])
        return feats
    except Exception as e:
        return np.zeros(210)

print("Reading files...")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Extracting training features...")
X_train = np.nan_to_num(np.array([extract_features(os.path.join(TRAIN_AUDIO_DIR, f)) for f in train_df['filename']]))
y_train = train_df['label'].values

print("Extracting testing features...")
X_test = np.nan_to_num(np.array([extract_features(os.path.join(TEST_AUDIO_DIR, f)) for f in test_df['filename']]))

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))
oof_et  = np.zeros(len(train_df))
oof_rdg = np.zeros(len(train_df))

pred_lgb = np.zeros(len(test_df))
pred_cat = np.zeros(len(test_df))
pred_et  = np.zeros(len(test_df))
pred_rdg = np.zeros(len(test_df))

print("\n--- Training 4-Model Ensemble across 5 Folds ---")
for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train[trn_idx], y_train[trn_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    
    # LightGBM
    m_lgb = lgb.LGBMRegressor(n_estimators=450, learning_rate=0.02, num_leaves=20, subsample=0.8, colsample_bytree=0.7, random_state=42+fold, verbosity=-1)
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
    oof_lgb[val_idx] = m_lgb.predict(X_val)
    pred_lgb += m_lgb.predict(X_test) / 5
    
    # CatBoost
    m_cat = CatBoostRegressor(iterations=550, learning_rate=0.03, depth=5, random_seed=42+fold, verbose=False)
    m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=35)
    oof_cat[val_idx] = m_cat.predict(X_val)
    pred_cat += m_cat.predict(X_test) / 5
    
    # ExtraTrees
    m_et = ExtraTreesRegressor(n_estimators=250, max_depth=12, random_state=42+fold, n_jobs=-1)
    m_et.fit(X_tr, y_tr)
    oof_et[val_idx] = m_et.predict(X_val)
    pred_et += m_et.predict(X_test) / 5
    
    # Ridge
    m_rdg = Ridge(alpha=20.0)
    m_rdg.fit(X_tr_s[trn_idx], y_tr)
    oof_rdg[val_idx] = m_rdg.predict(X_tr_s[val_idx])
    pred_rdg += m_rdg.predict(X_te_s) / 5

oof_ensemble = (0.35 * oof_lgb) + (0.35 * oof_cat) + (0.20 * oof_et) + (0.10 * oof_rdg)
final_preds = (0.35 * pred_lgb) + (0.35 * pred_cat) + (0.20 * pred_et) + (0.10 * pred_rdg)
final_preds = np.clip(final_preds, 0.0, 5.0)

rmse = np.sqrt(mean_squared_error(y_train, oof_ensemble))
mae = mean_absolute_error(y_train, oof_ensemble)

print("\n==========================================")
print(f"Ensemble 5-Fold CV RMSE: {rmse:.4f}")
print(f"Ensemble 5-Fold CV MAE:  {mae:.4f}")
print("==========================================")

sub = pd.DataFrame({'filename': test_df['filename'], 'label': final_preds})
out_path = os.path.join(SUBMISSION_DIR, "ensemble_tuned.csv")
sub.to_csv(out_path, index=False)
print(f"Saved: {out_path}")