#!/usr/bin/env python3
"""
temporal_analysis.py
====================

Evaluate model performance and feature reliability across different study durations,
retraining models from scratch for each duration. Uses 75/25 train/test split by 
whole days. Produces Figure 4-style plots showing:
(a) ICC vs study duration  
(b) AUC vs study duration with bootstrapped confidence intervals
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to save memory
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import keras_tuner as kt
import joblib
import warnings
import gc
import psutil
import shutil
import tempfile
warnings.filterwarnings('ignore')

# Import custom layers from train.py
from train import AttentionLayer, MultiHeadAttentionLayer, SelfAttentionLayer

# Configure TensorFlow memory usage
def configure_tensorflow_memory():
    """Configure TensorFlow to use less memory"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    
    # Limit CPU threads if no GPU
    if not gpus:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

# Call configuration at import time
configure_tensorflow_memory()

# Set style
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

FEATURES = [
    'hr_mean_5min','hr_min_5min','hr_max_5min','hr_std_5min',
    'steps_total_5min','steps_mean_5min','steps_min_5min','steps_max_5min',
    'steps_std_5min','steps_diff_5min',
    'hr_mean_10min','hr_min_10min','hr_max_10min','hr_std_10min',
    'steps_total_10min','steps_mean_10min','steps_min_10min','steps_max_10min',
    'steps_std_10min','steps_diff_10min',
    'hr_mean_30min','hr_min_30min','hr_max_30min','hr_std_30min',
    'steps_total_30min','steps_mean_30min','steps_min_30min','steps_max_30min',
    'steps_std_30min','steps_diff_30min',
    'hr_mean_60min','hr_min_60min','hr_max_60min','hr_std_60min',
    'steps_total_60min','steps_mean_60min','steps_min_60min','steps_max_60min',
    'steps_std_60min','steps_diff_60min',
    'stress_mean','stress_min','stress_max','stress_std',
    'stress_mean_lag_1','stress_mean_lag_3','stress_mean_lag_5',
    'BP_spike_lag_1','BP_spike_lag_3','BP_spike_lag_5',
    'hr_mean_5min_lag_1','hr_mean_5min_lag_3','hr_mean_5min_lag_5',
    'steps_total_10min_lag_1','steps_total_10min_lag_3','steps_total_10min_lag_5',
    'hr_steps_ratio','stress_weighted_hr','stress_steps_ratio','steps_hr_variability_ratio',
    'hr_mean_rolling_3','steps_total_rolling_5','hr_std_rolling_3',
    'cumulative_stress_30min','cumulative_steps_30min',
    'hour_of_day','day_of_week','is_working_hours','is_weekend',
    'time_since_last_BP_spike'
]

# Define features to use for ICC calculation (key physiological features)
ICC_FEATURES = [
    'hr_mean_5min', 'hr_std_5min', 'hr_mean_10min', 'hr_std_10min',
    'steps_total_5min', 'steps_mean_5min', 'steps_total_10min', 'steps_mean_10min',
    'hr_mean_30min', 'hr_std_30min', 'steps_total_30min', 'steps_mean_30min',
    'stress_mean', 'stress_std', 'hr_steps_ratio', 'stress_weighted_hr'
]

def print_memory_usage(label=""):
    """Print current memory usage"""
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"  Memory usage {label}: {mem_info.rss / 1024 / 1024:.1f} MB")
    except:
        pass

def aggressive_cleanup():
    """Aggressive memory cleanup"""
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    plt.close('all')
    for _ in range(3):
        gc.collect()

def clear_keras_session():
    """Clear Keras session and free GPU memory"""
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    gc.collect()

# ==== NEW: results cache helpers ============================================

def _result_cache_dir():
    d = os.path.join('results', 'temporal_cache')
    os.makedirs(d, exist_ok=True)
    return d

def _result_cache_path(scope: str, **keys) -> str:
    """
    scope: one of {'global_days','global_fixed','personalized_days','personalized_fixed'}
    keys: identifying fields, e.g. days=7, pid=17, train_days=9, test_days=7
    """
    parts = [scope] + [f"{k}-{keys[k]}" for k in sorted(keys.keys())]
    fname = "_".join(parts) + ".pkl"
    return os.path.join(_result_cache_dir(), fname)

def save_run_result(scope: str, data: dict, **keys):
    """Persist one unit of work's result (e.g., one total_days or one (pid,train_days))."""
    path = _result_cache_path(scope, **keys)
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_run_result(scope: str, **keys):
    """Return cached result dict or None if not available."""
    path = _result_cache_path(scope, **keys)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def load_participant_df(pid):
    """Load participant data"""
    path = os.path.join("processed", f"hp{pid}", "processed_bp_prediction_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["datetime_local"])
    df["pid"] = pid
    return df

def build_feature_intersection(settings):
    """Get common features across all participants"""
    inter = set(FEATURES)
    for s in settings:
        df = load_participant_df(s["pid"])
        inter &= {c for c in df.columns if c in FEATURES}
    return sorted(inter)

def split_days_75_25(dates, seed=None):
    """Split dates into 75% train / 25% test ensuring at least one test day"""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    dates = np.array(sorted(dates))
    total = len(dates)
    train_days = int(np.floor(0.75 * total))
    
    # Ensure at least one test day
    if total - train_days < 1:
        train_days = total - 1
    if train_days < 1:
        return None, None
        
    permuted = rng.permutation(dates)
    train_dates = np.sort(permuted[:train_days])
    test_dates = np.sort(permuted[train_days:])
    
    return train_dates, test_dates

def calculate_icc_3_k(ratings):
    """Calculate ICC(3,k) - two-way random average measures"""
    X = np.asarray(ratings, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        return np.nan
    
    # Remove any rows that are all NaN
    row_mask = ~np.all(np.isnan(X), axis=1)
    X = X[row_mask]
    
    if X.shape[0] < 2:
        return np.nan
    
    # Fill NaNs with row means
    for i in range(X.shape[0]):
        row = X[i, :]
        if np.any(np.isnan(row)):
            row_mean = np.nanmean(row)
            if not np.isnan(row_mean):
                X[i, np.isnan(row)] = row_mean
            else:
                X[i, :] = 0
    
    n, k = X.shape
    
    # Calculate means
    grand_mean = X.mean()
    row_means = X.mean(axis=1)
    col_means = X.mean(axis=0)
    
    # Sum of squares
    ss_total = np.sum((X - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols
    
    # Mean squares
    ms_rows = ss_rows / (n - 1) if n > 1 else 0
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else 0
    
    # ICC(3,k) formula
    if ms_rows + (k - 1) * ms_error == 0:
        return np.nan
    
    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)
    
    return max(0, min(1, icc))

def calculate_feature_icc_multiple_permutations(df, features, n_days, condition='all', n_permutations=10):
    """Calculate ICC for each feature using random permutations of days"""
    dates = sorted(df['datetime_local'].dt.date.unique())
    
    if len(dates) < n_days:
        return {}
    
    all_icc_values = {feat: [] for feat in features if feat in df.columns}
    
    for perm in range(n_permutations):
        # Randomly sample n_days
        rng = np.random.default_rng(perm)
        selected_dates = rng.choice(dates, size=min(n_days, len(dates)), replace=False)
        
        # Calculate ICC for this permutation
        for feat in features:
            if feat not in df.columns:
                continue
            
            # Create matrix: rows = hours, columns = days
            ratings = []
            
            # Group by hour across days
            for hour in range(24):
                hour_values = []
                for date in selected_dates:
                    day_df = df[df['datetime_local'].dt.date == date]
                    hour_df = day_df[day_df['datetime_local'].dt.hour == hour]
                    
                    if condition == 'spike':
                        hour_df = hour_df[hour_df['BP_spike'] == 1]
                    elif condition == 'healthy':
                        hour_df = hour_df[hour_df['BP_spike'] == 0]
                    
                    if len(hour_df) > 0:
                        val = hour_df[feat].mean()
                        hour_values.append(val)
                    else:
                        hour_values.append(np.nan)
                
                if len(hour_values) == len(selected_dates):
                    ratings.append(hour_values)
            
            if len(ratings) >= 2:  # Need at least 2 hours
                icc = calculate_icc_3_k(np.array(ratings))
                if not np.isnan(icc):
                    all_icc_values[feat].append(icc)
    
    # Return mean ICC across permutations
    mean_icc = {}
    for feat, values in all_icc_values.items():
        if values:
            mean_icc[feat] = np.mean(values)
    
    return mean_icc

def bootstrap_auc(y_true, y_pred, n_bootstrap=1000, sample_fraction=0.7, seed=42):
    """Calculate bootstrapped AUC with confidence intervals"""
    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    sample_size = int(sample_fraction * n_samples)
    
    auc_scores = []
    
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = rng.choice(n_samples, size=sample_size, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Check if both classes are present
        if len(np.unique(y_true_boot)) == 2:
            try:
                auc = roc_auc_score(y_true_boot, y_pred_boot)
                auc_scores.append(auc)
            except:
                pass
    
    if len(auc_scores) > 0:
        return np.mean(auc_scores), np.std(auc_scores)
    else:
        return np.nan, np.nan

def train_xgb_model(X_train, y_train, participant_settings, quick=True):
    """Train XGBoost model with reduced hyperparameter search for efficiency"""
    pos, neg = sum(y_train == 1), sum(y_train == 0)
    spw = neg / pos if pos > 0 else 1
    
    # Determine minority class size
    minority_class_size = min(pos, neg)
    
    # Check if we should use ADASYN
    min_samples_for_adasyn = 10  # Need at least this many samples in minority class
    use_adasyn = (not participant_settings.get('no_resample', False) and 
                minority_class_size >= min_samples_for_adasyn)
    
    if not use_adasyn:
        # Don't use ADASYN if too few samples
        pipeline_steps = [
            ('scaler', StandardScaler()),
            ('xgb', xgb.XGBClassifier(scale_pos_weight=spw, random_state=42, eval_metric='logloss'))
        ]
        if quick:
            param_grid = {
                'xgb__max_depth': [3, 5],
                'xgb__learning_rate': [0.05, 0.1],
                'xgb__n_estimators': [100, 150]
            }
        else:
            param_grid = {
                'xgb__max_depth': [3, 5, 7],
                'xgb__learning_rate': [0.01, 0.05, 0.1],
                'xgb__n_estimators': [100, 150, 200]
            }
    else:
        # Use ADASYN with properly adjusted n_neighbors
        base_n_neighbors = participant_settings.get('n_neighbors', 5)
        # ADASYN needs k < minority_class_size, so we use minority_class_size - 1
        max_n_neighbors = min(base_n_neighbors, minority_class_size - 1, 5)
        n_neighbors = max(1, max_n_neighbors)
        
        pipeline_steps = [
            ('scaler', StandardScaler()),
            ('adasyn', ADASYN(n_neighbors=n_neighbors, random_state=42)),
            ('xgb', xgb.XGBClassifier(scale_pos_weight=spw, random_state=42, eval_metric='logloss'))
        ]
        if quick:
            param_grid = {
                'adasyn__sampling_strategy': [0.65, 0.7],
                'xgb__max_depth': [3, 5],
                'xgb__learning_rate': [0.05, 0.1],
                'xgb__n_estimators': [100, 150]
            }
        else:
            param_grid = {
                'adasyn__sampling_strategy': [0.6, 0.65, 0.7, 0.75],
                'xgb__max_depth': [3, 5, 7],
                'xgb__learning_rate': [0.01, 0.05, 0.1],
                'xgb__n_estimators': [100, 150, 200]
            }
    
    pipeline = ImbPipeline(pipeline_steps)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=0
    )
    
    try:
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    except Exception as e:
        # If grid search fails, try without ADASYN
        print(f"  Grid search failed, trying without ADASYN: {str(e)}")
        pipeline_steps = [
            ('scaler', StandardScaler()),
            ('xgb', xgb.XGBClassifier(scale_pos_weight=spw, random_state=42, eval_metric='logloss'))
        ]
        param_grid = {
            'xgb__max_depth': [3, 5],
            'xgb__learning_rate': [0.05, 0.1],
            'xgb__n_estimators': [100, 150]
        }
        pipeline = ImbPipeline(pipeline_steps)
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            scoring='roc_auc',
            cv=3,
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

def build_lstm_model(hp, input_shape):
    """Build LSTM model for hyperparameter tuning"""
    model = Sequential()
    u1 = hp.Int('lstm_units', 64, 128, 32)
    model.add(Bidirectional(LSTM(u1, return_sequences=True), input_shape=input_shape))
    model.add(BatchNormalization())
    dr = hp.Float('dropout_rate', 0.2, 0.4, 0.1)
    model.add(Dropout(dr))
    u2 = hp.Int('lstm_units_2', 32, 64, 16)
    model.add(LSTM(u2, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(dr))
    
    att = hp.Choice('attention_variant', ['custom', 'multihead', 'selfattention'])
    if att == 'custom':
        model.add(AttentionLayer())
    elif att == 'multihead':
        nh = hp.Int('num_heads', 1, 2, 1)
        kd = hp.Int('key_dim', 16, 32, 16)
        model.add(MultiHeadAttentionLayer(nh, kd))
    else:
        model.add(SelfAttentionLayer())
    
    du = hp.Int('dense_units', 16, 32, 16)
    drg = hp.Choice('dense_reg', [0.0, 0.001, 0.01])
    model.add(Dense(du, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(drg)))
    model.add(Dropout(dr))
    model.add(Dense(1, activation='sigmoid'))
    
    lr = hp.Choice('learning_rate', [0.001, 0.0005])
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    return model

def train_lstm_model(X_train, y_train, X_val, y_val, participant_settings, quick=True):
    """Train LSTM model with reduced trials for efficiency and better memory management"""
    # Prepare data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Apply resampling if needed
    pos = sum(y_train == 1)
    neg = sum(y_train == 0)
    minority_class_size = min(pos, neg)
    min_samples_for_adasyn = 10
    
    use_adasyn = (not participant_settings.get('no_resample', False) and 
                minority_class_size >= min_samples_for_adasyn)
    
    if use_adasyn:
        base_n_neighbors = participant_settings.get('n_neighbors', 5)
        max_n_neighbors = min(base_n_neighbors, minority_class_size - 1, 5)
        n_neighbors = max(1, max_n_neighbors)
        
        ada = ADASYN(
            sampling_strategy=0.7,
            n_neighbors=n_neighbors,
            random_state=42
        )
        try:
            X_train_scaled, y_train = ada.fit_resample(X_train_scaled, y_train)
        except Exception as e:
            print(f"  ADASYN failed for LSTM, using original data: {str(e)}")
            # If ADASYN fails, use original data
    
    # Reshape for LSTM
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_val_lstm = X_val_scaled.reshape((X_val_scaled.shape[0], X_val_scaled.shape[1], 1))
    
    # Calculate class weights
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = {i: cw[i] for i in range(len(cw))}
    
    # Create temporary directory for tuner
    tuner_dir = tempfile.mkdtemp(prefix='lstm_tuner_')
    
    try:
        # Hyperparameter tuning
        def build_model(hp):
            return build_lstm_model(hp, (X_train_lstm.shape[1], 1))
        
        tuner = kt.RandomSearch(
            build_model,
            kt.Objective('val_auc', direction='max'),
            max_trials=5 if quick else 20,
            executions_per_trial=1,
            directory=tuner_dir,
            project_name='temporal',
            overwrite=True
        )
        
        # Add callbacks to prevent overfitting and reduce memory
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=5,
                restore_best_weights=False  # Don't keep old weights in memory
            )
        ]
        
        tuner.search(
            X_train_lstm, y_train,
            epochs=30 if quick else 50,
            batch_size=32,
            validation_data=(X_val_lstm, y_val),
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0
        )
        
        best_model = tuner.get_best_models(num_models=1)[0]
        
        # Clean up tuner files immediately
        shutil.rmtree(tuner_dir, ignore_errors=True)
        
        # Return model and scaler
        return best_model, scaler
    except Exception as e:
        print(f"  LSTM training failed: {str(e)}")
        # Clean up on failure
        if os.path.exists(tuner_dir):
            shutil.rmtree(tuner_dir, ignore_errors=True)
        
        # Return a simple model if hyperparameter tuning fails
        simple_model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train_lstm.shape[1], 1)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(32, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            AttentionLayer(),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        simple_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC(name='auc')]
        )
        
        simple_model.fit(
            X_train_lstm, y_train,
            epochs=30,
            batch_size=32,
            validation_data=(X_val_lstm, y_val),
            class_weight=class_weight,
            verbose=0
        )
        
        return simple_model, scaler

def ensemble_predictions(y_xgb, y_lstm, y_true):
    """Find optimal ensemble weight and return predictions"""
    best_alpha, best_auc = None, -1
    for alpha in np.linspace(0, 1, 11):
        y_ensemble = alpha * y_xgb + (1 - alpha) * y_lstm
        try:
            auc = roc_auc_score(y_true, y_ensemble)
            if auc > best_auc:
                best_alpha, best_auc = alpha, auc
        except:
            pass
    
    if best_alpha is None:
        best_alpha = 0.5
    
    return best_alpha * y_xgb + (1 - best_alpha) * y_lstm, best_alpha

def save_model(model, path, model_type='xgb'):
    """Save model to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if model_type == 'xgb':
        joblib.dump(model, path)
    else:  # lstm
        model.save(path)

def load_model(path, model_type='xgb'):
    """Load model from disk with proper cleanup"""
    if not os.path.exists(path):
        print(f"    Model not found at: {path}")
        return None
    
    try:
        if model_type == 'xgb':
            model = joblib.load(path)
            print(f"    ✓ Loaded XGBoost model from: {path}")
            return model
        else:  # lstm
            # Clear any existing session before loading
            K.clear_session()
            tf.compat.v1.reset_default_graph()
            
            model = tf.keras.models.load_model(
                path,
                custom_objects={
                    "AttentionLayer": AttentionLayer,
                    "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
                    "SelfAttentionLayer": SelfAttentionLayer
                },
                compile=False  # Don't compile on load to save memory
            )
            
            # Recompile with minimal settings
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=[tf.keras.metrics.AUC(name='auc')]
            )
            
            print(f"    ✓ Loaded LSTM model from: {path}")
            return model
    except Exception as e:
        print(f"    ✗ Failed to load model from {path}: {str(e)}")
        return None

def analyze_global_model(total_days_list, force_retrain=False):
    """Analyze global model performance across different study durations.

    Now uses a per-total_days results cache. If cached, we do NOT load/train models.
    """
    print("\n" + "="*60)
    print("Analyzing Global Model (retraining for each duration)")
    print("="*60)

    # Load settings and common features
    settings = json.load(open('participant_settings.json'))
    common_features = build_feature_intersection(settings)

    results = {
        'total_days': [],
        'icc_bp_spike': [],
        'icc_healthy': [],
        'auc_mean': [],
        'auc_std': []
    }

    for total_days in total_days_list:
        print(f"\nAnalyzing {total_days} days...")

        # ---------- Cache check ----------
        cached = load_run_result('global_days', days=total_days)
        if cached is not None and not force_retrain:
            print("  ✓ Using cached results for this total_days")
            results['total_days'].append(total_days)
            results['icc_bp_spike'].append(cached['icc_bp_spike'])
            results['icc_healthy'].append(cached['icc_healthy'])
            results['auc_mean'].append(cached['auc_mean'])
            results['auc_std'].append(cached['auc_std'])
            continue
        # ---------------------------------

        # Prepare containers
        day_icc_spike, day_icc_healthy = [], []
        day_auc_mean, day_auc_std = [], []

        # Build combined train/test across participants
        all_train_dfs, all_test_dfs = [], []

        for s in settings:
            pid = s['pid']
            try:
                df = load_participant_df(pid)
            except:
                continue

            earliest = df['datetime_local'].min()
            window_end = earliest + pd.Timedelta(days=total_days)
            df_window = df[(df['datetime_local'] >= earliest) &
                           (df['datetime_local'] < window_end)].copy()
            if df_window.empty:
                continue

            unique_dates = sorted(df_window['datetime_local'].dt.date.unique())
            if len(unique_dates) < 2:
                continue

            train_dates, test_dates = split_days_75_25(unique_dates, seed=pid + total_days * 100)
            if train_dates is None or test_dates is None:
                continue

            train_df = df_window[df_window['datetime_local'].dt.date.isin(train_dates)]
            test_df  = df_window[df_window['datetime_local'].dt.date.isin(test_dates)]

            drop_features = s.get('drop', [])
            keep_features = [f for f in common_features if f not in drop_features]

            all_train_dfs.append(train_df[keep_features + ['BP_spike']])
            all_test_dfs.append(test_df[keep_features + ['BP_spike']])

            # ICC on training data
            if len(train_dates) >= 2:
                icc_features = [f for f in ICC_FEATURES if f in train_df.columns]

                spike_df = train_df[train_df['BP_spike'] == 1]
                if len(spike_df) > 50:
                    icc_vals = calculate_feature_icc_multiple_permutations(
                        train_df, icc_features, len(train_dates), condition='spike', n_permutations=5
                    )
                    if icc_vals:
                        day_icc_spike.append(np.mean(list(icc_vals.values())))

                healthy_df = train_df[train_df['BP_spike'] == 0]
                if len(healthy_df) > 50:
                    icc_vals = calculate_feature_icc_multiple_permutations(
                        train_df, icc_features, len(train_dates), condition='healthy', n_permutations=5
                    )
                    if icc_vals:
                        day_icc_healthy.append(np.mean(list(icc_vals.values())))

        # Train/eval if we actually have data
        if all_train_dfs and all_test_dfs:
            global_train = pd.concat(all_train_dfs, ignore_index=True)
            global_test  = pd.concat(all_test_dfs,  ignore_index=True)

            if (len(global_train) > 100 and global_train['BP_spike'].sum() > 10 and
                global_test['BP_spike'].sum() > 0 and (len(global_test) - global_test['BP_spike'].sum()) > 0):

                X_train = global_train.drop('BP_spike', axis=1).fillna(global_train.drop('BP_spike', axis=1).median())
                y_train = global_train['BP_spike']
                X_test  = global_test.drop('BP_spike', axis=1).fillna(global_train.drop('BP_spike', axis=1).median())
                y_test  = global_test['BP_spike']

                # If we're here, results are not cached; proceed to model load/train.
                model_dir  = os.path.join('results', 'temporal_models', 'global', f'days_{total_days}')
                xgb_path   = os.path.join(model_dir, 'xgb_model.joblib')
                lstm_path  = os.path.join(model_dir, 'lstm_model.keras')
                scaler_path= os.path.join(model_dir, 'lstm_scaler.joblib')

                try:
                    # XGB
                    xgb_model = load_model(xgb_path, 'xgb') if not force_retrain else None
                    if xgb_model is None:
                        print("  Training XGBoost...")
                        xgb_model = train_xgb_model(X_train, y_train, {'no_resample': False}, quick=True)
                        save_model(xgb_model, xgb_path, 'xgb')
                    y_xgb = xgb_model.predict_proba(X_test)[:, 1]

                    # LSTM
                    lstm_model = load_model(lstm_path, 'lstm') if not force_retrain else None
                    lstm_scaler = joblib.load(scaler_path) if (os.path.exists(scaler_path) and not force_retrain) else None
                    if lstm_model is None or lstm_scaler is None:
                        print("  Training LSTM...")
                        lstm_model, lstm_scaler = train_lstm_model(
                            X_train, y_train, X_test, y_test, {'no_resample': False}, quick=True
                        )
                        save_model(lstm_model, lstm_path, 'lstm')
                        joblib.dump(lstm_scaler, scaler_path)

                    X_test_scaled = lstm_scaler.transform(X_test)
                    X_test_lstm   = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
                    y_lstm        = lstm_model.predict(X_test_lstm, verbose=0).ravel()

                    # Ensemble + bootstrap
                    y_ens, alpha  = ensemble_predictions(y_xgb, y_lstm, y_test)
                    auc_mean, auc_std = bootstrap_auc(y_test.values, y_ens)
                    print(f"  Global AUC: {auc_mean:.3f} ± {auc_std:.3f} (α={alpha:.2f})")

                    day_auc_mean.append(auc_mean)
                    day_auc_std.append(auc_std)
                except Exception as e:
                    print(f"  Error training global model: {str(e)}")

        # Append + save per-unit result (cached for next time)
        results['total_days'].append(total_days)
        results['icc_bp_spike'].append(day_icc_spike)
        results['icc_healthy'].append(day_icc_healthy)
        results['auc_mean'].append(day_auc_mean)
        results['auc_std'].append(day_auc_std)

        save_run_result(
            'global_days',
            data={
                'icc_bp_spike': day_icc_spike,
                'icc_healthy': day_icc_healthy,
                'auc_mean': day_auc_mean,
                'auc_std': day_auc_std
            },
            days=total_days
        )

    return results

def split_fixed_test(dates, n_train_days, test_days=7):
    """Split dates with fixed last test_days for testing, first n_train_days for training"""
    dates = np.array(sorted(dates))
    total = len(dates)
    
    # Need at least test_days + 1 day for training
    if total < test_days + 1:
        return None, None
    
    # Take last test_days as test set
    test_dates = dates[-test_days:]
    
    # Take first n_train_days as training (excluding test dates)
    available_train_days = total - test_days
    if n_train_days > available_train_days:
        return None, None
    
    train_dates = dates[:n_train_days]
    
    return train_dates, test_dates

def analyze_global_model_fixed_test(train_days_list, test_days=7, force_retrain=False):
    """Analyze global model with fixed test set; caches per n_train_days."""
    print("\n" + "="*60)
    print(f"Analyzing Global Model (Fixed {test_days}-day test set)")
    print("="*60)

    settings = json.load(open('participant_settings.json'))
    common_features = build_feature_intersection(settings)

    results = {
        'train_days': [],
        'icc_bp_spike': [],
        'icc_healthy': [],
        'auc_mean': [],
        'auc_std': []
    }

    for n_train_days in train_days_list:
        print(f"\nAnalyzing {n_train_days} training days...")

        # ---------- Cache check ----------
        cached = load_run_result('global_fixed', train_days=n_train_days, test_days=test_days)
        if cached is not None and not force_retrain:
            print("  ✓ Using cached results for this train_days")
            results['train_days'].append(n_train_days)
            results['icc_bp_spike'].append(cached['icc_bp_spike'])
            results['icc_healthy'].append(cached['icc_healthy'])
            results['auc_mean'].append(cached['auc_mean'])
            results['auc_std'].append(cached['auc_std'])
            continue
        # ---------------------------------

        day_icc_spike, day_icc_healthy = [], []
        day_auc_mean, day_auc_std = [], []

        all_train_dfs, all_test_dfs = [], []

        for s in settings:
            pid = s['pid']
            try:
                df = load_participant_df(pid)
            except:
                continue

            unique_dates = sorted(df['datetime_local'].dt.date.unique())
            train_dates, test_dates = split_fixed_test(unique_dates, n_train_days, test_days)
            if train_dates is None:
                continue

            train_df = df[df['datetime_local'].dt.date.isin(train_dates)]
            test_df  = df[df['datetime_local'].dt.date.isin(test_dates)]

            drop_features = s.get('drop', [])
            keep_features = [f for f in common_features if f not in drop_features]

            all_train_dfs.append(train_df[keep_features + ['BP_spike']])
            all_test_dfs.append(test_df[keep_features + ['BP_spike']])

            # ICC on training data
            if len(train_dates) >= 2:
                icc_features = [f for f in ICC_FEATURES if f in train_df.columns]

                spike_df = train_df[train_df['BP_spike'] == 1]
                if len(spike_df) > 50:
                    icc_vals = calculate_feature_icc_multiple_permutations(
                        train_df, icc_features, len(train_dates), condition='spike', n_permutations=5
                    )
                    if icc_vals:
                        day_icc_spike.append(np.mean(list(icc_vals.values())))

                healthy_df = train_df[train_df['BP_spike'] == 0]
                if len(healthy_df) > 50:
                    icc_vals = calculate_feature_icc_multiple_permutations(
                        train_df, icc_features, len(train_dates), condition='healthy', n_permutations=5
                    )
                    if icc_vals:
                        day_icc_healthy.append(np.mean(list(icc_vals.values())))

        if all_train_dfs and all_test_dfs:
            global_train = pd.concat(all_train_dfs, ignore_index=True)
            global_test  = pd.concat(all_test_dfs,  ignore_index=True)

            if (len(global_train) > 100 and global_train['BP_spike'].sum() > 10 and
                global_test['BP_spike'].sum() > 0 and (len(global_test) - global_test['BP_spike'].sum()) > 0):

                X_train = global_train.drop('BP_spike', axis=1)
                y_train = global_train['BP_spike']
                X_test  = global_test.drop('BP_spike', axis=1)
                y_test  = global_test['BP_spike']

                train_median = X_train.median()
                X_train = X_train.fillna(train_median)
                X_test  = X_test.fillna(train_median)

                model_dir   = os.path.join('results', 'temporal_models_fixed', 'global', f'train_days_{n_train_days}')
                xgb_path    = os.path.join(model_dir, 'xgb_model.joblib')
                lstm_path   = os.path.join(model_dir, 'lstm_model.keras')
                scaler_path = os.path.join(model_dir, 'lstm_scaler.joblib')

                try:
                    # XGB
                    xgb_model = load_model(xgb_path, 'xgb') if not force_retrain else None
                    if xgb_model is None:
                        print("  Training XGBoost...")
                        xgb_model = train_xgb_model(X_train, y_train, {'no_resample': False}, quick=True)
                        save_model(xgb_model, xgb_path, 'xgb')
                    y_xgb = xgb_model.predict_proba(X_test)[:, 1]

                    # LSTM
                    lstm_model = load_model(lstm_path, 'lstm') if not force_retrain else None
                    lstm_scaler = joblib.load(scaler_path) if (os.path.exists(scaler_path) and not force_retrain) else None
                    if lstm_model is None or lstm_scaler is None:
                        print("  Training LSTM...")
                        lstm_model, lstm_scaler = train_lstm_model(
                            X_train, y_train, X_test, y_test, {'no_resample': False}, quick=True
                        )
                        save_model(lstm_model, lstm_path, 'lstm')
                        joblib.dump(lstm_scaler, scaler_path)

                    X_test_scaled = lstm_scaler.transform(X_test)
                    X_test_lstm   = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
                    y_lstm        = lstm_model.predict(X_test_lstm, verbose=0).ravel()

                    # Ensemble + bootstrap
                    y_ens, alpha  = ensemble_predictions(y_xgb, y_lstm, y_test)
                    auc_mean, auc_std = bootstrap_auc(y_test.values, y_ens)
                    print(f"  Global AUC: {auc_mean:.3f} ± {auc_std:.3f} (α={alpha:.2f})")

                    day_auc_mean.append(auc_mean)
                    day_auc_std.append(auc_std)
                except Exception as e:
                    print(f"  Error training global model: {str(e)}")

        # Append + cache
        results['train_days'].append(n_train_days)
        results['icc_bp_spike'].append(day_icc_spike)
        results['icc_healthy'].append(day_icc_healthy)
        results['auc_mean'].append(day_auc_mean)
        results['auc_std'].append(day_auc_std)

        save_run_result(
            'global_fixed',
            data={
                'icc_bp_spike': day_icc_spike,
                'icc_healthy': day_icc_healthy,
                'auc_mean': day_auc_mean,
                'auc_std': day_auc_std
            },
            train_days=n_train_days,
            test_days=test_days
        )

    return results

def calculate_feasible_train_days(settings, test_days=7):
    """Calculate feasible training days for each participant and globally"""
    participant_max_days = {}
    
    for s in settings:
        pid = s['pid']
        try:
            df = load_participant_df(pid)
            unique_dates = sorted(df['datetime_local'].dt.date.unique())
            max_train = len(unique_dates) - test_days
            if max_train > 0:
                participant_max_days[pid] = max_train
        except:
            participant_max_days[pid] = 0
    
    # Global maximum (at least one participant can do it)
    global_max = max(participant_max_days.values()) if participant_max_days else 0
    
    # Generate odd numbers from 1 to global_max
    global_train_days = [i for i in range(1, global_max + 1, 2)] if global_max > 0 else []
    
    # Per-participant feasible days
    participant_train_days = {}
    for pid, max_days in participant_max_days.items():
        participant_train_days[pid] = [i for i in range(1, max_days + 1, 2)] if max_days > 0 else []
    
    return global_train_days, participant_train_days

def analyze_personalized_model_fixed_test(pid, settings, train_days_list, test_days=7, force_retrain=False):
    """Analyze personalized model with fixed test set; caches per (pid, train_days)."""
    print(f"\nAnalyzing Participant {pid} (Fixed {test_days}-day test set)")

    p_settings = next((s for s in settings if s['pid'] == pid), None)
    if not p_settings:
        return None

    results = {
        'train_days': [],
        'icc_bp_spike': [],
        'icc_healthy': [],
        'auc_mean': [],
        'auc_std': []
    }

    try:
        df = load_participant_df(pid)
    except:
        return None

    all_dates = sorted(df['datetime_local'].dt.date.unique())
    if len(all_dates) < test_days + 1:
        print(f"  Insufficient data for {test_days}-day test set")
        return None

    test_dates_fixed = all_dates[-test_days:]
    test_df_fixed = df[df['datetime_local'].dt.date.isin(test_dates_fixed)]
    test_spikes = test_df_fixed['BP_spike'].sum()
    test_healthy = len(test_df_fixed) - test_spikes
    if test_spikes == 0 or test_healthy == 0:
        print(f"  Test set missing one class (spikes: {test_spikes}, healthy: {test_healthy})")
        return None

    print(f"  Fixed test set: {test_spikes} spikes, {test_healthy} healthy samples")
    print(f"  Testing training days: {train_days_list}")

    for n_train_days in train_days_list:
        print(f"\n  Processing {n_train_days} training days...")

        # ---------- Cache check ----------
        cached = load_run_result('personalized_fixed', pid=pid, train_days=n_train_days, test_days=test_days)
        if cached is not None and not force_retrain:
            print("    ✓ Using cached results for this (pid, train_days)")
            results['train_days'].append(n_train_days)
            results['icc_bp_spike'].append(cached['icc_bp_spike'])
            results['icc_healthy'].append(cached['icc_healthy'])
            results['auc_mean'].append(cached['auc_mean'])
            results['auc_std'].append(cached['auc_std'])
            continue
        # ---------------------------------

        train_dates, _ = split_fixed_test(all_dates, n_train_days, test_days)
        if train_dates is None:
            results['train_days'].append(n_train_days)
            results['icc_bp_spike'].append(np.nan)
            results['icc_healthy'].append(np.nan)
            results['auc_mean'].append(np.nan)
            results['auc_std'].append(np.nan)
            continue

        train_df = df[df['datetime_local'].dt.date.isin(train_dates)]

        train_spikes = train_df['BP_spike'].sum()
        train_healthy = len(train_df) - train_spikes
        if train_spikes < 5 or train_healthy < 5:
            print(f"    Insufficient training samples (spikes: {train_spikes}, healthy: {train_healthy})")
            results['train_days'].append(n_train_days)
            results['icc_bp_spike'].append(np.nan)
            results['icc_healthy'].append(np.nan)
            results['auc_mean'].append(np.nan)
            results['auc_std'].append(np.nan)
            continue

        # ICC
        icc_spike = np.nan
        icc_healthy = np.nan
        if len(train_dates) >= 2:
            drop_features = p_settings.get('drop', [])
            icc_features = [f for f in ICC_FEATURES if f not in drop_features and f in train_df.columns]

            if train_spikes > 20:
                icc_vals = calculate_feature_icc_multiple_permutations(
                    train_df, icc_features, len(train_dates), condition='spike', n_permutations=3
                )
                if icc_vals:
                    icc_spike = np.mean(list(icc_vals.values()))

            if train_healthy > 20:
                icc_vals = calculate_feature_icc_multiple_permutations(
                    train_df, icc_features, len(train_dates), condition='healthy', n_permutations=3
                )
                if icc_vals:
                    icc_healthy = np.mean(list(icc_vals.values()))

        # Features + arrays
        drop_features = p_settings.get('drop', [])
        feature_cols = [f for f in FEATURES if f not in drop_features and
                        f in test_df_fixed.columns and f in train_df.columns]

        auc_mean = np.nan
        auc_std = np.nan

        if feature_cols:
            X_train_np = train_df[feature_cols].values
            y_train_np = train_df['BP_spike'].values
            X_test_np  = test_df_fixed[feature_cols].values
            y_test_np  = test_df_fixed['BP_spike'].values

            train_median = np.nanmedian(X_train_np, axis=0)
            X_train_np = np.nan_to_num(X_train_np, nan=train_median)
            X_test_np  = np.nan_to_num(X_test_np,  nan=train_median)

            # Model paths
            model_dir   = os.path.join('results', 'temporal_models_fixed', 'personalized',
                                       f'pid_{pid}', f'train_days_{n_train_days}')
            xgb_path    = os.path.join(model_dir, 'xgb_model.joblib')
            lstm_path   = os.path.join(model_dir, 'lstm_model.keras')
            scaler_path = os.path.join(model_dir, 'lstm_scaler.joblib')

            try:
                # XGB
                xgb_model = load_model(xgb_path, 'xgb') if not force_retrain else None
                if xgb_model is None:
                    print(f"    Training new XGBoost model...")
                    X_train_df = pd.DataFrame(X_train_np, columns=feature_cols)
                    xgb_model = train_xgb_model(X_train_df, pd.Series(y_train_np), p_settings, quick=True)
                    save_model(xgb_model, xgb_path, 'xgb')

                X_test_df = pd.DataFrame(X_test_np, columns=feature_cols)
                y_xgb = xgb_model.predict_proba(X_test_df)[:, 1]

                # LSTM
                lstm_model = load_model(lstm_path, 'lstm') if not force_retrain else None
                lstm_scaler = joblib.load(scaler_path) if (os.path.exists(scaler_path) and not force_retrain) else None

                if lstm_model is None or lstm_scaler is None:
                    print(f"    Training new LSTM model...")
                    X_train_df = pd.DataFrame(X_train_np, columns=feature_cols)
                    X_test_df  = pd.DataFrame(X_test_np,  columns=feature_cols)
                    lstm_model, lstm_scaler = train_lstm_model(
                        X_train_df, pd.Series(y_train_np),
                        X_test_df,  pd.Series(y_test_np),
                        p_settings, quick=True
                    )
                    save_model(lstm_model, lstm_path, 'lstm')
                    joblib.dump(lstm_scaler, scaler_path)

                X_test_df = pd.DataFrame(X_test_np, columns=feature_cols)
                X_test_scaled = lstm_scaler.transform(X_test_df)
                X_test_lstm   = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
                y_lstm        = lstm_model.predict(X_test_lstm, verbose=0).ravel()

                # Ensemble + bootstrap
                y_ens = ensemble_predictions(y_xgb, y_lstm, y_test_np)[0]
                auc_mean, auc_std = bootstrap_auc(y_test_np, y_ens)
                print(f"    AUC: {auc_mean:.3f} ± {auc_std:.3f}")

            except Exception as e:
                print(f"    ✗ Error: {str(e)}")

        # Append + cache
        results['train_days'].append(n_train_days)
        results['icc_bp_spike'].append(icc_spike)
        results['icc_healthy'].append(icc_healthy)
        results['auc_mean'].append(auc_mean)
        results['auc_std'].append(auc_std)

        save_run_result(
            'personalized_fixed',
            data={
                'icc_bp_spike': icc_spike,
                'icc_healthy': icc_healthy,
                'auc_mean': auc_mean,
                'auc_std': auc_std
            },
            pid=pid, train_days=n_train_days, test_days=test_days
        )

    return results

def analyze_personalized_model(pid, settings, total_days_list, force_retrain=False):
    """Analyze personalized model for a single participant (varying window); caches per (pid, total_days)."""
    print(f"\nAnalyzing Participant {pid} (retraining for each duration)")

    p_settings = next((s for s in settings if s['pid'] == pid), None)
    if not p_settings:
        return None

    results = {
        'total_days': [],
        'icc_bp_spike': [],
        'icc_healthy': [],
        'auc_mean': [],
        'auc_std': []
    }

    try:
        df = load_participant_df(pid)
    except:
        return None

    total_spikes = df['BP_spike'].sum()
    spike_rate = total_spikes / len(df)
    print(f"PID {pid}: Total BP spikes: {total_spikes} ({spike_rate:.2%})")

    for total_days in total_days_list:
        print(f"\n  Processing {total_days} days...")

        # ---------- Cache check ----------
        cached = load_run_result('personalized_days', pid=pid, days=total_days)
        if cached is not None and not force_retrain:
            print("    ✓ Using cached results for this (pid, total_days)")
            results['total_days'].append(total_days)
            results['icc_bp_spike'].append(cached['icc_bp_spike'])
            results['icc_healthy'].append(cached['icc_healthy'])
            results['auc_mean'].append(cached['auc_mean'])
            results['auc_std'].append(cached['auc_std'])
            continue
        # ---------------------------------

        # Window
        earliest = df['datetime_local'].min()
        window_end = earliest + pd.Timedelta(days=total_days)
        df_window = df[(df['datetime_local'] >= earliest) &
                       (df['datetime_local'] < window_end)].copy()

        if df_window.empty:
            results['total_days'].append(total_days)
            results['icc_bp_spike'].append(np.nan)
            results['icc_healthy'].append(np.nan)
            results['auc_mean'].append(np.nan)
            results['auc_std'].append(np.nan)
            save_run_result('personalized_days',
                            data={'icc_bp_spike': np.nan, 'icc_healthy': np.nan, 'auc_mean': np.nan, 'auc_std': np.nan},
                            pid=pid, days=total_days)
            continue

        unique_dates = sorted(df_window['datetime_local'].dt.date.unique())
        if len(unique_dates) < 2:
            results['total_days'].append(total_days)
            results['icc_bp_spike'].append(np.nan)
            results['icc_healthy'].append(np.nan)
            results['auc_mean'].append(np.nan)
            results['auc_std'].append(np.nan)
            save_run_result('personalized_days',
                            data={'icc_bp_spike': np.nan, 'icc_healthy': np.nan, 'auc_mean': np.nan, 'auc_std': np.nan},
                            pid=pid, days=total_days)
            continue

        train_dates, test_dates = split_days_75_25(unique_dates, seed=pid + total_days * 1000)
        if train_dates is None or test_dates is None:
            results['total_days'].append(total_days)
            results['icc_bp_spike'].append(np.nan)
            results['icc_healthy'].append(np.nan)
            results['auc_mean'].append(np.nan)
            results['auc_std'].append(np.nan)
            save_run_result('personalized_days',
                            data={'icc_bp_spike': np.nan, 'icc_healthy': np.nan, 'auc_mean': np.nan, 'auc_std': np.nan},
                            pid=pid, days=total_days)
            continue

        train_df = df_window[df_window['datetime_local'].dt.date.isin(train_dates)]
        test_df  = df_window[df_window['datetime_local'].dt.date.isin(test_dates)]

        # ICC
        icc_spike = np.nan
        icc_healthy = np.nan
        if len(train_dates) >= 2:
            drop_features = p_settings.get('drop', [])
            icc_features = [f for f in ICC_FEATURES if f not in drop_features and f in train_df.columns]

            spike_count = train_df['BP_spike'].sum()
            if spike_count > 20:
                icc_vals = calculate_feature_icc_multiple_permutations(
                    train_df, icc_features, len(train_dates), condition='spike', n_permutations=3
                )
                if icc_vals:
                    icc_spike = np.mean(list(icc_vals.values()))

            healthy_count = len(train_df) - spike_count
            if healthy_count > 20:
                icc_vals = calculate_feature_icc_multiple_permutations(
                    train_df, icc_features, len(train_dates), condition='healthy', n_permutations=3
                )
                if icc_vals:
                    icc_healthy = np.mean(list(icc_vals.values()))

        # Train/eval
        auc_mean = np.nan
        auc_std = np.nan
        test_spikes = test_df['BP_spike'].sum()
        test_healthy = len(test_df) - test_spikes
        train_spikes = train_df['BP_spike'].sum()

        if (len(test_df) > 10 and test_spikes > 0 and test_healthy > 0 and
            train_spikes > 5 and (len(train_df) - train_spikes) > 5):

            drop_features = p_settings.get('drop', [])
            feature_cols = [f for f in FEATURES if f not in drop_features and f in test_df.columns and f in train_df.columns]

            X_train_np = train_df[feature_cols].values
            y_train_np = train_df['BP_spike'].values
            X_test_np  = test_df[feature_cols].values
            y_test_np  = test_df['BP_spike'].values

            train_median = np.nanmedian(X_train_np, axis=0)
            X_train_np = np.nan_to_num(X_train_np, nan=train_median)
            X_test_np  = np.nan_to_num(X_test_np,  nan=train_median)

            model_dir   = os.path.join('results', 'temporal_models', 'personalized', f'pid_{pid}', f'days_{total_days}')
            xgb_path    = os.path.join(model_dir, 'xgb_model.joblib')
            lstm_path   = os.path.join(model_dir, 'lstm_model.keras')
            scaler_path = os.path.join(model_dir, 'lstm_scaler.joblib')

            try:
                # XGB
                xgb_model = load_model(xgb_path, 'xgb') if not force_retrain else None
                if xgb_model is None:
                    print(f"    Training new XGBoost model...")
                    X_train_df = pd.DataFrame(X_train_np, columns=feature_cols)
                    xgb_model = train_xgb_model(X_train_df, pd.Series(y_train_np), p_settings, quick=True)
                    save_model(xgb_model, xgb_path, 'xgb')

                X_test_df = pd.DataFrame(X_test_np, columns=feature_cols)
                y_xgb = xgb_model.predict_proba(X_test_df)[:, 1]

                # LSTM
                lstm_model = load_model(lstm_path, 'lstm') if not force_retrain else None
                lstm_scaler = joblib.load(scaler_path) if (os.path.exists(scaler_path) and not force_retrain) else None
                if lstm_model is None or lstm_scaler is None:
                    print(f"    Training new LSTM model...")
                    X_train_df = pd.DataFrame(X_train_np, columns=feature_cols)
                    X_test_df  = pd.DataFrame(X_test_np,  columns=feature_cols)
                    lstm_model, lstm_scaler = train_lstm_model(
                        X_train_df, pd.Series(y_train_np),
                        X_test_df,  pd.Series(y_test_np),
                        p_settings, quick=True
                    )
                    save_model(lstm_model, lstm_path, 'lstm')
                    joblib.dump(lstm_scaler, scaler_path)

                X_test_df = pd.DataFrame(X_test_np, columns=feature_cols)
                X_test_scaled = lstm_scaler.transform(X_test_df)
                X_test_lstm   = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
                y_lstm        = lstm_model.predict(X_test_lstm, verbose=0).ravel()

                y_ens = ensemble_predictions(y_xgb, y_lstm, y_test_np)[0]
                auc_mean, auc_std = bootstrap_auc(y_test_np, y_ens)
                print(f"    AUC: {auc_mean:.3f} ± {auc_std:.3f}")
            except Exception as e:
                print(f"    ✗ Error processing model for {total_days} days: {str(e)}")

        # Append + cache this unit
        results['total_days'].append(total_days)
        results['icc_bp_spike'].append(icc_spike)
        results['icc_healthy'].append(icc_healthy)
        results['auc_mean'].append(auc_mean)
        results['auc_std'].append(auc_std)

        save_run_result(
            'personalized_days',
            data={
                'icc_bp_spike': icc_spike,
                'icc_healthy': icc_healthy,
                'auc_mean': auc_mean,
                'auc_std': auc_std
            },
            pid=pid, days=total_days
        )

    return results

def add_jitter(values, jitter_amount=0.002):
    """Add small jitter to values to avoid overlapping points"""
    if len(values) == 0:
        return values
    return np.array(values) + np.random.normal(0, jitter_amount, len(values))

import math
import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import math, gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

def plot_personalized_grid(all_results, pids, output_path):
    n_cols = 5
    # filter
    valid = [pid for pid in pids 
             if pid in all_results and has_sufficient_data(all_results[pid])]
    if not valid:
        print("no data"); return

    total = len(valid)*2
    n_rows = math.ceil(total / n_cols)
    plots_last = total % n_cols or n_cols
    blank      = (n_cols - plots_last) / 2  # fractional

    fig = plt.figure(figsize=(25, 3*n_rows))
    # 1) main GS
    outer = GridSpec(n_rows, n_cols, figure=fig,
                     left=0.05, right=0.95, top=0.95, bottom=0.05,
                     hspace=0.35, wspace=0.25)

    def draw(idx, ax):
        # map idx → (pid_idx, is_auc) etc and plot exactly as before...
        pid_idx = idx // 2
        is_auc  = (idx % 2 == 1)
        pid     = valid[pid_idx]
        R       = all_results[pid]
        days    = np.array(R['total_days'])
        if not is_auc:
            # ICC plot (idx even)
            vs = np.array(R['icc_bp_spike']); valid_spike = ~np.isnan(vs)
            if valid_spike.any():
                vv = vs[valid_spike]
                vv[vv>0.98] = add_jitter(vv[vv>0.98])
                ax.plot(days[valid_spike], vv, 'o-', color='#e74c3c', ms=3, lw=1)
            vh = np.array(R['icc_healthy']); valid_healthy = ~np.isnan(vh)
            if valid_healthy.any():
                vv = vh[valid_healthy]
                vv[vv>0.98] = add_jitter(vv[vv>0.98])
                ax.plot(days[valid_healthy], vv, 's-', color='#3498db', ms=3, lw=1)
            ax.set(title=f'Participant {pid}: ICC', xlabel='Days', ylabel='ICC', ylim=(-0.05,1.05))
        else:
            # AUC plot (idx odd)
            vm = np.array(R['auc_mean']); valid_auc = ~np.isnan(vm)
            if valid_auc.any():
                ax.errorbar(days[valid_auc], vm[valid_auc],
                            yerr=np.array(R['auc_std'])[valid_auc],
                            marker='o', ms=3, capsize=2, capthick=0.8,
                            color='#27ae60', lw=1)
            ax.set(title=f'Participant {pid}: AUC', xlabel='Days', ylabel='AUC', ylim=(-0.05,1.05))
        ax.tick_params(labelsize=6); ax.grid(alpha=0.3)

    # 2) draw all *full* rows except the last if partial
    end_full = (n_rows-1)*n_cols if plots_last!=n_cols else n_rows*n_cols
    for idx in range(min(end_full, total)):
        r, c = divmod(idx, n_cols)
        ax = fig.add_subplot(outer[r, c])
        draw(idx, ax)

    # 3) nest for last row if it's partial
    if plots_last != n_cols:
        # grab the entire last row of the main GS
        last_row_spec = outer[n_rows-1, :]
        # make an inner GS with blank margins on left/right
        inner = GridSpecFromSubplotSpec(
            1, plots_last+2,
            subplot_spec=last_row_spec,
            width_ratios=[blank] + [1]*plots_last + [blank],
            wspace=0.25
        )
        # only fill the middle plots_last slots
        base_idx = (n_rows-1)*n_cols
        for i in range(plots_last):
            ax = fig.add_subplot(inner[0, i+1])
            idx = base_idx + i
            if idx < total:
                draw(idx, ax)

    # 4) legend + title + save
    legend_elems = [
        Line2D([0],[0], marker='o', color='#e74c3c', label='BP Spike'),
        Line2D([0],[0], marker='s', color='#3498db', label='Healthy'),
        Line2D([0],[0], marker='o', color='#27ae60', label='AUC')
    ]
    fig.legend(handles=legend_elems, loc='upper right',
               bbox_to_anchor=(0.98,0.99), fontsize=10,
               frameon=True, fancybox=True, shadow=True)
    plt.suptitle('Personalized Models: Temporal Analysis', y=0.98, fontsize=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    gc.collect()
    print(f"Saved {n_cols}×{n_rows} grid; last row had {plots_last} plots perfectly centered.")


def has_sufficient_data(results):
    """Check if a participant has sufficient data for plotting - requires AUC data"""
    # Check if there's any valid AUC data (this is now required)
    auc_valid = any(not np.isnan(val) for val in results['auc_mean'])
    
    # Only include participants with AUC data
    return auc_valid

def plot_figure4_style(results, title, output_path):
    """Create Figure 4 style plot with ICC and AUC panels"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Panel (a): ICC vs days
    days = results['total_days']
    
    # Process ICC data
    icc_spike_means = []
    icc_spike_errors = []
    icc_healthy_means = []
    icc_healthy_errors = []
    
    for i, d in enumerate(days):
        # BP spike ICC
        if isinstance(results['icc_bp_spike'][i], list):
            vals = [v for v in results['icc_bp_spike'][i] if not np.isnan(v)]
            if vals:
                icc_spike_means.append(np.mean(vals))
                icc_spike_errors.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            else:
                icc_spike_means.append(np.nan)
                icc_spike_errors.append(0)
        else:
            icc_spike_means.append(results['icc_bp_spike'][i])
            icc_spike_errors.append(0)
        
        # Healthy ICC
        if isinstance(results['icc_healthy'][i], list):
            vals = [v for v in results['icc_healthy'][i] if not np.isnan(v)]
            if vals:
                icc_healthy_means.append(np.mean(vals))
                icc_healthy_errors.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            else:
                icc_healthy_means.append(np.nan)
                icc_healthy_errors.append(0)
        else:
            icc_healthy_means.append(results['icc_healthy'][i])
            icc_healthy_errors.append(0)
    
    # Plot ICC with proper y-axis limits
    valid_spike = ~np.isnan(icc_spike_means)
    valid_healthy = ~np.isnan(icc_healthy_means)
    
    if np.any(valid_spike):
        spike_means = np.array(icc_spike_means)[valid_spike]
        spike_means[spike_means > 0.98] = add_jitter(spike_means[spike_means > 0.98])
        
        ax1.errorbar(np.array(days)[valid_spike], spike_means,
                     yerr=np.array(icc_spike_errors)[valid_spike],
                     marker='o', markersize=6, capsize=4, capthick=1.5,
                     color='#e74c3c', linewidth=2)
    
    if np.any(valid_healthy):
        healthy_means = np.array(icc_healthy_means)[valid_healthy]
        healthy_means[healthy_means > 0.98] = add_jitter(healthy_means[healthy_means > 0.98])
        
        ax1.errorbar(np.array(days)[valid_healthy], healthy_means,
                     yerr=np.array(icc_healthy_errors)[valid_healthy],
                     marker='s', markersize=6, capsize=4, capthick=1.5,
                     color='#3498db', linewidth=2)
    
    ax1.set_xlabel('Total days of data used')
    ax1.set_ylabel('ICC')
    ax1.set_title('(a) Feature Reliability')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Add note if no data
    if not np.any(valid_spike) and not np.any(valid_healthy):
        ax1.text(0.5, 0.5, 'Insufficient data for ICC calculation', 
                 transform=ax1.transAxes, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel (b): AUC vs days with bootstrapped error bars
    auc_means = []
    auc_errors = []
    
    for i, d in enumerate(days):
        if 'auc_mean' in results and 'auc_std' in results:
            # New format with bootstrapped results
            if isinstance(results['auc_mean'][i], list):
                vals = [v for v in results['auc_mean'][i] if not np.isnan(v)]
                stds = [v for v in results['auc_std'][i] if not np.isnan(v)]
                if vals:
                    # For multiple participants, combine uncertainties
                    auc_means.append(np.mean(vals))
                    # Combined standard error
                    if stds:
                        pooled_std = np.sqrt(np.mean(np.array(stds)**2))
                        auc_errors.append(pooled_std)
                    else:
                        auc_errors.append(0)
                else:
                    auc_means.append(np.nan)
                    auc_errors.append(0)
            else:
                auc_means.append(results['auc_mean'][i])
                auc_errors.append(results['auc_std'][i] if 'auc_std' in results else 0)
        else:
            # Old format compatibility
            if isinstance(results.get('auc_combined', [])[i], list):
                vals = [v for v in results['auc_combined'][i] if not np.isnan(v)]
                if vals:
                    auc_means.append(np.mean(vals))
                    auc_errors.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
                else:
                    auc_means.append(np.nan)
                    auc_errors.append(0)
            else:
                auc_means.append(results.get('auc_combined', [np.nan])[i])
                auc_errors.append(0)
    
    valid_auc = ~np.isnan(auc_means)
    
    if np.any(valid_auc):
        ax2.errorbar(np.array(days)[valid_auc], np.array(auc_means)[valid_auc],
                     yerr=np.array(auc_errors)[valid_auc],
                     marker='o', markersize=6, capsize=4, capthick=1.5,
                     color='#27ae60', linewidth=2)
    else:
        ax2.text(0.5, 0.5, 'Insufficient test data\nfor AUC calculation', 
                 transform=ax2.transAxes, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2.set_xlabel('Total days of data used')
    ax2.set_ylabel('AUC')
    ax2.set_title('(b) Model Performance')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    
    # Create custom legend handles for figure-level legend
    from matplotlib.lines import Line2D
    
    legend_elements = []
    if np.any(valid_spike):
        legend_elements.append(Line2D([0], [0], marker='o', color='#e74c3c', linewidth=2, 
                                    markersize=6, label='BP Spike', linestyle='-'))
    if np.any(valid_healthy):
        legend_elements.append(Line2D([0], [0], marker='s', color='#3498db', linewidth=2, 
                                    markersize=6, label='Healthy', linestyle='-'))
    if np.any(valid_auc):
        legend_elements.append(Line2D([0], [0], marker='o', color='#27ae60', linewidth=2, 
                                    markersize=6, label='AUC', linestyle='-'))
    
    # Add master legend outside the plot area if there are elements to show
    if legend_elements:
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95), 
                   fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Fixed title spacing - adjusted y value for better positioning
    plt.suptitle(title, fontsize=14, y=0.94)
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make room for legend
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)  # Explicitly close the figure
    plt.close('all')  # Close all figures to be safe
    gc.collect()
    
    print(f"Saved figure to {output_path}")

def load_existing_results(results_path):
    """Load existing pickle results if they exist"""
    if os.path.exists(results_path):
        print(f"Loading existing results from {results_path}")
        try:
            with open(results_path, 'rb') as f:
                data = pickle.load(f)
            return data.get('global'), data.get('personalized'), data.get('total_days_list')
        except Exception as e:
            print(f"Failed to load existing results: {str(e)}")
            return None, None, None
    return None, None, None

'''
def main():
    """Main function with batch processing for memory efficiency"""
    # Define study durations to test
    total_days_list = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
    
    # Load settings
    with open('participant_settings.json', 'r') as f:
        settings = json.load(f)
    
    # Output directory
    output_dir = os.path.join('results', 'temporal_analysis_retrained')
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we should force retrain
    import sys
    force_retrain = '--force-retrain' in sys.argv
    force_recompute = '--force-recompute' in sys.argv
    
    # Process in batches to avoid memory issues
    batch_processing = '--batch' in sys.argv
    batch_size = 3  # Process 3 participants at a time
    
    print("\n" + "="*60)
    print("TEMPORAL ANALYSIS - RETRAINING FOR EACH DURATION")
    if force_retrain:
        print("⚠️  FORCE RETRAIN MODE: All models will be retrained")
    else:
        print("✓ SMART MODE: Loading existing models when available")
    if batch_processing:
        print(f"✓ BATCH MODE: Processing {batch_size} participants at a time")
    print("="*60)

    pids = [s['pid'] for s in settings]
    
    # Try to load existing results first
    results_path = os.path.join(output_dir, 'temporal_results_retrained.pkl')
    global_results, personalized_results, existing_days_list = load_existing_results(results_path)
    
    # Check if we need to recompute
    need_recompute = (force_recompute or 
                     global_results is None or 
                     personalized_results is None or
                     existing_days_list != total_days_list)
    
    if not need_recompute:
        print("✓ Using existing results from pickle file")
        print("  Use --force-recompute to regenerate results")
    else:
        print("Computing new results...")
        
        # Check existing models
        model_base_dir = os.path.join('results', 'temporal_models')
        if os.path.exists(model_base_dir):
            print(f"\nModel directory exists: {model_base_dir}")
            # Count existing models
            global_count = 0
            personalized_count = 0
            
            global_dir = os.path.join(model_base_dir, 'global')
            if os.path.exists(global_dir):
                for days_dir in os.listdir(global_dir):
                    if days_dir.startswith('days_'):
                        model_files = os.listdir(os.path.join(global_dir, days_dir))
                        if 'xgb_model.joblib' in model_files:
                            global_count += 1
            
            personalized_dir = os.path.join(model_base_dir, 'personalized')
            if os.path.exists(personalized_dir):
                for pid_dir in os.listdir(personalized_dir):
                    if pid_dir.startswith('pid_'):
                        pid_path = os.path.join(personalized_dir, pid_dir)
                        for days_dir in os.listdir(pid_path):
                            if days_dir.startswith('days_'):
                                model_files = os.listdir(os.path.join(pid_path, days_dir))
                                if 'xgb_model.joblib' in model_files:
                                    personalized_count += 1
            
            print(f"Found {global_count} existing global models")
            print(f"Found {personalized_count} existing personalized models")
        else:
            print(f"\nNo existing model directory found at: {model_base_dir}")
            print("All models will be trained from scratch")
        
        # 1. Analyze global model
        print("\n" + "="*60)
        print("Analyzing Global Model")
        print("="*60)
        
        global_results = analyze_global_model(total_days_list, force_retrain=force_retrain)
        
        # Clear memory after global model
        aggressive_cleanup()
        
        # 2. Analyze personalized models
        print("\n" + "="*60)
        print("Analyzing Personalized Models")
        print("="*60)
        
        personalized_results = {}
        
        if batch_processing:
            # Process in batches
            for i in range(0, len(pids), batch_size):
                batch_pids = pids[i:i+batch_size]
                print(f"\nProcessing batch {i//batch_size + 1}: PIDs {batch_pids}")
                
                for pid in batch_pids:
                    results = analyze_personalized_model(pid, settings, total_days_list, force_retrain=force_retrain)
                    if results:
                        personalized_results[pid] = results
                    
                    # Clear memory after each participant
                    aggressive_cleanup()
                
                print(f"\nBatch {i//batch_size + 1} complete. Deep cleaning memory...")
                aggressive_cleanup()
                
                # Optional: Force Python to release memory back to OS
                try:
                    import ctypes
                    libc = ctypes.CDLL("libc.so.6")
                    libc.malloc_trim(0)
                except:
                    pass
                
                gc.collect()
                print_memory_usage("after deep clean")
        else:
            # Original processing (all at once)
            for pid in pids:
                results = analyze_personalized_model(pid, settings, total_days_list, force_retrain=force_retrain)
                if results:
                    personalized_results[pid] = results
        
        # Save raw results
        with open(results_path, 'wb') as f:
            pickle.dump({
                'global': global_results,
                'personalized': personalized_results,
                'total_days_list': total_days_list
            }, f)
        print(f"Saved results to {results_path}")
    
    # Generate plots from results (whether loaded or computed)
    print("\n" + "="*60)
    print("Generating Plots")
    print("="*60)
    
    # Create global plot
    if global_results:
        plot_figure4_style(
            global_results,
            'Global Model: Temporal Analysis',
            os.path.join(output_dir, 'global_temporal_analysis_retrained.png')
        )
    
    # Create individual personalized plots
    if personalized_results:
        for pid, results in personalized_results.items():
            plot_figure4_style(
                results,
                f'Participant {pid}: Temporal Analysis',
                os.path.join(output_dir, f'pid_{pid}_temporal_analysis_retrained.png')
            )
        
        # Create combined grid plot
        plot_personalized_grid(
            personalized_results,
            pids,
            os.path.join(output_dir, 'all_personalized_temporal_analysis_retrained.png')
        )
    
    print("\n✅ Temporal analysis (with retraining) completed!")
    print(f"Results saved to {output_dir}")
'''

def plot_fixed_test_analysis(results, title, output_path):
    """Create plot for fixed test set analysis showing training data requirements"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Use train_days instead of total_days
    days = results['train_days']
    
    # Panel (a): ICC vs training days
    icc_spike_means = []
    icc_spike_errors = []
    icc_healthy_means = []
    icc_healthy_errors = []
    
    for i, d in enumerate(days):
        # BP spike ICC
        if isinstance(results['icc_bp_spike'][i], list):
            vals = [v for v in results['icc_bp_spike'][i] if not np.isnan(v)]
            if vals:
                icc_spike_means.append(np.mean(vals))
                icc_spike_errors.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            else:
                icc_spike_means.append(np.nan)
                icc_spike_errors.append(0)
        else:
            icc_spike_means.append(results['icc_bp_spike'][i])
            icc_spike_errors.append(0)
        
        # Healthy ICC
        if isinstance(results['icc_healthy'][i], list):
            vals = [v for v in results['icc_healthy'][i] if not np.isnan(v)]
            if vals:
                icc_healthy_means.append(np.mean(vals))
                icc_healthy_errors.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            else:
                icc_healthy_means.append(np.nan)
                icc_healthy_errors.append(0)
        else:
            icc_healthy_means.append(results['icc_healthy'][i])
            icc_healthy_errors.append(0)
    
    # Plot ICC
    valid_spike = ~np.isnan(icc_spike_means)
    valid_healthy = ~np.isnan(icc_healthy_means)
    
    if np.any(valid_spike):
        spike_means = np.array(icc_spike_means)[valid_spike]
        spike_means[spike_means > 0.98] = add_jitter(spike_means[spike_means > 0.98])
        
        ax1.errorbar(np.array(days)[valid_spike], spike_means,
                     yerr=np.array(icc_spike_errors)[valid_spike],
                     marker='o', markersize=6, capsize=4, capthick=1.5,
                     color='#e74c3c', linewidth=2, label='BP Spike')
    
    if np.any(valid_healthy):
        healthy_means = np.array(icc_healthy_means)[valid_healthy]
        healthy_means[healthy_means > 0.98] = add_jitter(healthy_means[healthy_means > 0.98])
        
        ax1.errorbar(np.array(days)[valid_healthy], healthy_means,
                     yerr=np.array(icc_healthy_errors)[valid_healthy],
                     marker='s', markersize=6, capsize=4, capthick=1.5,
                     color='#3498db', linewidth=2, label='Healthy')
    
    ax1.set_xlabel('Training days (fixed 7-day test set)')
    ax1.set_ylabel('ICC')
    ax1.set_title('(a) Feature Reliability')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    
    # Panel (b): AUC vs training days
    auc_means = []
    auc_errors = []
    
    for i, d in enumerate(days):
        if isinstance(results['auc_mean'][i], list):
            vals = [v for v in results['auc_mean'][i] if not np.isnan(v)]
            stds = [v for v in results['auc_std'][i] if not np.isnan(v)]
            if vals:
                auc_means.append(np.mean(vals))
                if stds:
                    pooled_std = np.sqrt(np.mean(np.array(stds)**2))
                    auc_errors.append(pooled_std)
                else:
                    auc_errors.append(0)
            else:
                auc_means.append(np.nan)
                auc_errors.append(0)
        else:
            auc_means.append(results['auc_mean'][i])
            auc_errors.append(results['auc_std'][i] if 'auc_std' in results else 0)
    
    valid_auc = ~np.isnan(auc_means)
    
    if np.any(valid_auc):
        ax2.errorbar(np.array(days)[valid_auc], np.array(auc_means)[valid_auc],
                     yerr=np.array(auc_errors)[valid_auc],
                     marker='o', markersize=6, capsize=4, capthick=1.5,
                     color='#27ae60', linewidth=2, label='AUC')
        
        # Add horizontal line at 0.85 to show "good" performance threshold
        ax2.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5, label='Target AUC')
    
    ax2.set_xlabel('Training days (fixed 7-day test set)')
    ax2.set_ylabel('AUC')
    ax2.set_title('(b) Model Performance')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')
    
    plt.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    gc.collect()
    
    print(f"Saved figure to {output_path}")

def plot_personalized_fixed_grid(all_results, pids, output_path):
    """Create grid plot for personalized models with fixed test set"""
    n_cols = 5
    # Filter valid participants
    valid = [pid for pid in pids 
            if pid in all_results and all_results[pid] is not None 
            and 'auc_mean' in all_results[pid]
            and any(not np.isnan(val) for val in all_results[pid]['auc_mean'])]

    # Sort participant IDs numerically
    valid = sorted(valid, key=int)
    
    if not valid:
        print("No valid data for personalized fixed test grid")
        return
    
    total = len(valid) * 2  # Two plots per participant (ICC and AUC)
    n_rows = math.ceil(total / n_cols)
    plots_last = total % n_cols or n_cols
    blank = (n_cols - plots_last) / 2  # fractional blank space
    
    fig = plt.figure(figsize=(25, 3*n_rows))
    
    # Main GridSpec
    outer = GridSpec(n_rows, n_cols, figure=fig,
                     left=0.05, right=0.95, top=0.95, bottom=0.05,
                     hspace=0.35, wspace=0.25)
    
    def draw(idx, ax):
        """Draw individual subplot"""
        pid_idx = idx // 2
        is_auc = (idx % 2 == 1)
        pid = valid[pid_idx]
        R = all_results[pid]
        days = np.array(R['train_days'])  # Note: using train_days not total_days
        
        if not is_auc:
            # ICC plot
            vs = np.array(R['icc_bp_spike'])
            valid_spike = ~np.isnan(vs)
            if valid_spike.any():
                vv = vs[valid_spike]
                vv[vv > 0.98] = add_jitter(vv[vv > 0.98])
                ax.plot(days[valid_spike], vv, 'o-', color='#e74c3c', ms=3, lw=1)
            
            vh = np.array(R['icc_healthy'])
            valid_healthy = ~np.isnan(vh)
            if valid_healthy.any():
                vv = vh[valid_healthy]
                vv[vv > 0.98] = add_jitter(vv[vv > 0.98])
                ax.plot(days[valid_healthy], vv, 's-', color='#3498db', ms=3, lw=1)
            
            ax.set(title=f'Participant {pid}: ICC (7-day test)', 
                   xlabel='Training Days', ylabel='ICC', ylim=(-0.05, 1.05))
        else:
            # AUC plot
            vm = np.array(R['auc_mean'])
            valid_auc = ~np.isnan(vm)
            if valid_auc.any():
                ax.errorbar(days[valid_auc], vm[valid_auc],
                            yerr=np.array(R['auc_std'])[valid_auc],
                            marker='o', ms=3, capsize=2, capthick=0.8,
                            color='#27ae60', lw=1)
                # Add target line
                ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
            
            ax.set(title=f'Participant {pid}: AUC (7-day test)', 
                   xlabel='Training Days', ylabel='AUC', ylim=(-0.05, 1.05))
        
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Draw all full rows
    end_full = (n_rows-1)*n_cols if plots_last != n_cols else n_rows*n_cols
    for idx in range(min(end_full, total)):
        r, c = divmod(idx, n_cols)
        ax = fig.add_subplot(outer[r, c])
        draw(idx, ax)
    
    # Handle last row if partial (center-aligned)
    if plots_last != n_cols:
        last_row_spec = outer[n_rows-1, :]
        inner = GridSpecFromSubplotSpec(
            1, plots_last+2,
            subplot_spec=last_row_spec,
            width_ratios=[blank] + [1]*plots_last + [blank],
            wspace=0.25
        )
        base_idx = (n_rows-1)*n_cols
        for i in range(plots_last):
            ax = fig.add_subplot(inner[0, i+1])
            idx = base_idx + i
            if idx < total:
                draw(idx, ax)
    
    # Add legend and title
    legend_elems = [
        Line2D([0], [0], marker='o', color='#e74c3c', label='BP Spike'),
        Line2D([0], [0], marker='s', color='#3498db', label='Healthy'),
        Line2D([0], [0], marker='o', color='#27ae60', label='AUC')
    ]
    fig.legend(handles=legend_elems, loc='upper right',
               bbox_to_anchor=(0.98, 0.99), fontsize=10,
               frameon=True, fancybox=True, shadow=True)
    
    plt.suptitle('Personalized Model: Temporal Analysis (7-day Fixed Test Set)', 
                 y=0.98, fontsize=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)
    gc.collect()
    print(f"Saved personalized fixed test grid ({n_cols}×{n_rows}) to {output_path}")

def main():
    """Main function with both temporal analyses"""
    # Define study durations for original analysis
    total_days_list = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
    
    # Load settings
    with open('participant_settings.json', 'r') as f:
        settings = json.load(f)
    
    # Output directories
    output_dir_original = os.path.join('results', 'temporal_analysis_retrained')
    output_dir_fixed = os.path.join('results', 'temporal_analysis_fixed_test')
    os.makedirs(output_dir_original, exist_ok=True)
    os.makedirs(output_dir_fixed, exist_ok=True)
    
    # Check flags
    import sys
    force_retrain = '--force-retrain' in sys.argv
    force_recompute = '--force-recompute' in sys.argv
    batch_processing = '--batch' in sys.argv
    run_fixed_test = '--fixed-test' in sys.argv or '--all' in sys.argv
    run_original = '--original' in sys.argv or '--all' in sys.argv or (not run_fixed_test)
    
    batch_size = 3
    
    print("\n" + "="*60)
    print("TEMPORAL ANALYSIS")
    if force_retrain:
        print("⚠️  FORCE RETRAIN MODE: All models will be retrained")
    else:
        print("✓ SMART MODE: Loading existing models when available")
    if batch_processing:
        print(f"✓ BATCH MODE: Processing {batch_size} participants at a time")
    
    analyses_to_run = []
    if run_original:
        analyses_to_run.append("Original (varying window size)")
    if run_fixed_test:
        analyses_to_run.append("Fixed test set (last 7 days)")
    print(f"✓ Analyses to run: {', '.join(analyses_to_run)}")
    print("="*60)

    pids = [s['pid'] for s in settings]
    
    # ========== ORIGINAL ANALYSIS (varying window size) ==========
    if run_original:
        print("\n" + "="*60)
        print("PART 1: ORIGINAL TEMPORAL ANALYSIS (varying window size)")
        print("="*60)
        
        results_path = os.path.join(output_dir_original, 'temporal_results_retrained.pkl')
        global_results, personalized_results, existing_days_list = load_existing_results(results_path)
        
        need_recompute = (force_recompute or 
                         global_results is None or 
                         personalized_results is None or
                         existing_days_list != total_days_list)
        
        if not need_recompute:
            print("✓ Using existing results from pickle file")
        else:
            print("Computing new results...")
            
            # Analyze global model
            global_results = analyze_global_model(total_days_list, force_retrain=force_retrain)
            aggressive_cleanup()
            
            # Analyze personalized models
            personalized_results = {}
            
            if batch_processing:
                for i in range(0, len(pids), batch_size):
                    batch_pids = pids[i:i+batch_size]
                    print(f"\nProcessing batch {i//batch_size + 1}: PIDs {batch_pids}")
                    
                    for pid in batch_pids:
                        results = analyze_personalized_model(pid, settings, total_days_list, force_retrain=force_retrain)
                        if results:
                            personalized_results[pid] = results
                        aggressive_cleanup()
                    
                    print(f"\nBatch {i//batch_size + 1} complete.")
                    aggressive_cleanup()
            else:
                for pid in pids:
                    results = analyze_personalized_model(pid, settings, total_days_list, force_retrain=force_retrain)
                    if results:
                        personalized_results[pid] = results
            
            # Save results
            with open(results_path, 'wb') as f:
                pickle.dump({
                    'global': global_results,
                    'personalized': personalized_results,
                    'total_days_list': total_days_list
                }, f)
            print(f"Saved results to {results_path}")
        
        # Generate plots
        print("\nGenerating plots for original analysis...")
        if global_results:
            plot_figure4_style(
                global_results,
                'Global Model: Temporal Analysis',
                os.path.join(output_dir_original, 'global_temporal_analysis_retrained.png')
            )
        
        if personalized_results:
            for pid, results in personalized_results.items():
                plot_figure4_style(
                    results,
                    f'Participant {pid}: Temporal Analysis',
                    os.path.join(output_dir_original, f'pid_{pid}_temporal_analysis_retrained.png')
                )
            
            plot_personalized_grid(
                personalized_results,
                pids,
                os.path.join(output_dir_original, 'all_personalized_temporal_analysis_retrained.png')
            )
    
    # ========== FIXED TEST ANALYSIS (last 7 days) ==========
    if run_fixed_test:
        print("\n" + "="*60)
        print("PART 2: FIXED TEST SET ANALYSIS (last 7 days)")
        print("="*60)
        
        # Calculate feasible training days dynamically
        test_days = 7
        global_train_days, participant_train_days = calculate_feasible_train_days(settings, test_days)
        
        if not global_train_days:
            print("ERROR: No participants have enough data for fixed test analysis")
            return
        
        print(f"Maximum training days available: {max(global_train_days)}")
        print(f"Global training days to test (odd only): {global_train_days}")
        print(f"Participants with sufficient data: {len([p for p in participant_train_days if participant_train_days[p]])}")
        
        results_path_fixed = os.path.join(output_dir_fixed, 'temporal_results_fixed_test.pkl')
        
        # Check for existing results
        if os.path.exists(results_path_fixed) and not force_recompute:
            print(f"Loading existing fixed test results from {results_path_fixed}")
            with open(results_path_fixed, 'rb') as f:
                data = pickle.load(f)
            global_results_fixed = data.get('global')
            personalized_results_fixed = data.get('personalized')
        else:
            print("Computing new fixed test results...")
            
            # Analyze global model with fixed test
            global_results_fixed = analyze_global_model_fixed_test(
                global_train_days, test_days=7, force_retrain=force_retrain
            )
            aggressive_cleanup()

            # Analyze personalized models with fixed test
            personalized_results_fixed = {}
            
            if batch_processing:
                for i in range(0, len(pids), batch_size):
                    batch_pids = pids[i:i+batch_size]
                    print(f"\nProcessing batch {i//batch_size + 1}: PIDs {batch_pids}")
                    
                    for pid in batch_pids:
                        # Use participant-specific feasible days
                        if pid in participant_train_days and participant_train_days[pid]:
                            results = analyze_personalized_model_fixed_test(
                                pid, settings, participant_train_days[pid], 
                                test_days=7, force_retrain=force_retrain
                            )
                            if results:
                                personalized_results_fixed[pid] = results
                        aggressive_cleanup()
                    
                    print(f"\nBatch {i//batch_size + 1} complete.")
                    aggressive_cleanup()
            else:
                for pid in pids:
                    # Use participant-specific feasible days
                    if pid in participant_train_days and participant_train_days[pid]:
                        results = analyze_personalized_model_fixed_test(
                            pid, settings, participant_train_days[pid], 
                            test_days=7, force_retrain=force_retrain
                        )
                        if results:
                            personalized_results_fixed[pid] = results
            
            # Save results
            with open(results_path_fixed, 'wb') as f:
                pickle.dump({
                    'global': global_results_fixed,
                    'personalized': personalized_results_fixed,
                    'train_days_list': global_train_days
                }, f)
            print(f"Saved results to {results_path_fixed}")
        
        # Generate plots for fixed test analysis
        print("\nGenerating plots for fixed test analysis...")
        if global_results_fixed:
            plot_fixed_test_analysis(
                global_results_fixed,
                'Global Model: Temporal Analysis (7-day Fixed Test Set)',
                os.path.join(output_dir_fixed, 'global_training_requirements.png')
            )
        
        if personalized_results_fixed:
            # Create individual plots
            for pid, results in personalized_results_fixed.items():
                if results:  # Check if results exist
                    plot_fixed_test_analysis(
                        results,
                        f'Participant {pid}: Training Data Requirements (7-day test)',
                        os.path.join(output_dir_fixed, f'pid_{pid}_training_requirements.png')
                    )

                    # Add the new grid plot for all personalized results
                    plot_personalized_fixed_grid(
                        personalized_results_fixed,
                        pids,
                        os.path.join(output_dir_fixed, 'all_personalized_training_requirements_grid.png')
                    )
            
            # Create a combined summary plot for all participants
            valid_pids = [pid for pid in personalized_results_fixed if personalized_results_fixed[pid]]
            if valid_pids:
                # Create a summary figure showing all participants' AUC curves
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for pid in valid_pids:
                    results = personalized_results_fixed[pid]
                    days = results['train_days']
                    auc_means = []
                    
                    for i, d in enumerate(days):
                        if isinstance(results['auc_mean'][i], (int, float)):
                            auc_means.append(results['auc_mean'][i])
                        else:
                            auc_means.append(np.nan)
                    
                    valid_auc = ~np.isnan(auc_means)
                    if np.any(valid_auc):
                        ax.plot(np.array(days)[valid_auc], np.array(auc_means)[valid_auc],
                               marker='o', markersize=4, linewidth=1.5, alpha=0.7, label=f'P{pid}')
                
                ax.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5, label='Target AUC')
                ax.set_xlabel('Training days (fixed 7-day test set)')
                ax.set_ylabel('AUC')
                ax.set_title('All Participants: Training Data Requirements')
                ax.set_ylim(0.4, 1.05)
                ax.grid(True, alpha=0.3) 
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir_fixed, 'all_participants_training_requirements.png'),
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved combined plot to {output_dir_fixed}/all_participants_training_requirements.png")
    
    print("\n✅ Temporal analysis completed!")
    if run_original:
        print(f"Original analysis results saved to {output_dir_original}")
    if run_fixed_test:
        print(f"Fixed test analysis results saved to {output_dir_fixed}")
    
    # Print usage instructions
    print("\n" + "="*60)
    print("Usage instructions:")
    print("  python temporal_analysis.py                # Run original analysis only")
    print("  python temporal_analysis.py --fixed-test   # Run fixed test analysis only") 
    print("  python temporal_analysis.py --all          # Run both analyses")
    print("  Add --force-retrain to retrain all models")
    print("  Add --force-recompute to recompute all results")
    print("  Add --batch for memory-efficient batch processing")
    print("="*60)

if __name__ == '__main__':
    main()