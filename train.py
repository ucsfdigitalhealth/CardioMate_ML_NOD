#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import keras_tuner as kt
import joblib  # for saving XGB model
import pickle  # for saving attention weights and latent vectors
from tensorflow.keras import Input
import re
import ast

#############################################
# Custom Attention Layers with weight extraction
#############################################
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self._attention_weights = None
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="normal",
                                 trainable=True)
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        self._attention_weights = a  # Store attention weights
        output = x * a
        return tf.keras.backend.sum(output, axis=1)
    
    def get_attention_weights(self):
        return self._attention_weights

class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.supports_masking = True
        self._attention_weights = None
        
    def call(self, inputs):
        attn_output, attn_weights = self.mha(query=inputs, key=inputs, value=inputs, return_attention_scores=True)
        # Average attention weights across heads
        self._attention_weights = tf.reduce_mean(attn_weights, axis=1)
        return tf.reduce_mean(attn_output, axis=1)
    
    def get_attention_weights(self):
        return self._attention_weights

class SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.att = tf.keras.layers.Attention()
        self.supports_masking = True
        self._attention_weights = None
        
    def call(self, x):
        att_out, att_weights = self.att([x, x], return_attention_scores=True)
        self._attention_weights = att_weights
        return tf.reduce_mean(att_out, axis=1)
    
    def get_attention_weights(self):
        return self._attention_weights

# Helper function for bootstrap CI
def mean_sd_ci(arr):
    arr = np.asarray(arr, dtype=float)
    m  = np.nanmean(arr)
    sd = np.nanstd(arr, ddof=1)
    lo, hi = np.nanpercentile(arr, [2.5, 97.5])
    return m, sd, lo, hi

def extract_attention_weights_and_latents(model, X, batch_size=32):
    """
    Returns (attn_out, lstm_out, context_vectors).
    Supports AttentionLayer, MultiHeadAttentionLayer, and SelfAttentionLayer
    as defined in the same script.
    """
    # 1) Build if needed
    if not model.built:
        model.build(input_shape=(None,) + X.shape[1:])

    # 2) Get model input tensor
    try:
        model_input = model.input
    except (AttributeError, ValueError):
        model_input = model.layers[0].input

    # 3) Locate attention layer and its preceding LSTM
    att_layer = None
    for idx, lyr in enumerate(model.layers):
        if isinstance(lyr, (AttentionLayer,
                            MultiHeadAttentionLayer,
                            SelfAttentionLayer)):
            att_layer = lyr
            lstm_idx = idx - 1
            break
    if att_layer is None:
        raise RuntimeError("No attention layer found in the model")

    # 4) Extract the LSTM output sequence
    lstm_layer = model.layers[lstm_idx]
    lstm_extractor = Model(inputs=model_input, outputs=lstm_layer.output)
    lstm_out = lstm_extractor.predict(X, batch_size=batch_size, verbose=0)

    # 5) Compute raw attention scores → attn_out of shape (batch_size, timesteps)
    if isinstance(att_layer, AttentionLayer):
        W = att_layer.W.numpy()             # (features, 1)
        b = att_layer.b.numpy()             # (timesteps, 1)
        b = b.reshape((1, b.shape[0], 1))
        e = np.tanh(lstm_out.dot(W) + b)    # (batch, timesteps, 1)
        if e.shape[-1] == 1:
            e = np.squeeze(e, axis=-1)      # (batch, timesteps)
        exp_e = np.exp(e - e.max(axis=1, keepdims=True))
        attn_out = exp_e / exp_e.sum(axis=1, keepdims=True)

    elif isinstance(att_layer, MultiHeadAttentionLayer):
        inp = tf.keras.Input(shape=lstm_out.shape[1:])
        _, raw_w = att_layer.mha(inp, inp, inp, return_attention_scores=True)
        extract_model = Model(inputs=inp, outputs=raw_w)
        attn_raw = extract_model.predict(lstm_out, batch_size=batch_size, verbose=0)
        attn_out = attn_raw.mean(axis=1).mean(axis=1)  # (batch, T)

    elif isinstance(att_layer, SelfAttentionLayer):
        att = tf.keras.layers.Attention()
        Q = tf.convert_to_tensor(lstm_out, dtype=tf.float32)
        V = tf.convert_to_tensor(lstm_out, dtype=tf.float32)
        _, raw_w = att([Q, V], return_attention_scores=True)
        attn_out = tf.reduce_mean(raw_w, axis=-1).numpy()

    else:
        raise RuntimeError(f"Unknown attention layer type: {type(att_layer)}")

    if attn_out.ndim > 2:
        extra_axes = tuple(range(2, attn_out.ndim))
        attn_out = attn_out.mean(axis=extra_axes)

    context_vectors = np.einsum('bt,btf->bf', attn_out, lstm_out)
    return attn_out, lstm_out, context_vectors


def extract_results_from_file(results_path):
    """Extract val_auc, hyperparameters, and tuner time from existing results.txt"""
    if not os.path.exists(results_path):
        return None, None, None
    
    with open(results_path, 'r') as f:
        content = f.read()
    
    tuner_time_match = re.search(r'Total tuner elapsed time: (.+)', content)
    tuner_time = tuner_time_match.group(1) if tuner_time_match else "N/A (model loaded)"
    
    val_auc_match = re.search(r'🔹 Best val_auc.*?: (.+)', content)
    if val_auc_match:
        val_auc_str = val_auc_match.group(1).strip()
        try:
            best_val = float(val_auc_str)
        except ValueError:
            best_val = val_auc_str
    else:
        best_val = "Loaded from file"
    
    hyperparam_match = re.search(r'🔹 Best LSTM hyperparameters: (.+)', content)
    if hyperparam_match:
        hyperparam_str = hyperparam_match.group(1).strip()
        if hyperparam_str.startswith('{'):
            try:
                hyperparams = ast.literal_eval(hyperparam_str)
            except:
                hyperparams = hyperparam_str
        else:
            hyperparams = hyperparam_str
    else:
        hyperparams = "Loaded from existing model"
    
    return best_val, hyperparams, tuner_time

def _train_xgb_pipeline(X_train, y_train, spw, args):
    """Train an XGB pipeline honoring args.no_resample; returns (pipeline, best_sampling_strategy)."""
    if args.no_resample:
        pipeline_steps = [
            ('scaler', StandardScaler()),
            ('xgb', xgb.XGBClassifier(scale_pos_weight=spw, random_state=42))
        ]
        param_grid = {
            'xgb__max_depth': [3,5,7],
            'xgb__learning_rate': [0.01,0.05,0.1],
            'xgb__n_estimators': [100,150,200]
        }
    else:
        pipeline_steps = [
            ('scaler', StandardScaler()),
            ('adasyn', ADASYN(n_neighbors=args.n_neighbors, random_state=42)),
            ('xgb', xgb.XGBClassifier(scale_pos_weight=spw, random_state=42))
        ]
        param_grid = {
            'adasyn__sampling_strategy': [0.6,0.65,0.7,0.75],
            'xgb__max_depth': [3,5,7],
            'xgb__learning_rate': [0.01,0.05,0.1],
            'xgb__n_estimators': [100,150,200]
        }
    pipeline = ImbPipeline(pipeline_steps)
    grid_search = GridSearchCV(
        pipeline, param_grid,
        scoring='roc_auc', cv=3, n_jobs=-1, verbose=int(args.verbose)
    )
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    best_sampling_strategy = grid_search.best_params_.get('adasyn__sampling_strategy', 0.7)
    return best_xgb, best_sampling_strategy


#############################################
# Main
#############################################
def main():
    parser = argparse.ArgumentParser(description="Train BP spike prediction models")
    parser.add_argument('--participant_id', required=True, help="Participant ID, e.g., 31")
    parser.add_argument('--models', default='xgb,attn', help="Comma-separated list of models: xgb, attn")
    parser.add_argument(
        '--drop',
        action='append',
        default=[],
        help="Features to drop. You can pass this multiple times or comma-separate: "
            "--drop a,b --drop c"
    )
    parser.add_argument('--train_days', type=int, default=20, help="Days used for train/test split")
    parser.add_argument('--batch', type=int, default=32, help="Batch size for LSTM training")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    parser.add_argument('--cpu', action='store_true', help="Force CPU execution")
    parser.add_argument('--n_neighbors', type=int, default=5, help="Number of neighbors for ADASYN (default 5)")
    parser.add_argument('--no_resample', action='store_true', help="Skip ADASYN resampling entirely")
    parser.add_argument('--out_dir', default=None, help="Directory to save outputs (models/results).")
    args = parser.parse_args()

    # ---- build final drop list from repeated --drop and comma-separated specs ----
    drop_list = []
    for spec in args.drop:  # args.drop is a list because of action='append'
        drop_list.extend([s.strip() for s in spec.split(',') if s.strip()])

    # de-dup while preserving order
    _seen = set()
    drops = [f for f in drop_list if not (f in _seen or _seen.add(f))]

    # decide ablation only after we know the real merged list
    is_ablation = len(drops) > 0

    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    start_time = time.time()

    # choose base output directory
    pid_str = f'pid_{args.participant_id}'
    if args.out_dir is not None:
        base = args.out_dir
    elif is_ablation:
        # fallback when caller didn't pass --out_dir
        base = os.path.join('results_rm_NO_BP', pid_str)
    else:
        base = os.path.join('results_NO_BP', 'personalized', pid_str)

    os.makedirs(base, exist_ok=True)
    print(f"🔧 Output dir: {base}")
    if drops:
        print(f"🔻 Final drop list (merged): {drops}")

    # Load processed CSV
    data_path = os.path.join('processed', f'hp{args.participant_id}', 'processed_bp_prediction_data.csv')
    df = pd.read_csv(data_path, parse_dates=['datetime_local'])
    df['pid'] = int(args.participant_id)

    # Define features & target
    features = [
        'hr_mean_5min','hr_min_5min','hr_max_5min','hr_std_5min',
        'steps_total_5min','steps_mean_5min','steps_min_5min','steps_max_5min','steps_std_5min','steps_diff_5min',
        'hr_mean_10min','hr_min_10min','hr_max_10min','hr_std_10min',
        'steps_total_10min','steps_mean_10min','steps_min_10min','steps_max_10min','steps_std_10min','steps_diff_10min',
        'hr_mean_30min','hr_min_30min','hr_max_30min','hr_std_30min',
        'steps_total_30min','steps_mean_30min','steps_min_30min','steps_max_30min','steps_std_30min','steps_diff_30min',
        'hr_mean_60min','hr_min_60min','hr_max_60min','hr_std_60min',
        'steps_total_60min','steps_mean_60min','steps_min_60min','steps_max_60min','steps_std_60min','steps_diff_60min',
        'stress_mean','stress_min','stress_max','stress_std',
        'hr_steps_ratio','stress_weighted_hr','stress_steps_ratio','steps_hr_variability_ratio',
        'hr_mean_rolling_3','steps_total_rolling_5','hr_std_rolling_3',
        'cumulative_stress_30min','cumulative_steps_30min',
        'hour_of_day','day_of_week','is_working_hours','is_weekend',
        'time_since_last_BP_spike'
    ]
    target = 'BP_spike'

    df = df[['datetime_local', 'pid'] + features + [target]]
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')

    # Split
    cutoff = df['datetime_local'].min() + pd.Timedelta(days=args.train_days)
    train_df = df[df['datetime_local'] < cutoff]
    test_df  = df[df['datetime_local'] >= cutoff]
    X_train, y_train = train_df[features], train_df[target]
    X_test,  y_test  = test_df[features],  test_df[target]

    # ── NEW: make explicit copies, drop safely, and sanitize inf/NaN before scaling
    X_train = X_train.copy()
    X_test  = X_test.copy()

    if drops:
        X_train.drop(columns=drops, errors="ignore", inplace=True)
        X_test.drop(columns=drops,  errors="ignore", inplace=True)

    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf],  np.nan, inplace=True)
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    # BP counts (for logging/results)
    print("🔹 BP Spike Counts Before Resampling:")
    print(f"   - Training Set: {sum(y_train)} spikes ({100*sum(y_train)/len(y_train):.2f}%)")
    print(f"   - Test Set: {sum(y_test)} spikes ({100*sum(y_test)/len(y_test):.2f}%)")
    
    train_spikes = int(y_train.sum())
    train_pct    = 100 * train_spikes / len(y_train)
    test_spikes  = int(y_test.sum())
    test_pct     = 100 * test_spikes / len(y_test)

    # Build XGBoost pipeline
    pos, neg = sum(y_train==1), sum(y_train==0)
    spw = neg/pos if pos>0 else 1

    xgb_model_path = os.path.join(base, 'xgb_model.joblib')

    def _loaded_pipeline_feature_names(pipeline):
        try:
            return list(pipeline.named_steps['scaler'].feature_names_in_)
        except Exception:
            return None

    need_train = True
    best_sampling_strategy = 0.7

    if os.path.exists(xgb_model_path):
        print("🔹 Loading existing XGBoost model...")
        try:
            loaded_xgb = joblib.load(xgb_model_path)
            loaded_feats  = _loaded_pipeline_feature_names(loaded_xgb)
            current_feats = list(X_train.columns)
            if loaded_feats is not None and loaded_feats == current_feats:
                best_xgb = loaded_xgb
                need_train = False
                print("🔹 XGBoost model loaded from file (feature layout matches).")
                if hasattr(best_xgb, 'named_steps') and 'adasyn' in best_xgb.named_steps:
                    best_sampling_strategy = best_xgb.named_steps['adasyn'].sampling_strategy
            else:
                print("⚠️  Cached XGB feature layout mismatch. Will retrain.")
        except Exception as e:
            print(f"⚠️  Failed to load cached XGB model ({e}). Will retrain.")

    if need_train:
        print("🔹 Training new XGBoost model...")
        best_xgb, best_sampling_strategy = _train_xgb_pipeline(X_train, y_train, spw, args)
        print("🔹 XGB grid search complete.")

        # ---------------- SHAP summary plot (XGBoost) ----------------
    try:
        # Get the fitted XGBClassifier out of the pipeline
        xgb_clf = best_xgb.named_steps['xgb']

        # Apply the same scaler transformation used in the pipeline
        if 'scaler' in best_xgb.named_steps:
            X_test_for_shap = best_xgb.named_steps['scaler'].transform(X_test)
        else:
            X_test_for_shap = X_test.values

        # Compute SHAP values
        explainer = shap.TreeExplainer(xgb_clf)
        shap_values = explainer.shap_values(X_test_for_shap)

        # ---- Bar Plot: mean(|SHAP|) ----
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            features=X_test_for_shap,
            feature_names=X_test.columns,
            plot_type='bar',
            show=False
        )
        plt.title(f"Participant {args.participant_id} — XGB SHAP (mean |SHAP|)")
        plt.tight_layout()
        shap_bar_path = os.path.join(base, 'shap_summary_xgb_bar.png')
        plt.savefig(shap_bar_path, dpi=200)
        plt.close()

        # ---- Beeswarm Plot ----
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            features=X_test_for_shap,
            feature_names=X_test.columns,
            show=False
        )
        plt.title(f"Participant {args.participant_id} — XGB SHAP (beeswarm)")
        plt.tight_layout()
        shap_bee_path = os.path.join(base, 'shap_summary_xgb_beeswarm.png')
        plt.savefig(shap_bee_path, dpi=200)
        plt.close()

        print(f"🔹 Saved SHAP figures:\n   - {shap_bar_path}\n   - {shap_bee_path}")
    except Exception as e:
        print(f"⚠️  SHAP plotting failed: {e}")


    # Prepare data for LSTM
    scaler_lstm = StandardScaler()
    X_tr_s = scaler_lstm.fit_transform(X_train)
    X_te_s = scaler_lstm.transform(X_test)

    if args.no_resample:
        X_res, y_res = X_tr_s, y_train.values
    else:
        ada_best = ADASYN(
            sampling_strategy=best_sampling_strategy,
            n_neighbors=args.n_neighbors,
            random_state=42
        )
        X_res, y_res = ada_best.fit_resample(X_tr_s, y_train)

    X_train_lstm = X_res.reshape((X_res.shape[0], X_res.shape[1], 1))
    X_test_lstm  = X_te_s.reshape((X_te_s.shape[0], X_te_s.shape[1], 1))

    cw = compute_class_weight('balanced', classes=np.unique(y_res), y=y_res)
    class_weight = {i: cw[i] for i in range(len(cw))}
    print("🔹 Class Weights for LSTM:", class_weight)

    # Initialize variables for hyperparameters
    best_hyperparams = None
    tuner = None

    # ── Revised LSTM load logic (no functional change when shapes match; skip on ablations)
    lstm_model_path = os.path.join(base, 'lstm_model.keras')
    model_loaded = False

    if (not is_ablation) and os.path.exists(lstm_model_path):
        print("🔹 Loading existing LSTM model...")
        try:
            best_lstm = tf.keras.models.load_model(
                lstm_model_path,
                compile=False,          # we don't need optimizer state to predict
                safe_mode=False,        # allow legacy args like 'batch_shape'
                custom_objects={
                    'AttentionLayer': AttentionLayer,
                    'MultiHeadAttentionLayer': MultiHeadAttentionLayer,
                    'SelfAttentionLayer': SelfAttentionLayer
                }
            )
            # Verify input timesteps (dimension 1) matches current feature width
            in_shape = best_lstm.input_shape
            if isinstance(in_shape, (list, tuple)):
                in_shape = in_shape[0]
            saved_T = in_shape[1]
            current_T = X_train_lstm.shape[1]
            if saved_T != current_T:
                print(f"⚠️  Saved LSTM expects T={saved_T}, current T={current_T}. Will retrain.")
                model_loaded = False
            else:
                print("🔹 LSTM model loaded from file")
                # Extract results from existing results.txt for reporting
                results_path = os.path.join(base, 'results.txt')
                best_val, best_hyperparams, tuner_time_str = extract_results_from_file(results_path)
                if best_val is None:
                    best_val = "Loaded from file"
                if tuner_time_str is None:
                    tuner_time_str = "N/A (model loaded)"
                if best_hyperparams is None:
                    best_hyperparams = "Loaded from existing model"
                model_loaded = True
        except Exception as e:
            print(f"⚠️  Model loading failed due to compatibility issue: {e}")
            print("🔹 Will retrain the model...")
            model_loaded = False

    if not model_loaded:
        print("🔹 Training new LSTM model...")
        def build_model(hp):
            model = Sequential()
            u1 = hp.Int('lstm_units',64,256,32)
            model.add(Bidirectional(LSTM(u1, return_sequences=True),
                                    input_shape=(X_train_lstm.shape[1],1)))
            model.add(BatchNormalization())
            dr = hp.Float('dropout_rate',0.2,0.5,0.1)
            model.add(Dropout(dr))
            u2 = hp.Int('lstm_units_2',32,128,16)
            model.add(LSTM(u2, return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dr))
            att = hp.Choice('attention_variant',['custom','multihead','selfattention'])
            if att=='custom':
                model.add(AttentionLayer())
            elif att=='multihead':
                nh = hp.Int('num_heads',1,4,1)
                kd = hp.Int('key_dim',16,64,16)
                model.add(MultiHeadAttentionLayer(nh, kd))
            else:
                model.add(SelfAttentionLayer())
            du = hp.Int('dense_units',16,64,16)
            drg = hp.Choice('dense_reg',[0.0,0.001,0.01,0.1])
            model.add(Dense(du, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(drg)))
            model.add(Dropout(dr))
            model.add(Dense(1, activation='sigmoid'))
            lr = hp.Choice('learning_rate',[0.001,0.0005,0.0001])
            model.compile(optimizer=Adam(learning_rate=lr),
                          loss='binary_crossentropy',
                          metrics=[tf.keras.metrics.AUC(name='auc')])
            return model

        tuner = kt.RandomSearch(
            build_model,
            kt.Objective('val_auc', direction='max'),
            max_trials=20,
            executions_per_trial=1,
            directory='lstm_tuner',
            project_name=f'bp_spike_{args.participant_id}',
            overwrite=True
        )

        t0 = time.time()
        tuner.search(
            X_train_lstm, y_res,
            epochs=50,
            batch_size=args.batch,
            validation_data=(X_test_lstm, y_test),
            class_weight=class_weight,
            verbose=1
        )
        t1 = time.time()
        telapsed = int(t1 - t0)
        h, r = divmod(telapsed, 3600)
        m, s = divmod(r, 60)
        tuner_time_str = f"{h:02d}h {m:02d}m {s:02d}s"
        print(f"Total tuner elapsed time: {tuner_time_str}")

        best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
        best_val = best_trial.metrics.get_best_value('val_auc')
        print(f"🔹 Best val_auc So Far: {best_val}")

        best_lstm = tuner.get_best_models(num_models=1)[0]
        best_hyperparams = tuner.get_best_hyperparameters(num_trials=1)[0].values
        print("🔹 Best LSTM hyperparameters:", best_hyperparams)

    # Extract attention weights, latent + context vectors
    print("🔹 Extracting attention weights and context vectors...")
    attn_w, latent_vectors, context_vectors = extract_attention_weights_and_latents(
        best_lstm, X_test_lstm, args.batch
    )

    # Save analysis data
    analysis_data = {
        'attention_weights': attn_w,
        'latent_vectors': latent_vectors,
        'context_vectors': context_vectors,
        'y_test': y_test.values,
        'pid': args.participant_id
    }
    with open(os.path.join(base, 'analysis_data.pkl'), 'wb') as f:
        pickle.dump(analysis_data, f)
    print(f"🔹 Saved attention weights shape: {attn_w.shape}")
    print(f"🔹 Saved latent vectors shape: {latent_vectors.shape}")
    print(f"🔹 Saved context vectors shape: {context_vectors.shape}")

    # Ensemble & threshold scan
    alphas     = np.linspace(0,1,11)
    thresholds = np.arange(0,1.01,0.01)
    best_thr    = None
    final_sens  = None
    final_spec  = None
    try:
        y_xgb = best_xgb.predict_proba(X_test)[:, 1]
    except ValueError as e:
        # Last-ditch recovery if columns drifted (e.g., old cache): retrain once and try again.
        print(f"⚠️  XGB predict_proba failed due to feature mismatch: {e}")
        print("🔁 Retraining XGB on current columns and retrying prediction...")
        best_xgb, best_sampling_strategy = _train_xgb_pipeline(X_train, y_train, spw, args)
        y_xgb = best_xgb.predict_proba(X_test)[:, 1]
    y_lstm = best_lstm.predict(X_test_lstm).flatten()

    unique = np.unique(y_test)
    if unique.size < 2:
        print("⚠️ Only one class in y_test; skipping ensemble and threshold/Youden scan.")
        best_a, best_auc = 0.0, float('nan')
        skip_threshold_scan = True
    else:
        skip_threshold_scan = False
        alphas = np.linspace(0,1,11)
        best_auc, best_a = -1.0, 0.0

        print("\n🔹 Ensemble Weight Grid Search:")
        for a in alphas:
            ens = a * y_xgb + (1 - a) * y_lstm
            auc_v = roc_auc_score(y_test, ens)
            if np.isnan(auc_v):
                continue
            print(f"  Alpha: {a:.2f}, Beta: {1-a:.2f}, AUROC: {auc_v:.3f}")
            if auc_v > best_auc:
                best_auc, best_a = auc_v, a

        if best_auc < 0.0:
            print("⚠️ No valid ensemble AUC found; defaulting to LSTM alone.")
            best_a, best_auc = 0.0, roc_auc_score(y_test, y_lstm)

    print(f"\n🔹 Best Ensemble Weights: Alpha = {best_a:.2f}, Beta = {1-best_a:.2f}, AUROC = {best_auc:.3f}")

    # Youden threshold scan + plotting
    if not skip_threshold_scan:
        thresholds = np.arange(0,1.01,0.01)
        best_youden, best_thr = -1, None

        print("\n🔹 Youden Threshold Scan:")
        for t in thresholds:
            yb = ((best_a*y_xgb + (1-best_a)*y_lstm) >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, yb, labels=[0, 1]).ravel()
            sens = tp/(tp+fn) if (tp+fn)>0 else 0
            spec = tn/(tn+fp) if (tn+fp)>0 else 0
            youden = sens + spec - 1
            print(f"  Threshold: {t:.2f} | Sens: {sens:.2f} | Spec: {spec:.2f} | Youden: {youden:.2f}")
            if youden > best_youden:
                best_youden, best_thr, final_sens, final_spec = youden, t, sens, spec

        print(f"\n🔹 Best Threshold: {best_thr:.2f} (Sens: {final_sens:.3f}, Spec: {final_spec:.3f})")

        sens_list, spec_list = [], []
        for t in thresholds:
            yb = ((best_a*y_xgb + (1-best_a)*y_lstm) >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, yb, labels=[0,1]).ravel()
            sens_list.append(tp/(tp+fn) if (tp+fn)>0 else 0)
            spec_list.append(tn/(tn+fp) if (tn+fp)>0 else 0)

        plt.figure(figsize=(8,5))
        plt.plot(thresholds, sens_list, marker='o', label='Sensitivity')
        plt.plot(thresholds, spec_list, marker='s', label='Specificity')
        plt.xlabel('Decision Threshold')
        plt.ylabel('Value')
        plt.title(f'Participant {args.participant_id} ‒ Sens/Spec (AUROC: {best_auc:.3f})')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(base, 'sens_spec_plot.png'))
        plt.close()
    else:
        print("⚠️ Skipped threshold scan – no plot generated.")

    # Total training time
    end_time = time.time()
    total_elapsed = int(end_time - start_time)
    th, rem = divmod(total_elapsed,3600)
    tm, ts = divmod(rem,60)
    total_time_str = f"{th:02d}h {tm:02d}m {ts:02d}s"
    print(f"Total training time: {total_time_str}")

    # ─────────────────────────────────────────────────────────
    #  Bootstrap uncertainty estimates (FULL VERSION FROM ORIGINAL)
    # ─────────────────────────────────────────────────────────
    B = 1000
    rng = np.random.default_rng(42)
    scores = best_a * y_xgb + (1 - best_a) * y_lstm     # ensemble probs

    thr_grid  = np.arange(0.00, 1.01, 0.01)
    T         = len(thr_grid)

    sens_boot = np.empty((B, T))
    spec_boot = np.empty((B, T))
    auc_boot  = np.empty(B)

    for b in range(B):
        idx  = rng.choice(len(y_test), size=len(y_test), replace=True)
        yt   = y_test.iloc[idx].values
        sc   = scores[idx]

        auc_boot[b] = roc_auc_score(yt, sc)

        for t_i, thr in enumerate(thr_grid):
            yb = (sc >= thr)
            tp = (yt &  yb).sum()
            fn = (yt & ~yb).sum()
            tn = (~yt & ~yb).sum()
            fp = (~yt &  yb).sum()

            sens_boot[b, t_i] = tp / (tp+fn) if tp+fn else np.nan
            spec_boot[b, t_i] = tn / (tn+fp) if tn+fp else np.nan

    def summary(arr):
        m  = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, ddof=1, axis=0)
        lo, hi = np.nanpercentile(arr, [2.5, 97.5], axis=0)
        return m, sd, lo, hi

    sens_m, sens_sd, sens_lo, sens_hi = summary(sens_boot)
    spec_m, spec_sd, spec_lo, spec_hi = summary(spec_boot)
    auc_m , auc_sd , auc_lo , auc_hi  = mean_sd_ci(auc_boot)

    print(f"\nAUROC (row-bootstrap) {auc_m:.3f} ±{auc_sd:.3f} "
          f"(95% CI {auc_lo:.3f}–{auc_hi:.3f})")

    if best_thr is not None:
        best_i = np.where(thr_grid == best_thr)[0][0]
        print(f"[Youden thr {best_thr:.2f}]  "
              f"Sens {sens_m[best_i]:.2f}±{sens_sd[best_i]:.2f} | "
              f"Spec {spec_m[best_i]:.2f}±{spec_sd[best_i]:.2f}")
    else:
        print("⚠️ Skipping Youden‐threshold bootstrap summary—no valid threshold was chosen.")

    pd.DataFrame({
        'threshold': thr_grid,
        'sens_mean': sens_m,  'sens_sd': sens_sd,
        'sens_lo'  : sens_lo, 'sens_hi': sens_hi,
        'spec_mean': spec_m,  'spec_sd': spec_sd,
        'spec_lo'  : spec_lo, 'spec_hi': spec_hi
    }).to_csv(os.path.join(base, 'bootstrap_threshold_stats.csv'), index=False)
    print("Bootstrapped CIs for all thresholds → bootstrap_threshold_stats.csv")

    # ───────── Save artifacts into *this run's* base dir ─────────
    # XGB: always persist for this run (baseline or ablation)
    joblib.dump(best_xgb, os.path.join(base, 'xgb_model.joblib'))

    # LSTM: only persist the shared baseline model
    if not is_ablation:
        best_lstm.save(os.path.join(base, 'lstm_model.keras'))

    # AUROC pickle (per-run)
    with open(os.path.join(base, 'auroc_data.pkl'), 'wb') as f:
        pickle.dump({'pid': args.participant_id, 'personalized_auroc': best_auc}, f)

    # Write this run's results.txt
    results_path = os.path.join(base, 'results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Total elapsed time: {total_time_str}\n")
        f.write(f"Total tuner elapsed time: {tuner_time_str}\n")
        f.write(f"🔹 Best val_auc So Far: {best_val}\n")
        f.write(f"🔹 Best LSTM hyperparameters: {best_hyperparams}\n\n")
        f.write("🔹 Ensemble Weight Grid Search:\n")
        for a in alphas:
            auc_v = roc_auc_score(y_test, a*y_xgb + (1-a)*y_lstm)
            f.write(f"Alpha: {a:.2f}, Beta: {1-a:.2f}, AUROC: {auc_v:.3f}\n")
        f.write("\n")
        if best_thr is not None:
            f.write(f"🔹 Best Threshold in [0,1]: {best_thr:.2f}\n")
            f.write(f"🔹 Final Sensitivity (Recall): {final_sens:.3f}\n")
            f.write(f"🔹 Final Specificity: {final_spec:.3f}\n")
        else:
            f.write("🔹 Best Threshold in [0,1]: None (no valid threshold found)\n")
        f.write(f"Total training time: {total_time_str}\n")
        f.write(
            f"\nAUROC (row-bootstrap) {auc_m:.3f} ±{auc_sd:.3f}"
            f" (95% CI {auc_lo:.3f}–{auc_hi:.3f})\n"
        )
        if best_thr is not None:
            best_i = np.where(thr_grid == best_thr)[0][0]
            f.write(
                f"[Youden thr {best_thr:.2f}]  "
                f"Sens {sens_m[best_i]:.2f}±{sens_sd[best_i]:.2f} | "
                f"Spec {spec_m[best_i]:.2f}±{spec_sd[best_i]:.2f}\n"
            )
        f.write("\n🔹 BP Spike Counts Before Resampling:\n")
        f.write(f"   - Training Set: {train_spikes} spikes ({train_pct:.2f}%)\n")
        f.write(f"   - Test Set:     {test_spikes} spikes ({test_pct:.2f}%)\n")

    # Append to global baseline summary *only* for baseline runs
    if not is_ablation:
        concat_path = os.path.join('results_NO_BP', 'personalized', 'all_participants_results.txt')
        os.makedirs(os.path.dirname(concat_path), exist_ok=True)
        with open(concat_path, 'a') as f:
            if best_thr is not None:
                f.write(f"PID {args.participant_id}: AUROC {best_auc:.3f}, Thr {best_thr:.2f}\n")
            else:
                f.write(f"PID {args.participant_id}: AUROC {best_auc:.3f}, Thr None\n")


    print(f"✅ Personalized results saved to {base}")
    print(f"\nTotal training time: {th:02d}h {tm:02d}m {ts:02d}s")

if __name__ == '__main__':
    main()
