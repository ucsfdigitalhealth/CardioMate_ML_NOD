#!/usr/bin/env python3
"""
global_train.py
===============

Train a *global* ensemble (XGBoost + BiLSTM-Attention) for stress-induced
blood-pressure-spike prediction across multiple participants, preserving each
participant’s original train/test split.
"""

import pickle
import argparse
import json
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
from tensorflow.keras import Input

def mean_sd_ci(arr):
    arr = np.asarray(arr, dtype=float)
    m  = np.nanmean(arr)
    sd = np.nanstd(arr, ddof=1)
    lo, hi = np.nanpercentile(arr, [2.5, 97.5])
    return m, sd, lo, hi

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
    'hr_steps_ratio','stress_weighted_hr','stress_steps_ratio','steps_hr_variability_ratio',
    'hr_mean_rolling_3','steps_total_rolling_5','hr_std_rolling_3',
    'cumulative_stress_30min','cumulative_steps_30min',
    'hour_of_day','day_of_week','is_working_hours','is_weekend',
    'time_since_last_BP_spike'
]

#############################################
# Custom Attention Layers
#############################################
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self._attention_weights = None

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        self._attention_weights = a
        return tf.keras.backend.sum(x * a, axis=1)

    def get_attention_weights(self):
        return self._attention_weights

class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.supports_masking = True
        self._attention_weights = None

    def call(self, x):
        attn_output, attn_weights = self.mha(query=x, key=x, value=x, return_attention_scores=True)
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

#############################################
# Extraction utility
#############################################
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Lambda

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

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
    # shape: (batch_size, timesteps, features)

    # 5) Compute raw attention scores → attn_out of shape (batch_size, timesteps)
    if isinstance(att_layer, AttentionLayer):
        # custom single-head
        W = att_layer.W.numpy()             # (features, 1)
        b = att_layer.b.numpy()             # (timesteps, 1)
        # reshape b to (1, timesteps, 1) for proper broadcasting
        b = b.reshape((1, b.shape[0], 1))
        e = np.tanh(lstm_out.dot(W) + b)    # (batch, timesteps, 1)
        # only squeeze if last dim is size 1
        if e.shape[-1] == 1:
            e = np.squeeze(e, axis=-1)      # (batch, timesteps)
        exp_e = np.exp(e - e.max(axis=1, keepdims=True))
        attn_out = exp_e / exp_e.sum(axis=1, keepdims=True)

    elif isinstance(att_layer, MultiHeadAttentionLayer):
        # multi-head
        inp = tf.keras.Input(shape=lstm_out.shape[1:])
        _, raw_w = att_layer.mha(inp, inp, inp, return_attention_scores=True)
        extract_model = Model(inputs=inp, outputs=raw_w)
        attn_raw = extract_model.predict(lstm_out, batch_size=batch_size, verbose=0)
        # (batch, heads, T, T) → average heads then rows → (batch, T)
        attn_out = attn_raw.mean(axis=1).mean(axis=1)

    elif isinstance(att_layer, SelfAttentionLayer):
        # built‑in tf Attention
        att = tf.keras.layers.Attention()
        Q = tf.convert_to_tensor(lstm_out, dtype=tf.float32)
        V = tf.convert_to_tensor(lstm_out, dtype=tf.float32)
        _, raw_w = att([Q, V], return_attention_scores=True)  # (batch, T, T)
        attn_out = tf.reduce_mean(raw_w, axis=-1).numpy()     # (batch, T)

    else:
        raise RuntimeError(f"Unknown attention layer type: {type(att_layer)}")

    # 6) Collapse any extra dims just in case
    if attn_out.ndim > 2:
        extra_axes = tuple(range(2, attn_out.ndim))
        attn_out = attn_out.mean(axis=extra_axes)  # → (batch, timesteps)

    # 7) Compute context vectors via weighted sum over time
    #    result shape: (batch, features)
    context_vectors = np.einsum('bt,btf->bf', attn_out, lstm_out)

    return attn_out, lstm_out, context_vectors

#############################################
# Argument parsing
#############################################
def parse_args():
    p = argparse.ArgumentParser(description="Global BP-spike trainer")
    p.add_argument("--settings", required=True, help="JSON settings file")
    p.add_argument("--batch",       type=int, default=32)
    p.add_argument("--lstm_trials", type=int, default=20)
    p.add_argument("--no_resample", action="store_true")
    p.add_argument("--n_neighbors", type=int, default=5)
    p.add_argument("--cpu",         action="store_true")
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()

#############################################
# Data loading & prep
#############################################
def load_participant_df(pid):
    path = os.path.join("processed", f"hp{pid}", "processed_bp_prediction_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["datetime_local"])
    df["pid"] = pid
    return df

def build_feature_intersection(settings):
    inter = set(FEATURES)
    for s in settings:
        df = load_participant_df(s["pid"])
        inter &= {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    return sorted(inter)

def build_datasets(settings, feats):
    gtrain, gtest, per_pid_test = [], [], {}
    for s in settings:
        pid, days = s["pid"], s["train_days"]
        df = load_participant_df(pid)
        drops = set(s.get("drop", []))
        valid = [f for f in feats if f not in drops]
        cutoff = df["datetime_local"].min() + pd.Timedelta(days=days)
        tr, te = df[df["datetime_local"]<cutoff], df[df["datetime_local"]>=cutoff]
        tr[valid] = tr[valid].apply(pd.to_numeric, errors="coerce")
        te[valid] = te[valid].apply(pd.to_numeric, errors="coerce")
        gtrain.append(tr), gtest.append(te)
        per_pid_test[pid] = te
    gtr = pd.concat(gtrain); gte = pd.concat(gtest)
    gtr[feats] = gtr[feats].apply(pd.to_numeric, errors="coerce")
    gte[feats] = gte[feats].apply(pd.to_numeric, errors="coerce")
    return gtr[feats], gtr["BP_spike"], gte[feats], gte["BP_spike"], per_pid_test

#############################################
# XGBoost grid helper
#############################################
def xgb_grid(spw, resample, neighbors, verbose):
    if resample:
        steps = [
            ("scaler", StandardScaler()),
            ("adasyn", ADASYN(n_neighbors=neighbors, random_state=42)),
            ("xgb", xgb.XGBClassifier(scale_pos_weight=spw, random_state=42, eval_metric="logloss")),
        ]
        grid = {
            "adasyn__sampling_strategy": [0.6,0.7,0.75],
            "xgb__max_depth": [3,5,7],
            "xgb__learning_rate":[0.01,0.05,0.1],
            "xgb__n_estimators":[100,150,200],
        }
    else:
        steps = [
            ("scaler", StandardScaler()),
            ("xgb", xgb.XGBClassifier(scale_pos_weight=spw, random_state=42, eval_metric="logloss")),
        ]
        grid = {
            "xgb__max_depth":[3,5,7],
            "xgb__learning_rate":[0.01,0.05,0.1],
            "xgb__n_estimators":[100,150,200],
        }
    return GridSearchCV(ImbPipeline(steps), grid, scoring="roc_auc", cv=3, n_jobs=-1, verbose=int(verbose))

#############################################
# Keras‑Tuner builder
#############################################
def build_lstm_tuner(Xtr, ytr, Xval, yval, trials, batch):
    def mb(hp):
        m = Sequential()
        m.add(Bidirectional(LSTM(hp.Int("u1",64,256,32), return_sequences=True),
                            input_shape=(Xtr.shape[1],1)))
        m.add(BatchNormalization())
        dr = hp.Float("drop", 0.2,0.5,0.1)
        m.add(Dropout(dr))
        m.add(LSTM(hp.Int("u2",32,128,16), return_sequences=True))
        m.add(BatchNormalization()); m.add(Dropout(dr))
        att = hp.Choice("att",["custom","multi","self"])
        if att=="custom":
            m.add(AttentionLayer())
        elif att=="multi":
            m.add(MultiHeadAttentionLayer(hp.Int("heads",1,4,1), hp.Int("kdim",16,64,16)))
        else:
            m.add(SelfAttentionLayer())
        m.add(Dense(hp.Int("dense",16,64,16),
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(hp.Choice("reg",[0.0,0.001,0.01,0.1]))))
        m.add(Dropout(dr)); m.add(Dense(1,activation="sigmoid"))
        m.compile(optimizer=Adam(hp.Choice("lr",[1e-3,5e-4,1e-4])),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
        return m

    tuner = kt.RandomSearch(mb,
                            objective=kt.Objective("val_auc",direction="max"),
                            max_trials=trials,
                            directory="lstm_tuner",
                            project_name="global",
                            overwrite=True)
    tuner.search(Xtr, ytr,
                 validation_data=(Xval, yval),
                 epochs=50,
                 batch_size=batch,
                 verbose=1)
    return tuner

#############################################
# Ensemble weight search
#############################################
def ensemble_alpha(y_true, y_xgb, y_lstm):
    best_a, best_auc = None, -1
    for a in np.linspace(0,1,11):
        auc = roc_auc_score(y_true, a*y_xgb + (1-a)*y_lstm)
        if auc > best_auc:
            best_a, best_auc = a, auc
    return best_a, best_auc

#############################################
# Main
#############################################
def main():
    args = parse_args()
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    t0 = time.time()
    
    # Initialize tracking variables
    best_xgb_params = None
    best_lstm_val_auc = None

    # Prepare output directory and model paths
    base = os.path.join("results_NO_BP", "global_noBP_Lags")
    os.makedirs(base, exist_ok=True)
    xgb_path  = os.path.join(base, "xgb_global.joblib")
    lstm_path = os.path.join(base, "lstm_global_model.keras")

    # Load settings & build data
    settings = json.load(open(args.settings))
    feats = build_feature_intersection(settings)
    print(f"🔹 {len(feats)} common numeric features retained.")
    Xtr, ytr, Xte, yte, te_by_pid = build_datasets(settings, feats)
    print(f"🔹 Global-train: {Xtr.shape}, Global-test: {Xte.shape}")

    # Fill missing with medians
    med = Xtr.median()
    Xtr = Xtr.fillna(med)
    Xte = Xte.fillna(med)
    for pid, df in te_by_pid.items():
        te_by_pid[pid] = df.assign(**{c: df[c].fillna(med[c]) for c in med.index})

    # ——— Train or load XGBoost ———
    def _loaded_feats(pipeline):
        try:
            return list(pipeline.named_steps["scaler"].feature_names_in_)
        except Exception:
            return None

    need_xgb_train = True
    if os.path.exists(xgb_path):
        print("🔹 Loading existing XGB model…")
        try:
            xgb_best = joblib.load(xgb_path)
            lf = _loaded_feats(xgb_best)
            if lf is None or lf != list(Xtr.columns):
                print("⚠️  Cached XGB feature layout mismatch. Will retrain.")
            else:
                best_xgb_params = "Loaded from file"
                need_xgb_train = False
        except Exception as e:
            print(f"⚠️  Failed to load cached XGB model ({e}). Will retrain.")

    if need_xgb_train:
        print("🔹 Training XGB model…")
        spw = (len(ytr) - ytr.sum()) / ytr.sum() if ytr.sum() else 1
        gs = xgb_grid(spw, not args.no_resample, args.n_neighbors, args.verbose)
        gs.fit(Xtr, ytr)
        xgb_best = gs.best_estimator_
        best_xgb_params = gs.best_params_
        print("🔹 Best XGB params:", best_xgb_params)
        joblib.dump(xgb_best, xgb_path)


    # ——— Prepare data for LSTM ———
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    if args.no_resample:
        Xres, yres = Xtr_s, ytr.values
    else:
        ada = ADASYN(
            n_neighbors=args.n_neighbors,
            random_state=42,
            sampling_strategy=(xgb_best.named_steps["adasyn"].sampling_strategy
                               if hasattr(xgb_best, "named_steps") and "adasyn" in xgb_best.named_steps
                               else 0.7)
        )
        Xres, yres = ada.fit_resample(Xtr_s, ytr)
    Xres3 = Xres.reshape((Xres.shape[0], Xres.shape[1], 1))
    Xte3  = Xte_s.reshape((Xte_s.shape[0], Xte_s.shape[1], 1))

    # ——— Train or load LSTM ———
    lstm_best = None
    if os.path.exists(lstm_path):
        print("🔹 Loading existing LSTM model…")
        try:
            lstm_best = tf.keras.models.load_model(
                lstm_path,
                compile=False,              # don't restore optimizer state
                safe_mode=False,            # ignore legacy args like 'batch_shape'
                custom_objects={
                    "AttentionLayer": AttentionLayer,
                    "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
                    "SelfAttentionLayer": SelfAttentionLayer
                }
            )
            # sanity check: saved model timesteps must match current feature width
            in_shape = lstm_best.input_shape
            if isinstance(in_shape, (list, tuple)):  # handle multi-input case
                in_shape = in_shape[0]
            saved_T = in_shape[1]
            current_T = Xres3.shape[1]  # == len(feats)
            if saved_T != current_T:
                print(f"⚠️  Saved LSTM expects T={saved_T}, current T={current_T}. Will retrain.")
                lstm_best = None
            else:
                best_lstm_val_auc = "Loaded from file"
        except Exception as e:
            print(f"⚠️  LSTM load failed: {e}\n🔁 Retraining with tuner…")
            lstm_best = None

    if lstm_best is None:
        print("🔹 Tuning & training LSTM model…")
        tuner = build_lstm_tuner(Xres3, yres, Xte3, yte,
                                trials=args.lstm_trials, batch=args.batch)
        lstm_best = tuner.get_best_models(1)[0]
        best_lstm_val_auc = tuner.oracle.get_best_trials(1)[0].metrics.get_best_value("val_auc")
        print(f"🔹 Best LSTM val_auc: {best_lstm_val_auc:.3f}")
        lstm_best.save(lstm_path)

    # ——— Extract global attention, latent, and context vectors ———
    print("🔹 Extracting global attention weights and context vectors…")
    global_att, global_lat, global_ctx = extract_attention_weights_and_latents(
        lstm_best, Xte3, args.batch
    )

    # ——— Build per-PID predictions & extractions ———
    per = {}
    for pid, df in te_by_pid.items():
        Xp = df[feats]
        yp = df["BP_spike"].values
        Xp_s = scaler.transform(Xp).reshape((len(Xp), len(feats), 1))

        yx = xgb_best.predict_proba(Xp)[:,1]
        yl = lstm_best.predict(Xp_s).ravel()
        att, lat, ctx = extract_attention_weights_and_latents(
            lstm_best, Xp_s, args.batch
        )

        per[pid] = {
            "attention_weights": att,
            "latent_vectors": lat,
            "context_vectors": ctx,
            "y_test": yp,
            "xgb_probs": yx,
            "lstm_probs": yl
        }

    # ——— Save analysis data ———
    with open(os.path.join(base, "global_analysis_data.pkl"), "wb") as f:
        pickle.dump({
            "global": {
                "attention_weights": global_att,
                "latent_vectors": global_lat,
                "context_vectors": global_ctx,
                "y_test": yte.values
            },
            "per_pid": per
        }, f)

    # ——— Ensemble evaluation ———
    y_xgb  = xgb_best.predict_proba(Xte)[:,1]
    y_lstm = lstm_best.predict(Xte3).ravel()
    alpha, glob_auc = ensemble_alpha(yte, y_xgb, y_lstm)
    print(f"🔹 Best α = {alpha:.2f} | Global AUROC = {glob_auc:.3f}")

    # ——— Bootstrap CIs ———
    B   = 1000
    rng = np.random.default_rng(42)
    scores = alpha*y_xgb + (1-alpha)*y_lstm

    # row-bootstrap
    row_auc = []
    for _ in range(B):
        idx = rng.choice(len(yte), replace=True, size=len(yte))
        row_auc.append(roc_auc_score(yte.iloc[idx], scores[idx]))
    auc_m, auc_sd, auc_lo, auc_hi = mean_sd_ci(row_auc)
    print(f"\nAUROC (row-bootstrap) {auc_m:.3f} ±{auc_sd:.3f} (95% CI {auc_lo:.3f}–{auc_hi:.3f})")

    # cluster-bootstrap
    pid_rows = {pid: te_by_pid[pid].index.values for pid in te_by_pid}
    clust_auc = []
    for _ in range(B):
        samp = rng.choice(list(pid_rows), replace=True, size=len(pid_rows))
        rows = np.concatenate([pid_rows[p] for p in samp])
        clust_auc.append(roc_auc_score(yte.iloc[rows], scores[rows]))
    cl_m, cl_sd, cl_lo, cl_hi = mean_sd_ci(clust_auc)
    print(f"AUROC (cluster-bootstrap) {cl_m:.3f} ±{cl_sd:.3f} (95% CI {cl_lo:.3f}–{cl_hi:.3f})\n")

    # ——— Threshold scan ———
    best_thr = None
    best_y   = -1
    final_s, final_sp = None, None
    for t in np.arange(0,1.01,0.01):
        pred = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        sens = tp/(tp+fn) if tp+fn else 0
        spec = tn/(tn+fp) if tn+fp else 0
        yv = sens + spec - 1
        print(f"Threshold {t:.2f} | Sens {sens:.2f} | Spec {spec:.2f} | Youden {yv:.2f}")
        if yv > best_y:
            best_y, best_thr, final_s, final_sp = yv, t, sens, spec
    print(f"\n🔹 Optimal thr {best_thr:.2f} | Sens {final_s:.2f} | Spec {final_sp:.2f}")

  # ——— Ensemble evaluation ———
    y_xgb  = xgb_best.predict_proba(Xte)[:,1]
    y_lstm = lstm_best.predict(Xte3).ravel()
    alpha, glob_auc = ensemble_alpha(yte, y_xgb, y_lstm)
    print(f"🔹 Best α = {alpha:.2f} | Global AUROC = {glob_auc:.3f}")

    # ——— Bootstrap CIs ———
    B   = 1000
    rng = np.random.default_rng(42)
    scores = alpha * y_xgb + (1 - alpha) * y_lstm

    # row‐bootstrap
    row_auc = []
    for _ in range(B):
        idx = rng.choice(len(yte), size=len(yte), replace=True)
        row_auc.append(roc_auc_score(yte.iloc[idx], scores[idx]))
    auc_m, auc_sd, auc_lo, auc_hi = mean_sd_ci(row_auc)
    print(f"\nAUROC (row-bootstrap) {auc_m:.3f} ±{auc_sd:.3f} (95% CI {auc_lo:.3f}–{auc_hi:.3f})")

    # cluster‐bootstrap (by participant)
    pid_rows = {pid: te_by_pid[pid].index.values for pid in te_by_pid}
    clust_auc = []
    for _ in range(B):
        samp = rng.choice(list(pid_rows), size=len(pid_rows), replace=True)
        rows = np.concatenate([pid_rows[p] for p in samp])
        clust_auc.append(roc_auc_score(yte.iloc[rows], scores[rows]))
    cl_m, cl_sd, cl_lo, cl_hi = mean_sd_ci(clust_auc)
    print(f"AUROC (cluster-bootstrap) {cl_m:.3f} ±{cl_sd:.3f} (95% CI {cl_lo:.3f}–{cl_hi:.3f})\n")

    # ——— Threshold scanning ———
    best_thr = None
    best_y   = -1
    final_s, final_sp = None, None
    for t in np.arange(0, 1.01, 0.01):
        pred = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        youden = sens + spec - 1
        print(f"Threshold {t:.2f} | Sens {sens:.2f} | Spec {spec:.2f} | Youden {youden:.2f}")
        if youden > best_y:
            best_y, best_thr, final_s, final_sp = youden, t, sens, spec
    print(f"\n🔹 Optimal threshold: {best_thr:.2f} | Sens {final_s:.2f} | Spec {final_sp:.2f}")

    # ——— Per-participant AUROC (row-bootstrap) ———
    print("\nPer-participant AUROC (row-bootstrap B=1000):")
    for pid, df in te_by_pid.items():
        Xp = df[feats]
        yp = df["BP_spike"].values
        yx = xgb_best.predict_proba(Xp)[:,1]
        yl = lstm_best.predict(scaler.transform(Xp).reshape((len(Xp), len(feats), 1))).ravel()
        pr = alpha * yx + (1 - alpha) * yl

        boot = []
        for _ in range(B):
            idx = rng.choice(len(yp), size=len(yp), replace=True)
            try:
                boot.append(roc_auc_score(yp[idx], pr[idx]))
            except ValueError:
                pass
        if boot:
            m = np.mean(boot)
            sd = np.std(boot, ddof=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(f"PID {pid:02d}: {m:.3f} ±{sd:.3f} (95% CI {lo:.3f}–{hi:.3f})")
        else:
            print(f"PID {pid:02d}: NA   (too few positives)")

    # ——— Save per‐PID global AUROCs for scatter plot ———
    glob_aurocs = {}
    for pid, df in te_by_pid.items():
        Xp = df[feats]
        yp = df["BP_spike"].values
        yx = xgb_best.predict_proba(Xp)[:,1]
        yl = lstm_best.predict(scaler.transform(Xp).reshape((len(Xp), len(feats), 1))).ravel()
        pr = alpha * yx + (1 - alpha) * yl
        try:
            glob_aurocs[pid] = roc_auc_score(yp, pr)
        except ValueError:
            glob_aurocs[pid] = np.nan

    with open(os.path.join(base, "global_aurocs.pkl"), "wb") as f:
        pickle.dump(glob_aurocs, f)

    # ——— SHAP summary plot ———
    expl = shap.Explainer(xgb_best.named_steps["xgb"])
    idx = np.random.choice(len(Xte), size=min(len(Xte), 1000), replace=False)
    shap_vals = expl(xgb_best.named_steps["scaler"].transform(Xte.iloc[idx]))
    shap.summary_plot(shap_vals, Xte.iloc[idx], feature_names=feats, show=False)
    plt.savefig(os.path.join(base, "shap_summary.png"))
    plt.close()

    # ——— Sensitivity–Specificity tradeoff plot ———
    sens_list = []
    spec_list = []
    thr_grid = np.arange(0, 1.01, 0.01)
    for t in thr_grid:
        pred = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        sens_list.append(tp/(tp+fn) if (tp+fn)>0 else 0)
        spec_list.append(tn/(tn+fp) if (tn+fp)>0 else 0)

    plt.figure(figsize=(8,5))
    plt.plot(thr_grid, sens_list, marker='o', label='Sensitivity')
    plt.plot(thr_grid, spec_list, marker='s', label='Specificity')
    plt.xlabel("Decision threshold")
    plt.ylabel("Value")
    plt.title(f"Sensitivity–Specificity Tradeoff (AUROC {glob_auc:.3f})")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(base, "sens_spec_plot.png"))
    plt.close()

    # ——— Write results.txt ———
    with open(os.path.join(base, "results.txt"), "w") as f:
        # Per-participant results first
        f.write("Per-participant AUROC (row-bootstrap B=1000):\n")
        
        # Sort PIDs for consistent output
        sorted_pids = sorted(te_by_pid.keys())
        
        for pid in sorted_pids:
            df = te_by_pid[pid]
            Xp = df[feats]
            yp = df["BP_spike"].values
            
            # Check if enough positive cases (at least 2 positive and 2 negative)
            n_pos = yp.sum()
            n_neg = len(yp) - n_pos
            
            if n_pos < 2 or n_neg < 2:
                f.write(f"PID {pid:02d}: NA (too few cases)\n")
                continue
                
            yx = xgb_best.predict_proba(Xp)[:,1]
            yl = lstm_best.predict(scaler.transform(Xp).reshape((len(Xp), len(feats), 1))).ravel()
            pr = alpha * yx + (1 - alpha) * yl

            boot = []
            for _ in range(B):
                idx = rng.choice(len(yp), size=len(yp), replace=True)
                # Need at least one positive and one negative in bootstrap sample
                if yp[idx].sum() > 0 and yp[idx].sum() < len(idx):
                    try:
                        boot.append(roc_auc_score(yp[idx], pr[idx]))
                    except ValueError:
                        pass
            
            if len(boot) >= 100:  # Need enough successful bootstraps
                m = np.mean(boot)
                sd = np.std(boot, ddof=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                f.write(f"PID {pid:02d}: {m:.3f} ±{sd:.3f} (95% CI {lo:.3f}–{hi:.3f})\n")
            else:
                f.write(f"PID {pid:02d}: NA (too few cases)\n")
        
        # Best parameters
        f.write(f"\nBest XGB params: {best_xgb_params}\n")
        if isinstance(best_lstm_val_auc, str):
            f.write(f"Best LSTM val_auc: {best_lstm_val_auc}\n")
        else:
            f.write(f"Best LSTM val_auc: {best_lstm_val_auc:.3f}\n")
        
        # Ensemble results
        f.write(f"\nEnsemble α: {alpha:.2f} | Global AUROC: {glob_auc:.3f}\n")
        f.write(f"Optimal thr: {best_thr:.2f} | Sens {final_s:.2f} | Spec {final_sp:.2f}\n")
        
        # Bootstrap results
        f.write("\nPer-participant AUROC:\n")
        f.write(f"Row-bootstrap AUROC {auc_m:.3f} ±{auc_sd:.3f} (95% CI {auc_lo:.3f}–{auc_hi:.3f})\n")
        f.write(f"Cluster-bootstrap AUROC {cl_m:.3f} ±{cl_sd:.3f} (95% CI {cl_lo:.3f}–{cl_hi:.3f})\n")

    # ——— Final elapsed time ———
    elapsed = int(time.time() - t0)
    print(f"\nFinished in {elapsed//3600:02d}h {(elapsed%3600)//60:02d}m {(elapsed%60):02d}s")

if __name__ == "__main__":
    main()