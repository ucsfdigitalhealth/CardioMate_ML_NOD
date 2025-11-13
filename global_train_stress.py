#!/usr/bin/env python3
"""
global_train_stress.py
======================

Train a *global* ensemble (XGBoost + BiLSTM-Attention) for binary
stress prediction (`stress_high`) across multiple participants, while
preserving each participant’s original train/test split.
"""

import argparse, json, os, time, pickle, joblib, warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import keras_tuner as kt

# ---------------- Features (same as personalized stress trainer) ----------------
STRESS_FEATURES = [
    # HR/steps (5,10,30,60)
    'hr_mean_5min','hr_min_5min','hr_max_5min','hr_std_5min',
    'steps_total_5min','steps_mean_5min','steps_min_5min','steps_max_5min','steps_std_5min','steps_diff_5min',
    'hr_mean_10min','hr_min_10min','hr_max_10min','hr_std_10min',
    'steps_total_10min','steps_mean_10min','steps_min_10min','steps_max_10min','steps_std_10min','steps_diff_10min',
    'hr_mean_30min','hr_min_30min','hr_max_30min','hr_std_30min',
    'steps_total_30min','steps_mean_30min','steps_min_30min','steps_max_30min','steps_std_30min','steps_diff_30min',
    'hr_mean_60min','hr_min_60min','hr_max_60min','hr_std_60min',
    'steps_total_60min','steps_mean_60min','steps_min_60min','steps_max_60min','steps_std_60min','steps_diff_60min',
    # BP window stats
    'sbp_mean','sbp_min','sbp_max','sbp_std',
    'dbp_mean','dbp_min','dbp_max','dbp_std',
    'bp_spike_any',
    # Lags
    'bp_spike_any_lag_1','bp_spike_any_lag_3','bp_spike_any_lag_5',
    'hr_mean_5min_lag_1','hr_mean_5min_lag_3','hr_mean_5min_lag_5',
    'steps_total_10min_lag_1','steps_total_10min_lag_3','steps_total_10min_lag_5',
    'sbp_mean_lag_1','sbp_mean_lag_3','sbp_mean_lag_5',
    'dbp_mean_lag_1','dbp_mean_lag_3','dbp_mean_lag_5',
    # Ratios / rolling
    'hr_steps_ratio','steps_hr_variability_ratio',
    'hr_mean_rolling_3','steps_total_rolling_5','hr_std_rolling_3',
    # Calendar / recency
    'hour_of_day','day_of_week','is_working_hours','is_weekend',
    'time_since_last_bp','time_since_last_BP_spike'
]

TARGET = "stress_high"
DATECOL = "local_created_at"

# ---------------- Custom attention layers (as in your personalized script) -----
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs); self.supports_masking=True; self._attention_weights=None
    def build(self, input_shape):
        self.W = self.add_weight("att_weight", shape=(input_shape[-1],1), initializer="normal")
        self.b = self.add_weight("att_bias",   shape=(input_shape[1],1), initializer="zeros")
        super().build(input_shape)
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        self._attention_weights = a
        return tf.keras.backend.sum(x * a, axis=1)
    def get_attention_weights(self): return self._attention_weights

class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.supports_masking=True; self._attention_weights=None
    def call(self, x):
        out, w = self.mha(query=x, key=x, value=x, return_attention_scores=True)
        self._attention_weights = tf.reduce_mean(w, axis=1)
        return tf.reduce_mean(out, axis=1)
    def get_attention_weights(self): return self._attention_weights

class SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.att = tf.keras.layers.Attention()
        self.supports_masking=True; self._attention_weights=None
    def call(self, x):
        att_out, att_w = self.att([x, x], return_attention_scores=True)
        self._attention_weights = att_w
        return tf.reduce_mean(att_out, axis=1)
    def get_attention_weights(self): return self._attention_weights

# ---------------- Utilities ----------------------------------------------------
def mean_sd_ci(arr):
    arr = np.asarray(arr, dtype=float)
    m  = np.nanmean(arr); sd = np.nanstd(arr, ddof=1)
    lo, hi = np.nanpercentile(arr, [2.5, 97.5])
    return m, sd, lo, hi

def extract_attention_weights_and_latents(model, X, batch_size=32):
    if not model.built:
        model.build(input_shape=(None,) + X.shape[1:])
    # find attention layer and preceding LSTM
    att_types = (AttentionLayer, MultiHeadAttentionLayer, SelfAttentionLayer)
    att_idx = next((i for i, lyr in enumerate(model.layers) if isinstance(lyr, att_types)), None)
    if att_idx is None: raise RuntimeError("No attention layer found.")
    lstm_layer = model.layers[att_idx - 1]
    att_layer  = model.layers[att_idx]
    model_input = getattr(model, "input", model.layers[0].input)
    lstm_out = Model(model_input, lstm_layer.output).predict(X, batch_size=batch_size, verbose=0)  # (B,T,F)

    def _norm(a):
        a = np.asarray(a, dtype=np.float32)
        return a / (a.sum(axis=1, keepdims=True) + 1e-8)

    if isinstance(att_layer, AttentionLayer):
        W = att_layer.W.numpy()
        b_arr = att_layer.b.numpy()
        if b_arr.ndim == 2: b_arr = b_arr.reshape((1, b_arr.shape[0], 1))
        e = np.tanh(np.matmul(lstm_out, W) + b_arr)   # (B,T,1)
        a = np.squeeze(e, -1)
        a = _norm(np.exp(a - a.max(axis=1, keepdims=True)))
    elif isinstance(att_layer, MultiHeadAttentionLayer):
        tinp = tf.keras.Input(shape=lstm_out.shape[1:])
        _, raw = att_layer.mha(tinp, tinp, tinp, return_attention_scores=True)  # (B,H,T,T)
        attm = Model(tinp, raw)
        raw = attm.predict(lstm_out, batch_size=batch_size, verbose=0)
        a = _norm(raw.mean(axis=(1,2)))
    else:
        tmp = tf.keras.layers.Attention()
        q = tf.convert_to_tensor(lstm_out, dtype=tf.float32)
        _, raw = tmp([q, q], return_attention_scores=True)  # (B,T,T)
        a = _norm(tf.reduce_mean(raw, axis=1).numpy())

    ctx = np.einsum('bt,btf->bf', a, lstm_out)   # (B,F)
    return a, lstm_out, ctx

# ---------------- Args ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Global stress_high trainer")
    p.add_argument("--settings", required=True, help="JSON settings with [{'pid':..,'train_days':..,'drop':[...]}...]")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lstm_trials", type=int, default=20)
    p.add_argument("--no_resample", action="store_true")
    p.add_argument("--n_neighbors", type=int, default=5)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

# ---------------- Data I/O -----------------------------------------------------
def load_participant_df(pid):
    path = os.path.join("processed_stress", f"hp{pid}", "processed_stress_prediction_data.csv")
    if not os.path.exists(path): raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=[DATECOL])
    df["pid"] = pid
    return df

def build_feature_intersection(settings):
    inter = set(STRESS_FEATURES)
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
        cutoff = df[DATECOL].min() + pd.Timedelta(days=days)
        tr, te = df[df[DATECOL] < cutoff], df[df[DATECOL] >= cutoff]
        tr[valid] = tr[valid].apply(pd.to_numeric, errors="coerce")
        te[valid] = te[valid].apply(pd.to_numeric, errors="coerce")
        gtrain.append(tr[valid + [TARGET, "pid"]])
        gtest.append(te[valid + [TARGET, "pid"]])
        per_pid_test[pid] = te[valid + [TARGET]]
    gtr = pd.concat(gtrain, ignore_index=True)
    gte = pd.concat(gtest,  ignore_index=True)
    # final X, y
    return gtr[feats], gtr[TARGET], gte[feats], gte[TARGET], per_pid_test

# ---------------- XGB grid -----------------------------------------------------
def xgb_grid(spw, resample, neighbors, verbose):
    if resample:
        steps = [
            ("scaler", StandardScaler()),
            ("adasyn", ADASYN(n_neighbors=neighbors, random_state=42)),
            ("xgb", xgb.XGBClassifier(
                tree_method="hist",  # robust CPU path
                scale_pos_weight=spw,
                random_state=42,
                eval_metric="logloss"
            )),
        ]
        grid = {
            "adasyn__sampling_strategy": [0.6, 0.7, 0.75],
            "xgb__max_depth": [3,5,7],
            "xgb__learning_rate": [0.01,0.05,0.1],
            "xgb__n_estimators": [100,150,200],
        }
    else:
        steps = [
            ("scaler", StandardScaler()),
            ("xgb", xgb.XGBClassifier(
                tree_method="hist",
                scale_pos_weight=spw,
                random_state=42,
                eval_metric="logloss"
            )),
        ]
        grid = {
            "xgb__max_depth": [3,5,7],
            "xgb__learning_rate": [0.01,0.05,0.1],
            "xgb__n_estimators": [100,150,200],
        }
    return GridSearchCV(ImbPipeline(steps), grid, scoring="roc_auc", cv=3, n_jobs=-1, verbose=int(verbose))

# ---------------- LSTM tuner ---------------------------------------------------
def build_lstm_tuner(Xtr, ytr, Xval, yval, trials, batch):
    def mb(hp):
        m = Sequential()
        m.add(Bidirectional(LSTM(hp.Int("u1",64,256,32), return_sequences=True),
                            input_shape=(Xtr.shape[1],1)))
        m.add(BatchNormalization())
        dr = hp.Float("drop", 0.2, 0.5, 0.1)
        m.add(Dropout(dr))
        m.add(LSTM(hp.Int("u2",32,128,16), return_sequences=True))
        m.add(BatchNormalization()); m.add(Dropout(dr))
        att = hp.Choice("att", ["custom","multi","self"])
        if att=="custom":
            m.add(AttentionLayer())
        elif att=="multi":
            m.add(MultiHeadAttentionLayer(hp.Int("heads",1,4,1), hp.Int("kdim",16,64,16)))
        else:
            m.add(SelfAttentionLayer())
        m.add(Dense(hp.Int("dense",16,64,16),
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(hp.Choice("reg",[0.0,0.001,0.01,0.1]))))
        m.add(Dropout(dr))
        m.add(Dense(1, activation="sigmoid"))
        m.compile(optimizer=Adam(hp.Choice("lr",[1e-3,5e-4,1e-4])),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])
        return m

    tuner = kt.RandomSearch(
        mb,
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=trials,
        directory="lstm_tuner",
        project_name="global_stress",
        overwrite=True
    )
    tuner.search(Xtr, ytr, validation_data=(Xval, yval),
                 epochs=50, batch_size=batch, verbose=1)
    return tuner

# ---------------- Ensemble alpha ----------------------------------------------
def ensemble_alpha(y_true, y_xgb, y_lstm):
    best_a, best_auc = None, -1
    for a in np.linspace(0,1,11):
        auc = roc_auc_score(y_true, a*y_xgb + (1-a)*y_lstm)
        if auc > best_auc:
            best_a, best_auc = a, auc
    return best_a, best_auc

# ---------------- Main ---------------------------------------------------------
def main():
    args = parse_args()
    if args.cpu: os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    t0 = time.time()

    base = os.path.join("results_stress", "global")
    os.makedirs(base, exist_ok=True)
    xgb_path  = os.path.join(base, "xgb_global.joblib")
    lstm_path = os.path.join(base, "lstm_global_model.keras")

    settings = json.load(open(args.settings))
    feats = build_feature_intersection(settings)
    print(f"🔹 {len(feats)} common numeric features retained.")
    Xtr, ytr, Xte, yte, te_by_pid = build_datasets(settings, feats)
    print(f"🔹 Global-train: {Xtr.shape}, Global-test: {Xte.shape}")

    # Fill NaNs with train medians
    med = Xtr.median()
    Xtr = Xtr.fillna(med)
    Xte = Xte.fillna(med)
    for pid, df in te_by_pid.items():
        te_by_pid[pid] = df.fillna({c: med[c] for c in med.index if c in df})

    # ----- XGB -----
    if os.path.exists(xgb_path):
        print("🔹 Loading existing XGB model…")
        xgb_best = joblib.load(xgb_path)
        best_xgb_params = "Loaded from file"
    else:
        print("🔹 Training XGB model…")
        pos = ytr.sum()
        spw = (len(ytr) - pos) / pos if pos else 1
        gs = xgb_grid(spw, not args.no_resample, args.n_neighbors, args.verbose)
        gs.fit(Xtr, ytr)
        xgb_best = gs.best_estimator_
        best_xgb_params = gs.best_params_
        print("🔹 Best XGB params:", best_xgb_params)
        joblib.dump(xgb_best, xgb_path)

    # ----- LSTM -----
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    if args.no_resample:
        Xres, yres = Xtr_s, ytr.values
    else:
        # try ADASYN; fall back gracefully if it cannot generate samples
        try:
            if hasattr(xgb_best, "named_steps") and "adasyn" in xgb_best.named_steps:
                ss = xgb_best.named_steps["adasyn"].sampling_strategy
            else:
                ss = 0.7
            ada = ADASYN(n_neighbors=args.n_neighbors, random_state=42, sampling_strategy=ss)
            Xres, yres = ada.fit_resample(Xtr_s, ytr)
        except ValueError:
            print("⚠️  ADASYN could not generate samples; using original training set.")
            Xres, yres = Xtr_s, ytr.values

    Xres3 = Xres.reshape((Xres.shape[0], Xres.shape[1], 1))
    Xte3  = Xte_s.reshape((Xte_s.shape[0], Xte_s.shape[1], 1))

    if os.path.exists(lstm_path):
        print("🔹 Loading existing LSTM model…")
        lstm_best = tf.keras.models.load_model(
            lstm_path,
            custom_objects={
                "AttentionLayer": AttentionLayer,
                "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
                "SelfAttentionLayer": SelfAttentionLayer
            }
        )
        best_lstm_val_auc = "Loaded from file"
    else:
        print("🔹 Tuning & training LSTM model…")
        tuner = build_lstm_tuner(Xres3, yres, Xte3, yte, trials=args.lstm_trials, batch=args.batch)
        lstm_best = tuner.get_best_models(1)[0]
        best_lstm_val_auc = tuner.oracle.get_best_trials(1)[0].metrics.get_best_value("val_auc")
        print(f"🔹 Best LSTM val_auc: {best_lstm_val_auc:.3f}")
        lstm_best.save(lstm_path)

    # ----- Extract global attention/latents -----
    print("🔹 Extracting global attention/latent/context…")
    g_att, g_lat, g_ctx = extract_attention_weights_and_latents(lstm_best, Xte3, args.batch)

    # ----- Per-PID predictions & extractions -----
    per = {}
    for pid, df in te_by_pid.items():
        Xp = df[feats]; yp = df[TARGET].values
        Xp_s = scaler.transform(Xp).reshape((len(Xp), len(feats), 1))
        yx = xgb_best.predict_proba(Xp)[:,1]
        yl = lstm_best.predict(Xp_s, verbose=0).ravel()
        att, lat, ctx = extract_attention_weights_and_latents(lstm_best, Xp_s, args.batch)
        per[pid] = {
            "attention_weights": att,
            "latent_vectors": lat,
            "context_vectors": ctx,
            "y_test": yp,
            "xgb_probs": yx,
            "lstm_probs": yl
        }

    with open(os.path.join(base, "global_analysis_data.pkl"), "wb") as f:
        pickle.dump({
            "global": {
                "attention_weights": g_att,
                "latent_vectors": g_lat,
                "context_vectors": g_ctx,
                "y_test": yte.values
            },
            "per_pid": per
        }, f)

    # ----- Ensemble & evaluation -----
    y_xgb  = xgb_best.predict_proba(Xte)[:,1]
    y_lstm = lstm_best.predict(Xte3, verbose=0).ravel()
    alpha, glob_auc = ensemble_alpha(yte, y_xgb, y_lstm)
    print(f"🔹 Best α = {alpha:.2f} | Global AUROC = {glob_auc:.3f}")
    scores = alpha*y_xgb + (1-alpha)*y_lstm

    # ----- Bootstraps -----
    B = 1000; rng = np.random.default_rng(42)
    row_auc = [roc_auc_score(yte.iloc[idx], scores[idx])
               for idx in (rng.choice(len(yte), replace=True, size=len(yte)) for _ in range(B))]
    auc_m, auc_sd, auc_lo, auc_hi = mean_sd_ci(row_auc)
    print(f"\nAUROC (row-bootstrap) {auc_m:.3f} ±{auc_sd:.3f} (95% CI {auc_lo:.3f}–{auc_hi:.3f})")

    pid_rows = {pid: te_by_pid[pid].index.values for pid in te_by_pid}
    clust_auc = []
    for _ in range(B):
        samp = rng.choice(list(pid_rows), replace=True, size=len(pid_rows))
        rows = np.concatenate([pid_rows[p] for p in samp]) if len(samp) else np.array([], dtype=int)
        clust_auc.append(roc_auc_score(yte.iloc[rows], scores[rows]) if len(rows) else np.nan)
    cl_m, cl_sd, cl_lo, cl_hi = mean_sd_ci([a for a in clust_auc if not np.isnan(a)])
    print(f"AUROC (cluster-bootstrap) {cl_m:.3f} ±{cl_sd:.3f} (95% CI {cl_lo:.3f}–{cl_hi:.3f})\n")

    # ----- Threshold scan -----
    best_thr, best_y, final_s, final_sp = None, -1, None, None
    thr_grid = np.arange(0, 1.01, 0.01)
    for t in thr_grid:
        pred = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0,1]).ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
        youden = sens + spec - 1
        print(f"Threshold {t:.2f} | Sens {sens:.2f} | Spec {spec:.2f} | Youden {youden:.2f}")
        if youden > best_y:
            best_y, best_thr, final_s, final_sp = youden, t, sens, spec
    print(f"\n🔹 Optimal thr {best_thr:.2f} | Sens {final_s:.2f} | Spec {final_sp:.2f}")

    # ----- Per-PID AUROC (row bootstrap) -----
    print("\nPer-participant AUROC (row-bootstrap B=1000):")
    Bpid = 1000
    glob_aurocs = {}
    for pid, df in te_by_pid.items():
        Xp = df[feats]; yp = df[TARGET].values
        yx = xgb_best.predict_proba(Xp)[:,1]
        yl = lstm_best.predict(scaler.transform(Xp).reshape((len(Xp), len(feats), 1)), verbose=0).ravel()
        pr = alpha*yx + (1-alpha)*yl
        glob_aurocs[pid] = np.nan
        boot = []
        for _ in range(Bpid):
            idx = rng.choice(len(yp), size=len(yp), replace=True)
            if yp[idx].sum() > 0 and yp[idx].sum() < len(idx):
                try: boot.append(roc_auc_score(yp[idx], pr[idx]))
                except ValueError: pass
        if len(boot) >= 100:
            m = np.mean(boot); sd = np.std(boot, ddof=1); lo, hi = np.percentile(boot, [2.5,97.5])
            glob_aurocs[pid] = roc_auc_score(yp, pr) if (yp.sum()>0 and yp.sum()<len(yp)) else np.nan
            print(f"PID {pid:02d}: {m:.3f} ±{sd:.3f} (95% CI {lo:.3f}–{hi:.3f})")
        else:
            print(f"PID {pid:02d}: NA (too few cases)")

    with open(os.path.join(base, "global_aurocs.pkl"), "wb") as f:
        pickle.dump(glob_aurocs, f)

    # ----- SHAP summary (XGB) -----
    expl = shap.Explainer(xgb_best.named_steps["xgb"])
    idx = np.random.choice(len(Xte), size=min(len(Xte), 1000), replace=False)
    shap_vals = expl(xgb_best.named_steps["scaler"].transform(Xte.iloc[idx]))
    plt.figure(figsize=(9,6))
    shap.summary_plot(shap_vals, Xte.iloc[idx], feature_names=feats, show=False)
    plt.title("Global — SHAP summary (XGBoost)", pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(base, "shap_summary_global.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ----- Sens–Spec plot -----
    sens_list, spec_list = [], []
    for t in thr_grid:
        pred = (scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0,1]).ravel()
        sens_list.append(tp/(tp+fn) if (tp+fn)>0 else 0.0)
        spec_list.append(tn/(tn+fp) if (tn+fp)>0 else 0.0)

    plt.figure(figsize=(8,5))
    plt.plot(thr_grid, sens_list, marker='o', label='Sensitivity')
    plt.plot(thr_grid, spec_list, marker='s', label='Specificity')
    plt.axvline(best_thr, linestyle='--', linewidth=1, alpha=0.7,
                label=f'Best thr = {best_thr:.2f}')
    plt.xlabel("Decision threshold")
    plt.ylabel("Value")
    plt.ylim(0,1)
    plt.title(f"Global — Sensitivity & Specificity vs Threshold (AUROC {glob_auc:.3f})", pad=10)
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(base, "sens_spec_global.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ----- results.txt -----
    with open(os.path.join(base, "results.txt"), "w") as f:
        f.write(f"Global-train: {Xtr.shape}, Global-test: {Xte.shape}\n")
        f.write(f"Best XGB params: {best_xgb_params}\n")
        if isinstance(best_lstm_val_auc, str):
            f.write(f"Best LSTM val_auc: {best_lstm_val_auc}\n")
        else:
            f.write(f"Best LSTM val_auc: {best_lstm_val_auc:.3f}\n")
        f.write(f"\nEnsemble α: {alpha:.2f} | Global AUROC: {glob_auc:.3f}\n")
        f.write(f"Optimal thr: {best_thr:.2f} | Sens {final_s:.2f} | Spec {final_sp:.2f}\n")
        f.write(f"\nRow-bootstrap AUROC {auc_m:.3f} ±{auc_sd:.3f} (95% CI {auc_lo:.3f}–{auc_hi:.3f})\n")
        f.write(f"Cluster-bootstrap AUROC {cl_m:.3f} ±{cl_sd:.3f} (95% CI {cl_lo:.3f}–{cl_hi:.3f})\n")
        f.write("\nPer-participant AUROC (row-bootstrap B=1000):\n")
        for pid in sorted(glob_aurocs):
            f.write(f"PID {pid:02d}: {glob_aurocs[pid] if not np.isnan(glob_aurocs[pid]) else 'NA'}\n")

    elapsed = int(time.time() - t0)
    print(f"\nFinished in {elapsed//3600:02d}h {(elapsed%3600)//60:02d}m {(elapsed%60):02d}s")

if __name__ == "__main__":
    main()
