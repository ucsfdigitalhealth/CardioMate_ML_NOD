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
import joblib
import pickle
from tensorflow.keras import Input

# ---------------- Attention layers (unchanged) ----------------
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self._attention_weights = None
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        self._attention_weights = a
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

def mean_sd_ci(arr):
    arr = np.asarray(arr, dtype=float)
    m  = np.nanmean(arr)
    sd = np.nanstd(arr, ddof=1)
    lo, hi = np.nanpercentile(arr, [2.5, 97.5])
    return m, sd, lo, hi

def extract_attention_weights_and_latents(model, X, batch_size=32):

    if not model.built:
        model.build(input_shape=(None,) + X.shape[1:])

    # locate attention layer and the layer just before it (your LSTM output)
    att_types = (AttentionLayer, MultiHeadAttentionLayer, SelfAttentionLayer)
    att_idx = next((i for i, lyr in enumerate(model.layers) if isinstance(lyr, att_types)), None)
    if att_idx is None:
        raise RuntimeError("No attention layer found in the model.")
    lstm_layer = model.layers[att_idx - 1]
    att_layer  = model.layers[att_idx]

    # (B, T, F)
    model_input = getattr(model, "input", model.layers[0].input)
    lstm_out = Model(model_input, lstm_layer.output).predict(X, batch_size=batch_size, verbose=0)

    def _norm(a):
        a = np.asarray(a, dtype=np.float32)
        s = a.sum(axis=1, keepdims=True) + 1e-8
        return a / s

    # compute attention weights as (B, T)
    if isinstance(att_layer, AttentionLayer):
        W = att_layer.W.numpy()               # (F, 1)
        b_arr = att_layer.b.numpy()           # (T, 1) or (T,)
        if b_arr.ndim == 2:
            b_arr = b_arr.reshape((1, b_arr.shape[0], 1))  # -> (1, T, 1)
        e = np.tanh(np.matmul(lstm_out, W) + b_arr)        # (B, T, 1)
        a = np.squeeze(e, -1)                              # (B, T)
        a = np.exp(a - a.max(axis=1, keepdims=True))
        a = _norm(a)
    elif isinstance(att_layer, MultiHeadAttentionLayer):
        tmp_in = tf.keras.Input(shape=lstm_out.shape[1:])
        _, raw = att_layer.mha(tmp_in, tmp_in, tmp_in, return_attention_scores=True)  # (B, H, T, T)
        att_model = Model(tmp_in, raw)
        raw = att_model.predict(lstm_out, batch_size=batch_size, verbose=0)
        a = raw.mean(axis=(1, 2))  # (B, T)
        a = _norm(a)
    else:  # SelfAttentionLayer
        tmp_att = tf.keras.layers.Attention()
        q = tf.convert_to_tensor(lstm_out, dtype=tf.float32)
        _, raw = tmp_att([q, q], return_attention_scores=True)  # (B, T, T)
        a = tf.reduce_mean(raw, axis=1).numpy()  # (B, T)
        a = _norm(a)

    # context vectors (B, F)
    context = np.einsum('bt,btf->bf', a, lstm_out)
    return a, lstm_out, context

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Train models to PREDICT STRESS (binary) using BP values/spikes, HR, steps")
    parser.add_argument('--participant_id', required=True, help="Participant ID, e.g., 31")
    parser.add_argument('--models', default='xgb,attn', help="Comma-separated list of models: xgb, attn")
    parser.add_argument('--drop', default='', help="Comma-separated features to drop")
    parser.add_argument('--train_days', type=int, default=20, help="Days used for train/test split (relative to stress timeline)")
    parser.add_argument('--batch', type=int, default=32, help="Batch size for LSTM training")
    parser.add_argument('--verbose', action='store_true', help="Verbose output")
    parser.add_argument('--cpu', action='store_true', help="Force CPU execution")
    parser.add_argument('--n_neighbors', type=int, default=5, help="Number of neighbors for ADASYN (default 5)")
    parser.add_argument('--no_resample', action='store_true', help="Skip ADASYN resampling entirely")
    args = parser.parse_args()

    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    start_time = time.time()

    base = os.path.join('results_stress', 'personalized', f'pid_{args.participant_id}')
    os.makedirs(base, exist_ok=True)

    # ---------- KEY CHANGE: load stress-anchored processed file ----------
    data_path = os.path.join('processed_stress', f'hp{args.participant_id}', 'processed_stress_prediction_data.csv')
    df = pd.read_csv(data_path, parse_dates=['local_created_at'])
    df['pid'] = int(args.participant_id)

    # Features: HR/steps rollups + BP window stats + lags + calendar + time-since
    features = [
        # HR/steps (5,10,30,60)
        'hr_mean_5min','hr_min_5min','hr_max_5min','hr_std_5min',
        'steps_total_5min','steps_mean_5min','steps_min_5min','steps_max_5min','steps_std_5min','steps_diff_5min',
        'hr_mean_10min','hr_min_10min','hr_max_10min','hr_std_10min',
        'steps_total_10min','steps_mean_10min','steps_min_10min','steps_max_10min','steps_std_10min','steps_diff_10min',
        'hr_mean_30min','hr_min_30min','hr_max_30min','hr_std_30min',
        'steps_total_30min','steps_mean_30min','steps_min_30min','steps_max_30min','steps_std_30min','steps_diff_30min',
        'hr_mean_60min','hr_min_60min','hr_max_60min','hr_std_60min',
        'steps_total_60min','steps_mean_60min','steps_min_60min','steps_max_60min','steps_std_60min','steps_diff_60min',

        # BP window stats around stress record
        'sbp_mean','sbp_min','sbp_max','sbp_std',
        'dbp_mean','dbp_min','dbp_max','dbp_std',
        'bp_spike_any',

        # Lags (defined in preprocess)
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
    target = 'stress_high'  # binary classification

    # Select and coerce
    df = df[['local_created_at', 'pid'] + features + [target]]
    df[features] = df[features].apply(pd.to_numeric, errors='coerce')

    # Time split on stress timeline
    cutoff = df['local_created_at'].min() + pd.Timedelta(days=args.train_days)
    train_df = df[df['local_created_at'] < cutoff]
    test_df  = df[df['local_created_at'] >= cutoff]
    X_train, y_train = train_df[features], train_df[target]
    X_test,  y_test  = test_df[features],  test_df[target]

    # Optional drops
    if args.drop:
        drops = [c for c in args.drop.split(',') if c in X_train.columns]
        X_train.drop(columns=drops, inplace=True)
        X_test.drop(columns=drops, inplace=True)

    # Class counts
    pos_tr = int(y_train.sum()); pct_tr = 100 * pos_tr / len(y_train) if len(y_train) else 0
    pos_te = int(y_test.sum());  pct_te = 100 * pos_te / len(y_test)  if len(y_test) else 0
    print("🔹 Stress-high counts (target=1) Before Resampling:")
    print(f"   - Training Set: {pos_tr} ({pct_tr:.2f}%)")
    print(f"   - Test Set:     {pos_te} ({pct_te:.2f}%)")

    # XGBoost pipeline (classification)
    pos, neg = sum(y_train==1), sum(y_train==0)
    spw = neg/pos if pos>0 else 1

    if args.no_resample:
        pipeline_steps = [('scaler', StandardScaler()),
                          ('xgb', xgb.XGBClassifier(scale_pos_weight=spw, random_state=42))]
        param_grid = {'xgb__max_depth': [3,5,7],
                      'xgb__learning_rate': [0.01,0.05,0.1],
                      'xgb__n_estimators': [100,150,200]}
    else:
        pipeline_steps = [('scaler', StandardScaler()),
                          ('adasyn', ADASYN(n_neighbors=args.n_neighbors, random_state=42)),
                          ('xgb', xgb.XGBClassifier(scale_pos_weight=spw, random_state=42))]
        param_grid = {'adasyn__sampling_strategy': [0.6,0.65,0.7,0.75],
                      'xgb__max_depth': [3,5,7],
                      'xgb__learning_rate': [0.01,0.05,0.1],
                      'xgb__n_estimators': [100,150,200]}

    pipeline = ImbPipeline(pipeline_steps)
    grid_search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=3, n_jobs=-1, verbose=int(args.verbose))
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    print("🔹 Best parameters from grid search (XGBoost pipeline):")
    print(grid_search.best_params_)

    # LSTM prep (treat features as a sequence over 'timesteps' = n_features)
    scaler_lstm = StandardScaler()
    X_tr_s = scaler_lstm.fit_transform(X_train)
    X_te_s = scaler_lstm.transform(X_test)

    if args.no_resample:
        X_res, y_res = X_tr_s, y_train.values
    else:
        ada_best = ADASYN(sampling_strategy=grid_search.best_params_['adasyn__sampling_strategy'],
                          n_neighbors=args.n_neighbors, random_state=42)
        X_res, y_res = ada_best.fit_resample(X_tr_s, y_train)

    X_train_lstm = X_res.reshape((X_res.shape[0], X_res.shape[1], 1))
    X_test_lstm  = X_te_s.reshape((X_te_s.shape[0], X_te_s.shape[1], 1))

    classes = np.unique(y_res)
    cw = compute_class_weight('balanced', classes=classes, y=y_res)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print("🔹 Class Weights for LSTM:", class_weight)

    # LSTM with attention
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
        model.add(Dense(du, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(drg)))
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
        project_name=f'stress_pred_{args.participant_id}',
        overwrite=True
    )

    t0_tuner = time.time()
    tuner.search(
        X_train_lstm, y_res,
        epochs=50,
        batch_size=args.batch,
        validation_data=(X_test_lstm, y_test),
        class_weight=class_weight,
        verbose=1
    )
    t1_tuner = time.time()
    tuner_time_str = time.strftime("%Hh %Mm %Ss", time.gmtime(t1_tuner - t0_tuner))

    best_lstm = tuner.get_best_models(num_models=1)[0]
    print("🔹 Best LSTM hyperparameters:", tuner.oracle.get_best_trials(1)[0].hyperparameters.values)

    # Attention/latents
    print("🔹 Extracting attention weights and context vectors...")
    attn_w, latent_vectors, context_vectors = extract_attention_weights_and_latents(best_lstm, X_test_lstm, args.batch)

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

    # Ensemble & threshold scan (AUROC)
    y_xgb  = best_xgb.predict_proba(X_test)[:,1]
    y_lstm = best_lstm.predict(X_test_lstm).flatten()
    alphas = np.linspace(0,1,11)
    best_auc, best_a = -1, None

    print("\n🔹 Ensemble Weight Grid Search:")
    for a in alphas:
        ens = a * y_xgb + (1-a) * y_lstm
        try:
            auc_v = roc_auc_score(y_test, ens)
        except ValueError:
            auc_v = float('nan')
        print(f"Alpha: {a:.2f}, Beta: {1-a:.2f}, AUROC: {auc_v:.3f}")
        if not np.isnan(auc_v) and auc_v > best_auc:
            best_auc, best_a = auc_v, a

    if best_a is None:
        best_a   = 1.0
        best_auc = roc_auc_score(y_test, y_xgb)
        print(f"\n🔹 No valid ensemble AUC → falling back to XGB only: Alpha = 1.00, Beta = 0.00, AUROC = {best_auc:.3f}")
    else:
        print(f"\n🔹 Best Ensemble Weights: Alpha = {best_a:.2f}, Beta = {1-best_a:.2f}, AUROC = {best_auc:.3f}")

    # ----- Threshold scan (collect lines for results.txt) -----
    thresholds = np.arange(0, 1.01, 0.01)
    best_youden, best_thr = -1, None
    final_sens = final_spec = None

    # compute once and reuse everywhere (scan, plot, bootstrap)
    ens_scores = best_a * y_xgb + (1 - best_a) * y_lstm

    threshold_lines = []   # <- to write into results.txt
    print("\n🔹 Threshold scan:")
    for t in thresholds:
        yb = (ens_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, yb, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden = sens + spec - 1

        line = f"Threshold: {t:.2f} | Sensitivity: {sens:.2f} | Specificity: {spec:.2f} | Youden Index: {youden:.2f}"
        print(line)
        threshold_lines.append(line)

        if youden > best_youden:
            best_youden, best_thr, final_sens, final_spec = youden, t, sens, spec

    print(f"\n🔹 Best Threshold in [0,1]: {best_thr:.2f}")
    print(f"🔹 Final Sensitivity (Recall): {final_sens:.3f}")
    print(f"🔹 Final Specificity: {final_spec:.3f}")


    # ---- Sens–Spec tradeoff plot (with participant ID) ----
    sens_list, spec_list = [], []
    for t in thresholds:
        yb = (ens_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, yb, labels=[0, 1]).ravel()
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, sens_list, marker='o', label='Sensitivity')
    plt.plot(thresholds, spec_list, marker='s', label='Specificity')
    plt.axvline(best_thr, linestyle='--', linewidth=1, alpha=0.7,
                label=f'Best thr = {best_thr:.2f}')
    plt.xlabel('Decision Threshold')
    plt.ylabel('Value')
    plt.ylim(0, 1)
    plt.title(
        f'Participant {args.participant_id} — Sensitivity & Specificity vs Threshold '
        f'(AUROC: {best_auc:.3f})',
        pad=10
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base, f'sens_spec_pid_{args.participant_id}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ---- SHAP for XGB (with participant ID in the title) ----
    scaler_xgb = best_xgb.named_steps['scaler']
    model_xgb  = best_xgb.named_steps['xgb']

    X_shap = scaler_xgb.transform(X_test)
    expl   = shap.Explainer(model_xgb)
    sv     = expl(X_shap)

    plt.figure(figsize=(9,6))  # fresh figure
    shap.summary_plot(sv, X_test, feature_names=X_test.columns, show=False)
    plt.title(f'Participant {args.participant_id} — SHAP summary (XGBoost)', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(base, f'shap_summary_pid_{args.participant_id}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Bootstrap (same as before)
    B = 1000
    rng = np.random.default_rng(42)
    #scores = best_a * y_xgb + (1 - best_a) * y_lstm
    scores = ens_scores  # reuse ensemble scores
    sens_boot = np.empty((B, len(thresholds)))
    spec_boot = np.empty((B, len(thresholds)))
    auc_boot  = np.empty(B)

    for b in range(B):
        idx = rng.choice(len(y_test), replace=True, size=len(y_test))
        yt = y_test.iloc[idx].astype(bool).values          # <- make boolean
        sc = scores[idx]
        auc_boot[b] = roc_auc_score(yt.astype(int), sc)    # roc_auc_score expects {0,1}

        for i, thr in enumerate(thresholds):
            yb = sc >= thr                                 # bool
            tp = (yt & yb).sum()
            fn = (yt & ~yb).sum()
            tn = (~yt & ~yb).sum()
            fp = (~yt &  yb).sum()
            sens_boot[b, i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec_boot[b, i] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def summary(arr):
        m = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, ddof=1, axis=0)
        lo, hi = np.nanpercentile(arr, [2.5, 97.5], axis=0)
        return m, sd, lo, hi

    sens_m, sens_sd, sens_lo, sens_hi = summary(sens_boot)
    spec_m, spec_sd, spec_lo, spec_hi = summary(spec_boot)
    auc_m, auc_sd, auc_lo, auc_hi = mean_sd_ci(auc_boot)

    print(f"\nAUROC (row-bootstrap) {auc_m:.3f} ±{auc_sd:.3f} (95% CI {auc_lo:.3f}–{auc_hi:.3f})")
    best_i = np.where(thresholds == best_thr)[0][0]
    print(f"[Youden thr {best_thr:.2f}] Sens {sens_m[best_i]:.2f}±{sens_sd[best_i]:.2f}, Spec {spec_m[best_i]:.2f}±{spec_sd[best_i]:.2f}")

    pd.DataFrame({
        'threshold': thresholds,
        'sens_mean': sens_m, 'sens_sd': sens_sd, 'sens_lo': sens_lo, 'sens_hi': sens_hi,
        'spec_mean': spec_m, 'spec_sd': spec_sd, 'spec_lo': spec_lo, 'spec_hi': spec_hi
    }).to_csv(os.path.join(base, 'bootstrap_threshold_stats.csv'), index=False)

    # Save models and AUROC
    joblib.dump(best_xgb, os.path.join(base, 'xgb_model.joblib'))
    best_lstm.save(os.path.join(base, 'lstm_model.keras'))
    with open(os.path.join(base, 'auroc_data.pkl'), 'wb') as f:
        pickle.dump({'pid': args.participant_id, 'personalized_auroc': best_auc}, f)

    # Best val AUC
    best_lstm = tuner.get_best_models(num_models=1)[0]
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    best_val = best_trial.metrics.get_best_value('val_auc')
    print(f"🔹 Best val_auc So Far: {best_val:.3f}")
    print("🔹 Best LSTM hyperparameters:", best_trial.hyperparameters.values)

    # Summary file  (AFTER bootstrap so all stats exist)
    total_elapsed = int(time.time() - start_time)
    th, rem = divmod(total_elapsed, 3600)
    tm, ts = divmod(rem, 60)
    total_time_str = f"{th:02d}h {tm:02d}m {ts:02d}s"

    results_path = os.path.join(base, 'results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Total elapsed time: {total_time_str}\n")
        f.write(f"Total tuner elapsed time: {tuner_time_str}\n")

        # Best val AUC + HParams (same wording/format as console)
        best_lstm = tuner.get_best_models(num_models=1)[0]
        best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
        best_val = best_trial.metrics.get_best_value('val_auc')
        f.write(f"🔹 Best val_auc So Far: {best_val:.3f}\n")
        f.write(f"🔹 Best LSTM hyperparameters: {best_trial.hyperparameters.values}\n\n")

        # Best ensemble line
        f.write(f"🔹 Best Ensemble Weights: Alpha = {best_a:.2f}, Beta = {1-best_a:.2f}, AUROC = {best_auc:.3f}\n")

        # Full threshold scan (every line)
        f.write("\n🔹 Threshold scan:\n")
        f.write("\n".join(threshold_lines) + "\n")

        # Best threshold summary from scan
        f.write(f"\n🔹 Best Threshold in [0,1]: {best_thr:.2f}\n")
        f.write(f"🔹 Final Sensitivity (Recall): {final_sens:.3f}\n")
        f.write(f"🔹 Final Specificity: {final_spec:.3f}\n")

        # Bootstrap AUROC + CI and Youden row with CIs
        f.write(f"\n🔹 AUROC (row-bootstrap) {auc_m:.3f} ±{auc_sd:.3f} (95% CI {auc_lo:.3f}–{auc_hi:.3f})\n")
        best_i = int(np.where(thresholds == best_thr)[0][0])
        f.write(f"[Youden thr {best_thr:.2f}] Sens {sens_m[best_i]:.2f}±{sens_sd[best_i]:.2f}, "
                f"Spec {spec_m[best_i]:.2f}±{spec_sd[best_i]:.2f}\n")

        # AUROC grid over alphas
        f.write("\n🔹 Ensemble Weight Grid Search (AUROC):\n")
        for a in alphas:
            auc_v = roc_auc_score(y_test, a*y_xgb + (1-a)*y_lstm)
            f.write(f"Alpha: {a:.2f}, Beta: {1-a:.2f}, AUROC: {auc_v:.3f}\n")

        f.write("\nBootstrapped CIs for all thresholds → bootstrap_threshold_stats.csv\n")


    concat_path = os.path.join('results_stress', 'personalized', 'all_participants_results.txt')
    with open(concat_path, 'a') as f:
        f.write(f"PID {args.participant_id}: AUROC {best_auc:.3f}, Thr {best_thr:.2f}\n")

    print(f"✅ Personalized results saved to {base}")
    print(f"\nTotal training time: {th:02d}h {tm:02d}m {ts:02d}s")

if __name__ == '__main__':
    main()
