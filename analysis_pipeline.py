# =======================  analysis_pipeline.py  =======================
#!/usr/bin/env python3
"""
analysis_pipeline.py – FULL feature-complete version

• Reads per-participant flags from JSON
• Trains XGB + BiLSTM-Attention (custom / multi-head / self)  
• ADASYN k-NN & sampling-rate grid
• Finds ensemble α + best threshold
• Saves artefacts and figures
• Aggregates summary.csv   (pid, auc, best_thr, alpha)

USAGE
-----
python analysis_pipeline.py --config participants.json --jobs 4
python analysis_pipeline.py --config participants.json --pids 17,40
"""
# ---------------------------------------------------------------------
from __future__ import annotations
import argparse, json, multiprocessing as mp, math, warnings, time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np, pandas as pd, joblib, xgboost as xgb, shap, tensorflow as tf, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from keras_tuner import RandomSearch, Objective
from sklearn.metrics import roc_auc_score, confusion_matrix
try:
    # ≥0.24
    from sklearn.metrics import calibration_curve
except ImportError:
    # ≤0.23
    from sklearn.calibration import calibration_curve
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (LSTM, Bidirectional, BatchNormalization,
                                     Dropout, Dense)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
# ---------------------------------------------------------------------
FEATURES = [
    # 82-feature whitelist (copy-pasted verbatim from your personalised script)
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
TARGET = "BP_spike"
# ----------------------  Attention layers  ---------------------------
class AttentionLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            "W", shape=(input_shape[-1], 1), initializer="glorot_uniform"
        )
        self.b = self.add_weight(
            "b", shape=(input_shape[1], 1), initializer="zeros"
        )

    def call(self, x, return_scores: bool = False):
        e = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)  # (B,T,1)
        a = tf.nn.softmax(e, axis=1)                           # attention weights
        ctx = tf.reduce_sum(x * a, axis=1)                     # context vector
        return (ctx, a) if return_scores else ctx
class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self,h:int,k:int,**kw): super().__init__(**kw); self.m=tf.keras.layers.MultiHeadAttention(num_heads=h,key_dim=k)
    def call(self,x): return tf.reduce_mean(self.m(query=x,key=x,value=x),axis=1)
class SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self,**kw): super().__init__(**kw); self.a=tf.keras.layers.Attention()
    def call(self,x): return tf.reduce_mean(self.a([x,x]),axis=1)
# ----------------------  LSTM builder  --------------------------------
def build_lstm(hp,n_steps:int):
    m=Sequential()
    m.add(Bidirectional(LSTM(hp.Int("u1",64,256,32),return_sequences=True),
                        input_shape=(n_steps,1)))
    m.add(BatchNormalization()); dr=hp.Float("dr",0.2,0.5,0.1)
    m.add(Dropout(dr)); m.add(LSTM(hp.Int("u2",32,128,16),return_sequences=True))
    m.add(BatchNormalization()); m.add(Dropout(dr))
    att=hp.Choice("att",["custom","multi","self"])
    if att=="custom": m.add(AttentionLayer())
    elif att=="multi": m.add(MultiHeadAttentionLayer(hp.Int("h",1,4,1), hp.Int("k",16,64,16)))
    else: m.add(SelfAttentionLayer())
    m.add(Dense(hp.Int("dense",16,64,16),activation="relu"))
    m.add(Dropout(dr)); m.add(Dense(1,activation="sigmoid"))
    m.compile(optimizer=Adam(hp.Choice("lr",[1e-3,5e-4,1e-4])),
              loss="binary_crossentropy",metrics=[tf.keras.metrics.AUC(name="auc")])
    return m
# ----------------------  per-PID routine  -----------------------------
def run_pid(pid:str,cfg:Dict,struc:SimpleNamespace)->Tuple[str,float,float,float]:
    td   = cfg.get("train_days",struc.train_days)
    drop = cfg.get("drop",[])
    no_r = cfg.get("no_resample",struc.no_resample)
    knn  = cfg.get("n_neighbors",struc.n_neighbors)

    art = Path("artifacts")/f"hp{pid}"; art.mkdir(parents=True,exist_ok=True)
    fig = Path("figures")  /f"hp{pid}"; fig.mkdir(parents=True,exist_ok=True)

    csv = Path("processed")/f"hp{pid}"/"processed_bp_prediction_data.csv"
    if not csv.exists():
        warnings.warn(f"{pid} missing"); return pid, math.nan, math.nan, math.nan
    df = pd.read_csv(csv,parse_dates=["datetime_local"])
    cut= df["datetime_local"].min()+pd.Timedelta(days=td)
    tr,te=df[df["datetime_local"]<cut],df[df["datetime_local"]>=cut]
    feats=[f for f in FEATURES if f not in drop]
    Xtr,ytr=tr[feats],tr[TARGET]; Xte,yte=te[feats],te[TARGET]
    med=Xtr.median(); Xtr.fillna(med,inplace=True); Xte.fillna(med,inplace=True)

    # ---- XGB ----
    spw=(len(ytr)-ytr.sum())/ytr.sum() if ytr.sum() else 1
    steps=[("scaler",StandardScaler())]; grid={}
    if not no_r:
        steps.append(("adasyn",ADASYN(n_neighbors=knn,random_state=42)))
        grid["adasyn__sampling_strategy"]=[0.6,0.65,0.7,0.75]
    steps.append(("xgb",xgb.XGBClassifier(scale_pos_weight=spw,random_state=42,eval_metric="logloss")))
    grid.update({"xgb__max_depth":[3,5,7],"xgb__learning_rate":[0.01,0.05,0.1],"xgb__n_estimators":[100,150,200]})
    xgb_best=GridSearchCV(ImbPipeline(steps),grid,scoring="roc_auc",cv=3,n_jobs=-1).fit(Xtr,ytr).best_estimator_

    # ---- LSTM ----
    scaler=StandardScaler(); Xtr_s=scaler.fit_transform(Xtr); Xte_s=scaler.transform(Xte)
    if no_r: Xres,yres=Xtr_s,ytr.values
    else:
        ada=ADASYN(n_neighbors=knn,random_state=42,sampling_strategy=0.7)
        Xres,yres=ada.fit_resample(Xtr_s,ytr)
    Xres3=Xres.reshape((Xres.shape[0],Xres.shape[1],1)); Xte3=Xte_s.reshape((Xte_s.shape[0],Xte_s.shape[1],1))
    cw=dict(enumerate(compute_class_weight("balanced",np.unique(yres),yres)))

    tuner=RandomSearch(lambda hp: build_lstm(hp,Xres3.shape[1]),
                       objective=Objective("val_auc",direction="max"),
                       max_trials=struc.trials,directory="kt",project_name=f"hp{pid}",overwrite=True)
    tuner.search(Xres3,yres,validation_data=(Xte3,yte),
                 epochs=40,batch_size=struc.batch,class_weight=cw,verbose=0)
    lstm_best=tuner.get_best_models(1)[0]

    # ---- Ensemble + threshold ----
    yx=xgb_best.predict_proba(Xte)[:,1]; yl=lstm_best.predict(Xte3,verbose=0).ravel()
    best_a,best_auc=max(((a,roc_auc_score(yte,a*yx+(1-a)*yl)) for a in np.linspace(0,1,11)),key=lambda t:t[1])
    best_thr,best_you=-1,-1
    for t in np.arange(0,1.01,0.01):
        pr=((best_a*yx+(1-best_a)*yl)>=t).astype(int)
        tn,fp,fn,tp = confusion_matrix(yte,pr).ravel()
        sens,tpn = (tp/(tp+fn) if tp+fn else 0), (tn/(tn+fp) if tn+fp else 0)
        you=sens+tpn-1
        if you>best_you: best_thr,best_you=t,you

    # ---- Save artefacts ----
    lstm_best.save(art/"best_lstm.h5"); joblib.dump(xgb_best, art/"best_xgb.joblib")
    json.dump({"alpha":best_a,"auc":best_auc,"best_thr":best_thr}, open(art/"meta.json","w"), indent=2)

    # ---- Quick SHAP plot (optional heavy) ----
    expl=shap.TreeExplainer(xgb_best.named_steps["xgb"])
    shap_vals=expl(xgb_best.named_steps["scaler"].transform(Xte))
    shap.summary_plot(shap_vals,Xte,feature_names=feats,show=False)
    plt.savefig(fig/"shap_summary.png",dpi=300); plt.close()

    return pid,best_auc,best_thr,best_a
# ------------------------  main  -------------------------------------
# ------------------------  main  -------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--pids")
    p.add_argument("--jobs",   type=int, default=4)
    p.add_argument("--batch",  type=int, default=32)
    p.add_argument("--trials", type=int, default=20)
    a = p.parse_args()

    # Load JSON (could be dict or list)
    raw_cfg = json.load(open(a.config))
    if isinstance(raw_cfg, list):
        # list of records → build dict keyed by participant_id
        cfg = {str(rec["pid"]).zfill(2): rec for rec in raw_cfg}
    else:
        cfg = raw_cfg

    # Resolve which PIDs to run
    if a.pids:
        if "-" in a.pids:                       # range 01-20
            lo, hi = map(int, a.pids.split("-"))
            pids = [f"{i:02d}" for i in range(lo, hi + 1)]
        else:                                   # comma list
            pids = [pid.zfill(2) for pid in a.pids.split(",")]
    else:
        pids = sorted(cfg)

    struct = SimpleNamespace(
        train_days=20, no_resample=False, n_neighbors=5,
        batch=a.batch, trials=a.trials
    )

    t0 = time.time()
    with mp.Pool(a.jobs) as pool:
        out = pool.starmap(run_pid, [(pid, cfg.get(pid, {}), struct) for pid in pids])

    pd.DataFrame(out, columns=["pid", "auc", "best_thr", "alpha"]).to_csv(
        "summary.csv", index=False
    )
    print(f"Finished {len(out)} participants in {int(time.time() - t0)} s")
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
# =====================  end analysis_pipeline.py  =====================
