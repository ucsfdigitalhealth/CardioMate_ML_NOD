#!/usr/bin/env bash
set -Eeuo pipefail

echo "🔁 Running grouped feature ablation with participant-specific settings (resume + logging)"
echo "========================================================================================"

# Stop immediately on Ctrl-C
trap 'echo "🛑 Stopped by Ctrl-C. Safe to resume later."; exit 130' INT

# --------------------------------
# Participants (same as your list)
# --------------------------------
PARTICIPANTS=(10 15 16 17 20 24 25 26 30 31 33 35 36 39 40)
#PARTICIPANTS=(34 32 18 22 23)

# --------------------------------
# Grouped feature drops (coarse)
# --------------------------------
declare -A FEATURE_GROUPS
FEATURE_GROUPS=(
  ["hr_features"]="--drop hr_mean_5min,hr_steps_ratio,steps_hr_variability_ratio,stress_weighted_hr,hr_mean_10min,hr_mean_30min,hr_mean_60min,hr_mean_rolling_3,hr_std_5min,hr_std_10min,hr_std_30min,hr_std_60min,hr_std_rolling_3"
  ["step_features"]="--drop steps_total_5min,steps_total_10min,steps_hr_variability_ratio,hr_steps_ratio,stress_steps_ratio,steps_total_30min,steps_total_60min,steps_total_rolling_5,steps_mean_5min,steps_mean_10min,steps_mean_30min,steps_mean_60min,steps_min_5min,steps_max_5min,steps_std_5min,steps_diff_5min,steps_min_10min,steps_max_10min,steps_std_10min,steps_diff_10min,steps_min_30min,steps_max_30min,steps_std_30min,steps_diff_30min,steps_min_60min,steps_max_60min,steps_std_60min,steps_diff_60min"
  ["stress_features"]="--drop stress_steps_ratio,stress_mean,stress_min,stress_weighted_hr,stress_max,stress_std"
  ["ratios"]="--drop hr_steps_ratio,stress_weighted_hr,stress_steps_ratio,steps_hr_variability_ratio"
  ["cumulative"]="--drop cumulative_stress_30min,cumulative_steps_30min"
  ["time_features"]="--drop hour_of_day,day_of_week,is_working_hours,is_weekend"
)

# -------------------------------------------------
# Per-participant PREPROCESS thresholds (your list)
# -------------------------------------------------
declare -A PREPROC_ARGS
PREPROC_ARGS[17]="--bp_sys_thresh 146 --bp_dia_thresh 100"
PREPROC_ARGS[40]="--bp_sys_thresh 130 --bp_dia_thresh 80"
PREPROC_ARGS[39]="--bp_sys_thresh 130 --bp_dia_thresh 80"
PREPROC_ARGS[36]="--bp_sys_thresh 101 --bp_dia_thresh 54"
PREPROC_ARGS[35]="--bp_sys_thresh 130 --bp_dia_thresh 80"
PREPROC_ARGS[34]="--bp_sys_thresh 156 --bp_dia_thresh 83"
PREPROC_ARGS[32]="--bp_sys_thresh 143 --bp_dia_thresh 83"
PREPROC_ARGS[25]="--bp_sys_thresh 130 --bp_dia_thresh 80"
PREPROC_ARGS[16]="--bp_sys_thresh 130 --bp_dia_thresh 90"
PREPROC_ARGS[10]="--bp_sys_thresh 100 --bp_dia_thresh 60"
PREPROC_ARGS[15]="--bp_sys_thresh 130 --bp_dia_thresh 80"
PREPROC_ARGS[18]="--bp_sys_thresh 130 --bp_dia_thresh 80"
PREPROC_ARGS[20]="--bp_sys_thresh 130 --bp_dia_thresh 80"
PREPROC_ARGS[22]="--bp_sys_thresh 120 --bp_dia_thresh 80"
PREPROC_ARGS[23]="--bp_sys_thresh 130 --bp_dia_thresh 80"
PREPROC_ARGS[24]="--bp_sys_thresh 130 --bp_dia_thresh 80"
PREPROC_ARGS[26]="--bp_sys_thresh 135 --bp_dia_thresh 85"
PREPROC_ARGS[33]="--bp_sys_thresh 130 --bp_dia_thresh 80"
PREPROC_ARGS[31]="--bp_sys_thresh 130 --bp_dia_thresh 80"
PREPROC_ARGS[30]="--bp_sys_thresh 130 --bp_dia_thresh 80"

# -------------------------------------------------
# Per-participant TRAIN baseline args (no --drop)
# -------------------------------------------------
declare -A TRAIN_ARGS
TRAIN_ARGS[17]="--train_days 4  --batch 32 --verbose --n_neighbors 5 --drop cumulative_stress_30min --no_resample"
TRAIN_ARGS[40]="--train_days 20 --batch 32 --verbose --n_neighbors 5"
TRAIN_ARGS[39]="--train_days 20 --batch 32 --verbose --drop stress_std --n_neighbors 5"
TRAIN_ARGS[36]="--train_days 20 --batch 32 --verbose --n_neighbors 5 --no_resample"
TRAIN_ARGS[35]="--train_days 20 --batch 32 --verbose --n_neighbors 3"
TRAIN_ARGS[34]="--train_days 3  --batch 32 --verbose --n_neighbors 5 --drop stress_std --no_resample"
TRAIN_ARGS[32]="--train_days 14 --batch 32 --verbose --n_neighbors 5 --drop stress_std --no_resample"
TRAIN_ARGS[25]="--train_days 20 --batch 32 --verbose --n_neighbors 5 --no_resample"
TRAIN_ARGS[16]="--train_days 20 --batch 32 --verbose --n_neighbors 5 --drop stress_std --no_resample"
TRAIN_ARGS[10]="--train_days 18 --batch 32 --verbose --n_neighbors 2"
TRAIN_ARGS[15]="--train_days 20 --batch 32 --verbose --n_neighbors 5"
TRAIN_ARGS[18]="--train_days 20 --batch 32 --verbose --drop stress_std --n_neighbors 2"
TRAIN_ARGS[20]="--train_days 17 --batch 32 --verbose --n_neighbors 5"
TRAIN_ARGS[22]="--train_days 14 --batch 32 --verbose --drop stress_std --n_neighbors 5"
TRAIN_ARGS[23]="--train_days 17 --batch 32 --verbose --drop stress_std --n_neighbors 5"
TRAIN_ARGS[24]="--train_days 20 --batch 32 --verbose --n_neighbors 5"
TRAIN_ARGS[26]="--train_days 16 --batch 32 --verbose --n_neighbors 5"
TRAIN_ARGS[33]="--train_days 20 --batch 32 --verbose --n_neighbors 5"
TRAIN_ARGS[31]="--train_days 20 --batch 32 --verbose --n_neighbors 5"
TRAIN_ARGS[30]="--train_days 20 --batch 32 --verbose --n_neighbors 5"

RESULTS_DIR="results_rm_NO_BP2"
mkdir -p "$RESULTS_DIR"
: > "$RESULTS_DIR/failed_runs.log"

SUMMARY="$RESULTS_DIR/ablation_summary.csv"
echo "participant,drop_group,AUROC" > "$SUMMARY"

run_with_fallback() {
  local pid="$1" log="$2" base_args="$3" drop_flags="$4"

  echo "▶️  train.py --participant_id $pid $base_args $drop_flags" | tee -a "$log"
  if python train.py --participant_id "$pid" --out_dir "$OUT_DIR" $base_args $drop_flags 2>&1 | tee -a "$log"; then
    return 0
  fi

  if [[ "$base_args" != *"--no_resample"* ]]; then
    echo "⚠️  First attempt failed; retry with --no_resample" | tee -a "$log"
    python train.py --participant_id "$pid" --out_dir "$OUT_DIR" $base_args --no_resample $drop_flags 2>&1 | tee -a "$log"
    return $?
  fi

  return 1
}

for PID in "${PARTICIPANTS[@]}"; do
  PRE="${PREPROC_ARGS[$PID]:-}"
  BASE="${TRAIN_ARGS[$PID]:-}"

  if [[ -z "$PRE" || -z "$BASE" ]]; then
    echo "❌ Missing config for participant $PID" | tee -a "$RESULTS_DIR/failed_runs.log"
    continue
  fi

  # Participant state dir + preprocess caching
  PSTATE_DIR="$RESULTS_DIR/hp${PID}"
  mkdir -p "$PSTATE_DIR"
  PLOG="$PSTATE_DIR/preprocess.log"
  ARGS_FILE="$PSTATE_DIR/.preprocess.args"
  PREP_DONE="$PSTATE_DIR/.preprocess.done"

  if [[ -f "$PREP_DONE" ]] && cmp -s <(echo "$PRE") "$ARGS_FILE"; then
    echo "🧹 Preprocess hp${PID}: up-to-date, skipping." | tee -a "$PLOG"
  else
    echo "$PRE" > "$ARGS_FILE"
    echo "🧹 Preprocess hp${PID}: python preprocess.py --participant_id $PID $PRE" | tee "$PLOG"
    python preprocess.py --participant_id "$PID" $PRE 2>&1 | tee -a "$PLOG"
    touch "$PREP_DONE"
  fi

  # Grouped ablations
  for GROUP in "${!FEATURE_GROUPS[@]}"; do
    DROP_FLAGS="${FEATURE_GROUPS[$GROUP]}"
    OUT_DIR="${RESULTS_DIR}/hp${PID}/drop_${GROUP}"
    LOG="${OUT_DIR}/log.txt"
    DONE="${OUT_DIR}/.done"
    INPROG="${OUT_DIR}/.inprogress"

    if [[ -f "$DONE" ]]; then
      echo "✅ Skip hp${PID} (drop_${GROUP}) — already done."
      continue
    fi

    mkdir -p "$OUT_DIR"

    if [[ -f "$INPROG" ]]; then
      echo "♻️  Found stale in-progress flag for hp${PID} (drop_${GROUP}); re-running."
      rm -f "$INPROG"
    fi
    : > "$INPROG"

    echo "🚀 Training hp${PID} with drop_${GROUP} ..." | tee "$LOG"
    echo "Baseline args: $BASE" | tee -a "$LOG"
    echo "Drop flags   : $DROP_FLAGS" | tee -a "$LOG"

    if run_with_fallback "$PID" "$LOG" "$BASE" "$DROP_FLAGS"; then
      echo "✅ Finished hp${PID} (drop_${GROUP})" | tee -a "$LOG"

      # copy/parse results
      if [[ -f "processed/hp${PID}/results.txt" ]]; then
        cp "processed/hp${PID}/results.txt" "$OUT_DIR/results.txt"
        AUROC=$(grep -i "AUROC" "$OUT_DIR/results.txt" | tail -1 | awk '{print $NF}')
      else
        AUROC=$(grep -i "AUROC" "$LOG" | tail -1 | awk '{print $NF}')
      fi
      : "${AUROC:=NA}"
      echo "hp${PID},${GROUP},${AUROC}" >> "$SUMMARY"

      mv "$INPROG" "$DONE"
    else
      echo "❌ Training failed for hp${PID} (drop_${GROUP})" | tee -a "$LOG"
      echo "hp${PID},drop_${GROUP}" >> "$RESULTS_DIR/failed_runs.log"
      rm -f "$INPROG"
    fi
  done
done

echo "🎯 Done. Summary at: $SUMMARY"
