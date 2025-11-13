# ============================================
# Complete BP Spike Prediction Pipeline Makefile
# ============================================

.PHONY: all clean preprocess-all train-all global temporal stress ablation

# Participant list
PIDS := 17 40 39 36 35 34 32 25 16 10 15 18 20 22 23 24 26 33 31 30

# Feature groups for ablation
FEATURE_GROUPS := hr_features step_features stress_features bp_lags ratios cumulative time_features

# Main target - runs everything
all: train-all global ablation temporal stress
	@echo "✅ All analyses completed successfully!"

# ─────────────────────────────────────────────
# 1) Individual participant preprocessing
# ─────────────────────────────────────────────
.PHONY: preprocess-all $(patsubst %,preprocess-%,$(PIDS))

preprocess-all: $(patsubst %,preprocess-%,$(PIDS))
	@echo "✅ All participants preprocessed"

# Define preprocessing arguments
PREPROC_ARGS_17 := --bp_sys_thresh 146 --bp_dia_thresh 100
PREPROC_ARGS_40 := --bp_sys_thresh 130 --bp_dia_thresh 80
PREPROC_ARGS_39 := --bp_sys_thresh 130 --bp_dia_thresh 80
PREPROC_ARGS_36 := --bp_sys_thresh 101 --bp_dia_thresh 54
PREPROC_ARGS_35 := --bp_sys_thresh 130 --bp_dia_thresh 80
PREPROC_ARGS_34 := --bp_sys_thresh 156 --bp_dia_thresh 83
PREPROC_ARGS_32 := --bp_sys_thresh 143 --bp_dia_thresh 83
PREPROC_ARGS_25 := --bp_sys_thresh 130 --bp_dia_thresh 80
PREPROC_ARGS_16 := --bp_sys_thresh 130 --bp_dia_thresh 90
PREPROC_ARGS_10 := --bp_sys_thresh 100 --bp_dia_thresh 60
PREPROC_ARGS_15 := --bp_sys_thresh 130 --bp_dia_thresh 80
PREPROC_ARGS_18 := --bp_sys_thresh 130 --bp_dia_thresh 80
PREPROC_ARGS_20 := --bp_sys_thresh 130 --bp_dia_thresh 80
PREPROC_ARGS_22 := --bp_sys_thresh 120 --bp_dia_thresh 80
PREPROC_ARGS_23 := --bp_sys_thresh 130 --bp_dia_thresh 80
PREPROC_ARGS_24 := --bp_sys_thresh 130 --bp_dia_thresh 80
PREPROC_ARGS_26 := --bp_sys_thresh 135 --bp_dia_thresh 85
PREPROC_ARGS_33 := --bp_sys_thresh 130 --bp_dia_thresh 80
PREPROC_ARGS_31 := --bp_sys_thresh 130 --bp_dia_thresh 80
PREPROC_ARGS_30 := --bp_sys_thresh 130 --bp_dia_thresh 80

# Preprocessing with cache validation
define PREPROCESS_RULE
preprocess-$(1):
	@mkdir -p results_rm/hp$(1)
	@if [ -f "results_rm/hp$(1)/.preprocess.done" ] && \
	   [ -f "results_rm/hp$(1)/.preprocess.args" ] && \
	   [ "$$$$(cat results_rm/hp$(1)/.preprocess.args 2>/dev/null)" = "$$(PREPROC_ARGS_$(1))" ]; then \
		echo "🧹 Preprocess hp$(1): up-to-date, skipping."; \
	else \
		echo "$$(PREPROC_ARGS_$(1))" > results_rm/hp$(1)/.preprocess.args; \
		echo "🧹 Preprocess hp$(1): python preprocess.py --participant_id $(1) $$(PREPROC_ARGS_$(1))"; \
		python preprocess.py --participant_id $(1) $$(PREPROC_ARGS_$(1)) 2>&1 | tee results_rm/hp$(1)/preprocess.log; \
		touch results_rm/hp$(1)/.preprocess.done; \
	fi
endef

# Generate all preprocessing rules
$(foreach p,$(PIDS),$(eval $(call PREPROCESS_RULE,$(p))))

# ─────────────────────────────────────────────
# 2) Individual participant training (baseline)
# ─────────────────────────────────────────────
.PHONY: train-all $(patsubst %,train-%,$(PIDS))

train-all: $(patsubst %,train-%,$(PIDS))
	@echo "✅ All participants trained"

train-17: preprocess-17
	python train.py --participant_id 17 \
		--train_days 4 \
		--batch 32 \
		--verbose \
		--drop cumulative_stress_30min \
		--no_resample

train-40: preprocess-40
	python train.py --participant_id 40 \
		--train_days 20 \
		--batch 32 \
		--verbose

train-39: preprocess-39
	python train.py --participant_id 39 \
		--train_days 20 \
		--batch 32 \
		--verbose \
		--drop stress_std

train-36: preprocess-36
	python train.py --participant_id 36 \
		--train_days 20 \
		--batch 32 \
		--verbose \
		--no_resample

train-35: preprocess-35
	python train.py --participant_id 35 \
		--train_days 33 \
		--batch 32 \
		--verbose \
		--n_neighbors 3

train-34: preprocess-34
	python train.py --participant_id 34 \
		--train_days 3 \
		--batch 32 \
		--verbose \
		--drop stress_std \
		--no_resample

train-32: preprocess-32
	python train.py --participant_id 32 \
		--train_days 14 \
		--batch 32 \
		--verbose \
		--drop stress_std \
		--no_resample

train-25: preprocess-25
	python train.py --participant_id 25 \
		--train_days 20 \
		--batch 32 \
		--verbose \
		--no_resample

train-16: preprocess-16
	python train.py --participant_id 16 \
		--train_days 20 \
		--batch 32 \
		--verbose \
		--drop stress_std \
		--no_resample

train-10: preprocess-10
	python train.py --participant_id 10 \
		--train_days 18 \
		--batch 32 \
		--verbose \
		--n_neighbors 2

train-15: preprocess-15
	python train.py --participant_id 15 \
		--train_days 20 \
		--batch 32 \
		--verbose

train-18: preprocess-18
	python train.py --participant_id 18 \
		--train_days 20 \
		--batch 32 \
		--verbose \
		--drop stress_std \
		--n_neighbors 2

train-20: preprocess-20
	python train.py --participant_id 20 \
		--train_days 17 \
		--batch 32 \
		--verbose

train-22: preprocess-22
	python train.py --participant_id 22 \
		--train_days 14 \
		--batch 32 \
		--verbose \
		--drop stress_std

train-23: preprocess-23
	python train.py --participant_id 23 \
		--train_days 17 \
		--batch 32 \
		--verbose \
		--drop stress_std

train-24: preprocess-24
	python train.py --participant_id 24 \
		--train_days 20 \
		--batch 32 \
		--verbose

train-26: preprocess-26
	python train.py --participant_id 26 \
		--train_days 16 \
		--batch 32 \
		--verbose

train-33: preprocess-33
	python train.py --participant_id 33 \
		--train_days 20 \
		--batch 32 \
		--verbose

train-31: preprocess-31
	python train.py --participant_id 31 \
		--train_days 20 \
		--batch 32 \
		--verbose

train-30: preprocess-30
	python train.py --participant_id 30 \
		--train_days 20 \
		--batch 32 \
		--verbose

# ─────────────────────────────────────────────
# 3) Global ensemble training
# ─────────────────────────────────────────────
global: train-all
	@echo "🔹 Training global ensemble model…"
	python global_train.py \
		--settings participant_settings.json \
		--batch 32 \
		--lstm_trials 20 \
		--verbose
	python analysis_figures.py

# ─────────────────────────────────────────────
# 4) Feature Ablation Study
# ─────────────────────────────────────────────
.PHONY: ablation $(foreach p,$(PIDS),$(foreach g,$(FEATURE_GROUPS),ablation-$(p)-$(g)))

# Define drop flags for each feature group
DROP_FLAGS_hr_features := hr_mean_5min,hr_steps_ratio,steps_hr_variability_ratio,stress_weighted_hr,hr_mean_10min,hr_mean_30min,hr_mean_60min,hr_mean_5min_lag_1,hr_mean_5min_lag_3,hr_mean_5min_lag_5,hr_mean_rolling_3,hr_std_5min,hr_std_10min,hr_std_30min,hr_std_60min,hr_std_rolling_3
DROP_FLAGS_step_features := steps_total_5min,steps_total_10min,steps_hr_variability_ratio,hr_steps_ratio,stress_steps_ratio,steps_total_30min,steps_total_60min,steps_total_10min_lag_1,steps_total_10min_lag_3,steps_total_10min_lag_5,steps_total_rolling_5,steps_mean_5min,steps_mean_10min,steps_mean_30min,steps_mean_60min,steps_min_5min,steps_max_5min,steps_std_5min,steps_diff_5min,steps_min_10min,steps_max_10min,steps_std_10min,steps_diff_10min,steps_min_30min,steps_max_30min,steps_std_30min,steps_diff_30min,steps_min_60min,steps_max_60min,steps_std_60min,steps_diff_60min
DROP_FLAGS_stress_features := stress_steps_ratio,stress_mean,stress_min,stress_weighted_hr,stress_max,stress_std,stress_mean_lag_1,stress_mean_lag_3,stress_mean_lag_5
DROP_FLAGS_bp_lags := BP_spike_lag_1,BP_spike_lag_3,BP_spike_lag_5,time_since_last_BP_spike
DROP_FLAGS_ratios := hr_steps_ratio,stress_weighted_hr,stress_steps_ratio,steps_hr_variability_ratio
DROP_FLAGS_cumulative := cumulative_stress_30min,cumulative_steps_30min
DROP_FLAGS_time_features := hour_of_day,day_of_week,is_working_hours,is_weekend

# Define baseline training arguments (from shell script)
TRAIN_ARGS_17 := --train_days 4 --batch 32 --verbose --n_neighbors 5 --no_resample
TRAIN_ARGS_40 := --train_days 20 --batch 32 --verbose --n_neighbors 5
TRAIN_ARGS_39 := --train_days 20 --batch 32 --verbose --n_neighbors 5
TRAIN_ARGS_36 := --train_days 20 --batch 32 --verbose --n_neighbors 5 --no_resample
TRAIN_ARGS_35 := --train_days 33 --batch 32 --verbose --n_neighbors 3
TRAIN_ARGS_34 := --train_days 3 --batch 32 --verbose --n_neighbors 5 --no_resample
TRAIN_ARGS_32 := --train_days 14 --batch 32 --verbose --n_neighbors 5 --no_resample
TRAIN_ARGS_25 := --train_days 20 --batch 32 --verbose --n_neighbors 5 --no_resample
TRAIN_ARGS_16 := --train_days 20 --batch 32 --verbose --n_neighbors 5 --no_resample
TRAIN_ARGS_10 := --train_days 18 --batch 32 --verbose --n_neighbors 2
TRAIN_ARGS_15 := --train_days 20 --batch 32 --verbose --n_neighbors 5
TRAIN_ARGS_18 := --train_days 20 --batch 32 --verbose --n_neighbors 2
TRAIN_ARGS_20 := --train_days 17 --batch 32 --verbose --n_neighbors 5
TRAIN_ARGS_22 := --train_days 14 --batch 32 --verbose --n_neighbors 5
TRAIN_ARGS_23 := --train_days 17 --batch 32 --verbose --n_neighbors 5
TRAIN_ARGS_24 := --train_days 20 --batch 32 --verbose --n_neighbors 5
TRAIN_ARGS_26 := --train_days 16 --batch 32 --verbose --n_neighbors 5
TRAIN_ARGS_33 := --train_days 20 --batch 32 --verbose --n_neighbors 5
TRAIN_ARGS_31 := --train_days 20 --batch 32 --verbose --n_neighbors 5
TRAIN_ARGS_30 := --train_days 20 --batch 32 --verbose --n_neighbors 5

# Main ablation target
ablation: $(foreach p,$(PIDS),$(foreach g,$(FEATURE_GROUPS),ablation-$(p)-$(g)))
	@echo "✅ Feature ablation study completed"
	@echo "📊 Creating ablation summary..."
	@python -c "import os; os.makedirs('results_rm', exist_ok=True)"
	@echo "participant,drop_group,AUROC" > results_rm/ablation_summary.csv
	@for pid in $(PIDS); do \
		for group in $(FEATURE_GROUPS); do \
			if [ -f "results_rm/hp$$pid/drop_$$group/results.txt" ]; then \
				auroc=$$(grep -i "AUROC" "results_rm/hp$$pid/drop_$$group/results.txt" | tail -1 | awk '{print $$NF}' || echo "NA"); \
				echo "hp$$pid,$$group,$$auroc" >> results_rm/ablation_summary.csv; \
			fi \
		done \
	done
	@echo "🎯 Done. Summary at: results_rm/ablation_summary.csv"

# Generate ablation targets for each participant and feature group
define ABLATION_RULE
ablation-$(1)-$(2): preprocess-$(1)
	@mkdir -p results_rm/hp$(1)/drop_$(2)
	@if [ -f "results_rm/hp$(1)/drop_$(2)/.done" ]; then \
		echo "✅ Skip hp$(1) (drop_$(2)) — already done."; \
	else \
		if [ -f "results_rm/hp$(1)/drop_$(2)/.inprogress" ]; then \
			echo "♻️  Found stale in-progress flag for hp$(1) (drop_$(2)); re-running."; \
			rm -f "results_rm/hp$(1)/drop_$(2)/.inprogress"; \
		fi; \
		touch "results_rm/hp$(1)/drop_$(2)/.inprogress"; \
		echo "🚀 Training hp$(1) with drop_$(2) ..." | tee results_rm/hp$(1)/drop_$(2)/log.txt; \
		echo "Baseline args: $$(TRAIN_ARGS_$(1))" | tee -a results_rm/hp$(1)/drop_$(2)/log.txt; \
		echo "Drop flags   : $$(DROP_FLAGS_$(2))" | tee -a results_rm/hp$(1)/drop_$(2)/log.txt; \
		echo "▶️  train.py --participant_id $(1) $$(TRAIN_ARGS_$(1)) --drop $$(DROP_FLAGS_$(2))" | tee -a results_rm/hp$(1)/drop_$(2)/log.txt; \
		if python train.py --participant_id $(1) $$(TRAIN_ARGS_$(1)) --drop $$(DROP_FLAGS_$(2)) 2>&1 | tee -a results_rm/hp$(1)/drop_$(2)/log.txt; then \
			echo "✅ Finished hp$(1) (drop_$(2))" | tee -a results_rm/hp$(1)/drop_$(2)/log.txt; \
			if [ -f "processed/hp$(1)/results.txt" ]; then \
				cp "processed/hp$(1)/results.txt" "results_rm/hp$(1)/drop_$(2)/results.txt"; \
			fi; \
			mv "results_rm/hp$(1)/drop_$(2)/.inprogress" "results_rm/hp$(1)/drop_$(2)/.done"; \
		else \
			if [[ "$$(TRAIN_ARGS_$(1))" != *"--no_resample"* ]]; then \
				echo "⚠️  First attempt failed; retry with --no_resample" | tee -a results_rm/hp$(1)/drop_$(2)/log.txt; \
				if python train.py --participant_id $(1) $$(TRAIN_ARGS_$(1)) --no_resample --drop $$(DROP_FLAGS_$(2)) 2>&1 | tee -a results_rm/hp$(1)/drop_$(2)/log.txt; then \
					echo "✅ Finished hp$(1) (drop_$(2)) with --no_resample" | tee -a results_rm/hp$(1)/drop_$(2)/log.txt; \
					if [ -f "processed/hp$(1)/results.txt" ]; then \
						cp "processed/hp$(1)/results.txt" "results_rm/hp$(1)/drop_$(2)/results.txt"; \
					fi; \
					mv "results_rm/hp$(1)/drop_$(2)/.inprogress" "results_rm/hp$(1)/drop_$(2)/.done"; \
				else \
					echo "❌ Training failed for hp$(1) (drop_$(2))" | tee -a results_rm/hp$(1)/drop_$(2)/log.txt; \
					echo "hp$(1),drop_$(2)" >> results_rm/failed_runs.log; \
					rm -f "results_rm/hp$(1)/drop_$(2)/.inprogress"; \
				fi; \
			else \
				echo "❌ Training failed for hp$(1) (drop_$(2))" | tee -a results_rm/hp$(1)/drop_$(2)/log.txt; \
				echo "hp$(1),drop_$(2)" >> results_rm/failed_runs.log; \
				rm -f "results_rm/hp$(1)/drop_$(2)/.inprogress"; \
			fi; \
		fi; \
	fi
endef

# Generate all ablation rules
$(foreach p,$(PIDS),$(foreach g,$(FEATURE_GROUPS),$(eval $(call ABLATION_RULE,$(p),$(g)))))

# ─────────────────────────────────────────────
# 5) Temporal stability analysis
# ─────────────────────────────────────────────
temporal: train-all
	@echo "🔹 Running temporal stability analysis…"
	python temporal_analysis.py

# ─────────────────────────────────────────────
# 6) Clean up
# ─────────────────────────────────────────────
clean:
	@echo "🧹 Cleaning up generated files..."
	rm -rf results/
	rm -rf results_rm/
	rm -rf processed/
	rm -rf lstm_tuner/
	rm -f participant_settings.json
	rm -f all_participants_results.txt
	@echo "✅ Cleanup complete"

# ─────────────────────────────────────────────
# 7) Help
# ─────────────────────────────────────────────
help:
	@echo "BP Spike Prediction Pipeline"
	@echo "==========================="
	@echo ""
	@echo "Available targets:"
	@echo "  make all              - Run complete pipeline (train, global, ablation, temporal, stress)"
	@echo "  make train-all        - Train all participants with baseline settings"
	@echo "  make global           - Train global ensemble model"
	@echo "  make ablation         - Run grouped feature ablation study"
	@echo "  make temporal         - Run temporal stability analysis"
	@echo "  make stress           - Generate stress-related analysis"
	@echo "  make clean            - Remove all generated files"
	@echo ""
	@echo "Features:"
	@echo "  - Cached preprocessing with argument validation"
	@echo "  - Automatic --no_resample fallback on training failure"
	@echo "  - In-progress tracking for interrupted runs"
	@echo "  - Failed runs logging"
	@echo "  - Results copying from processed/ to results_rm/"
	@echo "  - Detailed logging with tee"