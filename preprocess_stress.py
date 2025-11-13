# ---------- preprocess.py ----------
#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Preprocess data for STRESS prediction (inputs include BP values/spikes, HR, steps)")
    parser.add_argument('--participant_id', required=True, help="Participant ID, e.g., 31")
    parser.add_argument('--roll_windows', default='5,10,30,60', help="Comma-separated rolling window sizes in minutes")
    parser.add_argument('--lag_horizons', default='1,3,5', help="Comma-separated lag horizons in rows")
    parser.add_argument('--agg_lengths', default='3,5', help="Two aggregate lengths (small, large) in rows")
    parser.add_argument('--bp_sys_thresh', type=int, default=130, help="Systolic threshold (default 130)")
    parser.add_argument('--bp_dia_thresh', type=int, default=80, help="Diastolic threshold (default 80)")
    parser.add_argument('--bp_window_min', type=int, default=15, help="± window (minutes) around stress to aggregate BP stats")
    parser.add_argument('--stress_thresh', type=float, default=6.0, help="Threshold for stress_high (binary target)")
    parser.add_argument('--work_hours', default='9,17', help="Work-hour start,end in 24h format")
    parser.add_argument('--weekend_day', type=int, default=5, help="Integer for weekend start (0=Mon…6=Sun)")
    args = parser.parse_args()

    pid = args.participant_id
    base = os.path.join('hp', f'hp{pid}')
    file_hr     = os.path.join(base, f"hp{pid}_hr.csv")
    file_steps  = os.path.join(base, f"hp{pid}_steps.csv")
    file_bp     = os.path.join(base, f"blood_pressure_readings_ID{pid}_cleaned.csv")
    file_stress = os.path.join(base, f"questionnaire_responses_ID{pid}.csv")

    # Step 1: Load
    df_hr     = pd.read_csv(file_hr)
    df_steps  = pd.read_csv(file_steps)
    df_bp     = pd.read_csv(file_bp)
    df_stress = pd.read_csv(file_stress)

    # Step 2: Timestamps
    df_hr['time']                = pd.to_datetime(df_hr['time']).dt.tz_localize(None)
    df_steps['time']             = pd.to_datetime(df_steps['time']).dt.tz_localize(None)
    df_bp['datetime_local']      = pd.to_datetime(df_bp['datetime_local']).dt.tz_localize(None)
    df_stress['local_created_at']= pd.to_datetime(df_stress['local_created_at']).dt.tz_localize(None)

    # Step 3: Auto-drop residual first day (same logic)
    first_bp = df_bp['datetime_local'].dt.date.min()
    first_hr = df_hr['time'].dt.date.min()
    cutoff   = min(first_bp, first_hr)
    to_remove_bp     = df_bp[df_bp['datetime_local'].dt.date == cutoff]
    to_remove_stress = df_stress[df_stress['local_created_at'].dt.date == cutoff]
    df_bp     = df_bp[df_bp['datetime_local'].dt.date > cutoff]
    df_stress = df_stress[df_stress['local_created_at'].dt.date > cutoff]
    print(f"Removed {len(to_remove_bp)} record(s) on {cutoff} from {file_bp}")
    print(f"Removed {len(to_remove_stress)} record(s) on {cutoff} from {file_stress}")

    # Step 4: Sort
    df_hr     = df_hr.sort_values('time')
    df_steps  = df_steps.sort_values('time')
    df_bp     = df_bp.sort_values('datetime_local')
    df_stress = df_stress.sort_values('local_created_at')

    # Step 5: BP spike flag on BP table
    df_bp['BP_spike'] = ((df_bp['systolic'] > args.bp_sys_thresh) |
                         (df_bp['diastolic'] > args.bp_dia_thresh)).astype(int)

    # Step 6: Merge HR & Steps and rollups
    df_bio = pd.merge_asof(df_hr, df_steps, on='time', direction='backward',
                           suffixes=('_hr','_steps')).set_index('time')

    windows = [int(x) for x in args.roll_windows.split(',')]
    for w in windows:
        s = f"{w}min"
        df_bio[f'hr_mean_{s}']     = df_bio['value_hr'].rolling(s).mean()
        df_bio[f'hr_min_{s}']      = df_bio['value_hr'].rolling(s).min()
        df_bio[f'hr_max_{s}']      = df_bio['value_hr'].rolling(s).max()
        df_bio[f'hr_std_{s}']      = df_bio['value_hr'].rolling(s).std()
        df_bio[f'steps_total_{s}'] = df_bio['value_steps'].rolling(s).sum()
        df_bio[f'steps_mean_{s}']  = df_bio['value_steps'].rolling(s).mean()
        df_bio[f'steps_min_{s}']   = df_bio['value_steps'].rolling(s).min()
        df_bio[f'steps_max_{s}']   = df_bio['value_steps'].rolling(s).max()
        df_bio[f'steps_std_{s}']   = df_bio['value_steps'].rolling(s).std()
        df_bio[f'steps_diff_{s}']  = df_bio[f'steps_max_{s}'] - df_bio[f'steps_min_{s}']
    df_bio = df_bio.reset_index()

    # ---------- KEY CHANGE ----------
    # Step 7: Anchor on STRESS times (target), not BP times
    df_merged = pd.merge_asof(
        df_stress[['local_created_at','stressLevel_value']].sort_values('local_created_at'),
        df_bio.sort_values('time'),
        left_on='local_created_at',
        right_on='time',
        direction='backward'
    )

    # Step 8: BP features in ±window around each stress record
    win = args.bp_window_min
    def bp_stats_around(ts):
        lo, hi = ts - pd.Timedelta(minutes=win), ts + pd.Timedelta(minutes=win)
        sub = df_bp[(df_bp['datetime_local'] >= lo) & (df_bp['datetime_local'] <= hi)]
        return pd.Series({
            'sbp_mean': sub['systolic'].mean(),
            'sbp_min':  sub['systolic'].min(),
            'sbp_max':  sub['systolic'].max(),
            'sbp_std':  sub['systolic'].std(),
            'dbp_mean': sub['diastolic'].mean(),
            'dbp_min':  sub['diastolic'].min(),
            'dbp_max':  sub['diastolic'].max(),
            'dbp_std':  sub['diastolic'].std(),
            'bp_spike_any': int((sub['BP_spike'] == 1).any()) if len(sub) else 0
        })
    bp_feats = df_merged['local_created_at'].apply(bp_stats_around)
    df_merged = pd.concat([df_merged, bp_feats], axis=1)

    # Step 9: Time since last BP reading and last spike (via merge_asof)
    tmp_bp   = df_bp[['datetime_local']].rename(columns={'datetime_local':'bp_time'}).sort_values('bp_time')
    tmp_spk  = df_bp[df_bp['BP_spike']==1][['datetime_local']].rename(columns={'datetime_local':'spike_time'}).sort_values('spike_time')

    df_merged = pd.merge_asof(df_merged.sort_values('local_created_at'), tmp_bp, left_on='local_created_at', right_on='bp_time', direction='backward')
    df_merged['time_since_last_bp'] = (df_merged['local_created_at'] - df_merged['bp_time']).dt.total_seconds()/60

    df_merged = pd.merge_asof(df_merged.sort_values('local_created_at'), tmp_spk, left_on='local_created_at', right_on='spike_time', direction='backward')
    df_merged['time_since_last_BP_spike'] = (df_merged['local_created_at'] - df_merged['spike_time']).dt.total_seconds()/60

    # Step 10: Additional engineered features (NO stress-leakage)
    # Lags (row-wise, along stress timeline)
    lag_list = [int(x) for x in args.lag_horizons.split(',')]
    lag_feats = ['bp_spike_any','hr_mean_5min','steps_total_10min','sbp_mean','dbp_mean']
    for lf in lag_feats:
        for lag in lag_list:
            df_merged[f'{lf}_lag_{lag}'] = df_merged[lf].shift(lag)

    df_merged['hr_steps_ratio']             = df_merged['hr_mean_5min'] / (df_merged['steps_total_10min'] + 1)
    df_merged['steps_hr_variability_ratio'] = df_merged['steps_std_10min'] / (df_merged['hr_std_10min'] + 1e-5)

    df_merged['hr_mean_rolling_3']      = df_merged['hr_mean_5min'].rolling(3).mean()
    df_merged['steps_total_rolling_5']  = df_merged['steps_total_10min'].rolling(5).mean()
    df_merged['hr_std_rolling_3']       = df_merged['hr_std_10min'].rolling(3).std()

    # Calendar features from stress timestamp
    df_merged['hour_of_day']      = df_merged['local_created_at'].dt.hour
    df_merged['day_of_week']      = df_merged['local_created_at'].dt.dayofweek
    df_merged['is_working_hours'] = df_merged['hour_of_day'].between(*map(int,args.work_hours.split(','))).astype(int)
    df_merged['is_weekend']       = (df_merged['day_of_week'] >= args.weekend_day).astype(int)

    # Targets
    df_merged['stress_value'] = pd.to_numeric(df_merged['stressLevel_value'], errors='coerce')
    df_merged['stress_high']  = (df_merged['stress_value'] >= args.stress_thresh).astype(int)

    # Step 11: Handle missing
    df_merged.ffill(inplace=True)
    df_merged.bfill(inplace=True)

    # Step 12: Save
    out_dir = os.path.join('processed_stress', f'hp{pid}')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'processed_stress_prediction_data.csv')
    df_merged.to_csv(out_file, index=False)
    print(f"✅ Saved processed data to {out_file}")
    print("Heads-up: target is stress_high (binary). For regression, use stress_value.")

if __name__ == '__main__':
    main()
