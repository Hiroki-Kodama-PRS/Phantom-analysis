import nbformat
import os

notebook_path = "/Users/hirokikodama/Documents/GitHub/Phantom-analysis/SciRep/Analysis for SciRep.ipynb"

# New content for the Amplitude/Area Features cell
new_cell_source = r"""import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, detrend, butter, filtfilt
from sklearn.metrics import roc_auc_score, roc_curve
import math
from scipy.fft import rfftfreq, rfft 

# ========= 基本設定 =========
fs = 2000  # Hz
depths = [3, 9, 15, 21]
state_map = {1: "Normal", 2: "Ischaemia", 3: "Congestion"}
channels = {"ppgA_Red": "Red", "ppgA_IR": "IR"} 
DEBUG_MODE = True
DEBUG_FILE = "debug_log_part1_to_3_amplitude_features.txt"
with open(DEBUG_FILE, "w") as f:
    f.write("--- Debug Log Start (Amplitude & Area Features) ---\n")

def DEBUG_LOG(message):
    if DEBUG_MODE:
        with open(DEBUG_FILE, "a") as f:
            f.write(message + "\n")

# --- BPF, BPM推定関数 (変更なし) ---
def bandpass_filter(sig, fs, low=0.5, high=5.0, order=3):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)

def estimate_bpm(signal, fs=2000):
    sig = pd.Series(signal, dtype=float).interpolate().fillna(0).values
    sig = detrend(sig)
    sig = bandpass_filter(sig, fs)
    prom = max(np.std(sig) * 0.3, np.ptp(sig) * 0.05)
    peaks, _ = find_peaks(sig, prominence=prom, distance=int(fs * 0.5))
    if len(peaks) < 2:
        return np.nan
    duration_sec = len(sig) / fs
    bpm = len(peaks) / (duration_sec / 60.0)
    return bpm

# --- ★★★ 1拍ごとの特徴抽出 (面積特徴量+DC/PI計算を追加) ★★★
def extract_pulse_features(signal, raw_signal, fs, dist=0.3):
    # ロバスト性向上のためのピーク検出閾値 (信号振幅の10%以上の高さを要求)
    threshold_height = np.min(signal) + (np.ptp(signal) * 0.1)
    peaks, _ = find_peaks(signal, height=threshold_height, distance=int(fs * dist))
    
    feats = []
    if len(peaks) < 2: return []
        
    for p in peaks:
        left = max(0, p - int(0.5 * fs))
        right = min(len(signal), p + int(0.5 * fs))
        seg = signal[left:right]
        if len(seg) < 5: continue
        if np.any(np.isnan(seg)) or np.all(seg == seg[0]): continue
        
        peak_idx = np.argmax(seg)
        trough_idx = np.argmin(seg[:peak_idx+1])
        next_trough_idx = peak_idx + np.argmin(seg[peak_idx:])
        
        if trough_idx < peak_idx < next_trough_idx:
            
            # --- 共通の基線と振幅の計算 (Processed) ---
            baseline = seg[trough_idx]
            peak_val = seg[peak_idx]
            amp = peak_val - baseline
            
            # Pulse Area (全面積)
            full_seg = seg[trough_idx:next_trough_idx+1]
            area_pulse = np.trapz(full_seg - baseline, dx=1/fs) 
            
            # --- 面積特徴量の計算 (S-AUC, D-AUC) ---
            # S-AUC: 収縮期面積 (トラフからピークまで)
            sys_seg = seg[trough_idx:peak_idx+1]
            area_systolic = np.trapz(sys_seg - baseline, dx=1/fs)
            
            # D-AUC: 拡張期面積 (ピークから次のトラフまで)
            dias_seg = seg[peak_idx:next_trough_idx+1]
            area_diastolic = np.trapz(dias_seg - baseline, dx=1/fs)
            
            # --- ★★★ DC / PI Calculation (from Raw) ★★★ ---
            dc_val = np.nan
            pi_val = np.nan
            if raw_signal is not None:
                # Global indices mapping
                global_start = left + trough_idx
                global_end = left + next_trough_idx
                
                if global_end < len(raw_signal):
                    raw_seg_pulse = raw_signal[global_start : global_end + 1]
                    if len(raw_seg_pulse) > 0:
                        dc_val = np.mean(raw_seg_pulse)
                        if abs(dc_val) > 1e-6:
                            pi_val = amp / dc_val # PI = AC(Processed) / DC(Raw)
            
            feats.append((area_pulse, area_systolic, area_diastolic, amp, dc_val, pi_val))
    return feats

# ========= Part 1 & 2: データロードと特徴抽出 =========
records = []
pulse_id_counter = 0 

for depth in depths:
    for i_state, state_name in state_map.items():
        file_path = Path(f"{depth}mm_data{i_state}.csv")
        DEBUG_LOG(f"\\n--- Processing: {file_path.name} ({state_name}) ---")

        if not file_path.exists(): continue
        df = pd.read_csv(file_path)

        # --- BPM推定 ---
        ir_sig = df.get("ppgA_IR", pd.Series([]).values)
        if len(ir_sig) == 0: continue
            
        bpm = estimate_bpm(ir_sig, fs)
        window_sec = 60.0 / bpm if not np.isnan(bpm) and bpm > 0 else 1.0
        window_sec = float(np.clip(window_sec, 0.6, 2.0))
        
        step = int(round(fs * window_sec))
        current_pulse_id = pulse_id_counter 

        for col, ch_label in channels.items():
            if col not in df.columns: continue
            sig = df[col].values
            
            # ★★★ Raw Signal Loading for DC ★★★
            raw_col = col + "_raw"
            if raw_col in df.columns:
                raw_sig = df[raw_col].values
            else:
                # Fallback: if raw not found, use sig (DC=0 likely, but prevents crash)
                if pulse_id_counter == 0: DEBUG_LOG(f"Warning: {raw_col} not found used {col} for DC")
                raw_sig = sig

            # (1) Window-based Amplitude (Amplitudeのサンプリングを増やす目的で維持)
            n_win = len(sig) // step
            if pulse_id_counter == 0: DEBUG_LOG(f"  Channel: {ch_label}, Total Windows: {n_win}")
            
            for w in range(n_win):
                s, e = w * step, (w + 1) * step
                seg = sig[s:e]
                if len(seg) < 5: continue
                amp = seg.max() - seg.min()
                
                # Windowed DC/PI
                raw_seg = raw_sig[s:e]
                dc_val = np.mean(raw_seg)
                pi_val = amp / dc_val if abs(dc_val) > 1e-6 else np.nan
                
                records.append({
                    "Depth": depth, "State": state_name, "Channel": ch_label,
                    "Pulse_ID": current_pulse_id,
                    "Amplitude": amp,
                    "DC": dc_val, "PI": pi_val,
                    "Pulse_Area": np.nan, "S_AUC": np.nan, "D_AUC": np.nan
                })

            # (2) peaksで詳細特徴抽出 (Peak-based)
            feats = extract_pulse_features(sig, raw_sig, fs)
            if pulse_id_counter == 0: DEBUG_LOG(f"  Channel: {ch_label}, Extracted Pulse Features: {len(feats)} pulses.")
            
            for area_pulse, area_systolic, area_diastolic, amp, dc_val, pi_val in feats:
                # 振幅と面積特徴量のみを記録
                records.append({
                    "Depth": depth, "State": state_name, "Channel": ch_label,
                    "Pulse_ID": current_pulse_id,
                    "Amplitude": amp,
                    "DC": dc_val, "PI": pi_val,
                    "Pulse_Area": area_pulse,
                    "S_AUC": area_systolic,
                    "D_AUC": area_diastolic,
                })
        
        pulse_id_counter += 1 

df_feat = pd.DataFrame(records)
if df_feat.empty:
    raise RuntimeError("Error: No data loaded. Please check CSV files.")
DEBUG_LOG(f"\\n--- Part 2 Complete: Total Feature Rows: {len(df_feat)} ---")


# ========= Part 3: 複合指標と正規化 (振幅・面積特徴に特化) =========
# Area Under Curve Ratio (Ratio between systolic and diastolic areas)
df_feat["AUC_Ratio"] = df_feat["S_AUC"] / (df_feat["D_AUC"] + 1e-6)


# --- 正規化（Redは3mm Red Normal基準、IRは3mm IR Normal基準） ---
# Amplitudeの正規化
def normalize_channel_amp(df, ch):
    base_data = df.query("Channel==@ch and Depth==3 and State=='Normal'")["Amplitude"]
    base = base_data.mean()
    DEBUG_LOG(f"Normalization Base (Amplitude, {ch}): Mean={base:.4e}, Count={len(base_data)}")
    
    if pd.isna(base) or base == 0:
        df.loc[df.Channel == ch, "Amp_norm"] = np.nan
    else:
        df.loc[df.Channel == ch, "Amp_norm"] = df.loc[df.Channel == ch, "Amplitude"] / base

normalize_channel_amp(df_feat, "Red")
normalize_channel_amp(df_feat, "IR")

# 面積特徴量の正規化
area_features = ["Pulse_Area", "S_AUC", "D_AUC"]

for f in area_features:
    def normalize_channel_area(df, ch, feature):
        base_data = df.query("Channel==@ch and Depth==3 and State=='Normal'")[feature]
        base = base_data.mean()
        
        DEBUG_LOG(f"Normalization Base ({feature}, {ch}): Mean={base:.4e}, Count={len(base_data)}")
        
        if pd.isna(base) or base == 0:
            df.loc[df.Channel == ch, f"{feature}_norm"] = np.nan
        else:
            df.loc[df.Channel == ch, f"{feature}_norm"] = df.loc[df.Channel == ch, feature] / base
            
    normalize_channel_area(df_feat, "Red", f)
    normalize_channel_area(df_feat, "IR", f)


# --- Red/IR比 ---
# Amplitudeと面積の正規化された比率を計算 (Red/IR_Ratio)

amp_data = df_feat.dropna(subset=['Amplitude', 'Amp_norm']).copy()

# 必要なカラムを選択 (DC, PI, Raw Amplitudeを追加)
cols_to_merge = ['Depth', 'State', 'Pulse_ID', 'Amp_norm', 'Amplitude', 'DC', 'PI'] + [f"{f}_norm" for f in area_features]

df_red = amp_data[amp_data.Channel == "Red"][cols_to_merge].rename(
    columns=lambda x: x + "_Red" if x not in ['Depth', 'State', 'Pulse_ID'] else x
)

df_ir = amp_data[amp_data.Channel == "IR"][cols_to_merge].rename(
    columns=lambda x: x + "_IR" if x not in ['Depth', 'State', 'Pulse_ID'] else x
)

df_ratio = pd.merge(df_red, df_ir, on=['Depth','State','Pulse_ID'], how='inner')
DEBUG_LOG(f"Ratio Data Rows (Merged for Amplitude/Area): {len(df_ratio)}")

# Red/IR Ratio 特徴量の計算
df_ratio["Red_IR_amp_norm_ratio"] = df_ratio["Amp_norm_Red"] / (df_ratio["Amp_norm_IR"] + 1e-6)

# ★★★ New Raw Ratios ★★★
df_ratio["Red_IR_Raw_Amp_Ratio"] = df_ratio["Amplitude_Red"] / (df_ratio["Amplitude_IR"] + 1e-6)
df_ratio["Red_IR_DC_Ratio"] = df_ratio["DC_Red"] / (df_ratio["DC_IR"] + 1e-6)
df_ratio["Red_IR_PI_Ratio"] = df_ratio["PI_Red"] / (df_ratio["PI_IR"] + 1e-6)

# ★★★ 修正箇所: 面積のRed/IR比の列名計算 (KeyError対策) ★★★
for f in area_features:
    # 列名に_normを付けて、評価リスト(ratio_features_to_eval)と一致させる
    df_ratio[f"{f}_norm_ratio"] = df_ratio[f"{f}_norm_Red"] / (df_ratio[f"{f}_norm_IR"] + 1e-6)
# ★★★ (修正完了) ★★★


# ========= Part 4: ROC AUC計算 (振幅・面積特徴に限定) =========
pairs = [("Normal","Ischaemia"),("Normal","Congestion"),("Ischaemia","Congestion")]

# 評価対象の特徴量リスト (DC, PI added)
features_to_eval = [
    "Amp_norm", "Pulse_Area_norm", "S_AUC_norm", "D_AUC_norm", "AUC_Ratio",
    "DC", "PI"
] 

# Red/IR Ratio 特徴量のリスト (New Ratios added)
ratio_features_to_eval = [
    "Red_IR_amp_norm_ratio",
    "Pulse_Area_norm_ratio", "S_AUC_norm_ratio", "D_AUC_norm_ratio",
    "Red_IR_Raw_Amp_Ratio", "Red_IR_DC_Ratio", "Red_IR_PI_Ratio"
]

# AUC Bootstrap Helper
def bootstrap_auc_ci(y_true, y_score, n_bootstrap=200):
    rng = np.random.default_rng(42)
    aucs=[]
    n=len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0,n,n)
        if len(np.unique(y_true[idx]))<2 or np.all(y_score[idx]==y_score[idx][0]): continue
        aucs.append(roc_auc_score(y_true[idx],y_score[idx]))
    if not aucs: return np.nan,np.nan,np.nan
    return np.mean(aucs), np.percentile(aucs,2.5), np.percentile(aucs,97.5)


rows=[]
# Channel-specific features
for ch in ["Red","IR"]:
    for depth in depths:
        sub = df_feat[(df_feat.Channel==ch)&(df_feat.Depth==depth)]
        if sub.empty: continue
        for f in features_to_eval:
            for c1,c2 in pairs:
                d=sub[sub.State.isin([c1,c2])]
                if d.empty: continue
                y_true=(d.State==c1).astype(int).values
                y_score=d[f].values
                mask=~np.isnan(y_score)
                y_true=y_true[mask]; y_score=y_score[mask]
                
                n_samples_used = len(y_true)
                if n_samples_used < 5 or len(np.unique(y_true))<2: 
                    DEBUG_LOG(f"[WARN] {ch} {depth}mm {c1} vs {c2} ({f}): Insufficient samples ({n_samples_used}) or classes.")
                    continue
                
                auc_bs,ci_lo,ci_hi=bootstrap_auc_ci(y_true,y_score)
                rows.append({"Channel":ch,"Depth":depth,"Feature":f,"Pair":f"{c1} vs {c2}",
                             "AUC_bootstrap_mean":auc_bs,"CI_lower":ci_lo,"CI_upper":ci_hi,"n_samples":n_samples_used})

# Ratio features
for depth in depths:
    sub=df_ratio[df_ratio.Depth==depth]
    if sub.empty: continue
    for f in ratio_features_to_eval:
        for c1,c2 in pairs:
            d=sub[sub.State.isin([c1,c2])]
            if d.empty: continue
            y_true=(d.State==c1).astype(int).values
            y_score=d[f].values
            mask=~np.isnan(y_score)
            y_true=y_true[mask]; y_score=y_score[mask]
            
            n_samples_used = len(y_true)
            if n_samples_used < 5 or len(np.unique(y_true))<2: 
                DEBUG_LOG(f"[WARN] Ratio {depth}mm {c1} vs {c2} ({f}): Insufficient samples ({n_samples_used}) or classes.")
                continue
            
            auc_bs,ci_lo,ci_hi=bootstrap_auc_ci(y_true,y_score)
            rows.append({"Channel":"Red/IR_Ratio","Depth":depth,"Feature":f,"Pair":f"{c1} vs {c2}",
                         "AUC_bootstrap_mean":auc_bs,"CI_lower":ci_lo,"CI_upper":ci_hi,"n_samples":n_samples_used})


df_auc_feat = pd.DataFrame(rows)
df_auc_feat.to_csv("AUC_amplitude_area_features.csv", index=False)
DEBUG_LOG("--- Part 4 Complete: AUC calculation for Amplitude/Area Features saved. ---")
print("✅ Amplitude/Area features AUC analysis complete. Check AUC_amplitude_area_features.csv and log file.")
"""

# Update the notebook
try:
    nb = nbformat.read(notebook_path, as_version=4)
    found = False
    
    for cell in nb.cells:
        if cell.cell_type == 'code':
            # Identify the target cell by its debug log filename
            if 'DEBUG_FILE = "debug_log_part1_to_3_amplitude_features.txt"' in cell.source:
                cell.source = new_cell_source
                found = True
                print("Found and updated the target cell.")
                break
    
    if not found:
        print("Error: Target cell not found.")
    else:
        nbformat.write(nb, notebook_path)
        print(f"Successfully saved updated notebook to {notebook_path}")

except Exception as e:
    print(f"An error occurred: {e}")
