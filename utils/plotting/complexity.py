import pandas as pd
from ..constants import RESULTS_DIR

def plot_complexity_measures(dataset_key, configuration_name, flat_data, drift_info_by_id):
    import matplotlib.pyplot as plt
    df = pd.DataFrame(flat_data)
    df["start_time"] = pd.to_datetime(df["start_time"]); df["end_time"] = pd.to_datetime(df["end_time"])
    label_map = {"sudden":"Sudden","gradual_start":"Gradual start","gradual_end":"Gradual end","start":"Start","end":"End"}
    measures = [c for c in df.columns if c.startswith("measure_")]
    for mcol in measures:
        mname = mcol.removeprefix("measure_")
        plt.figure(figsize=(12,5))
        y_min, y_max = df[mcol].min(), df[mcol].max()
        for _, r in df.iterrows():
            plt.plot([r["start_time"], r["end_time"]], [r[mcol], r[mcol]])
            if "measure_Support" in df.columns:
                mid = r["start_time"] + (r["end_time"]-r["start_time"])/2
                y_pos = r[mcol] + 0.002 * (y_max - y_min if y_max != y_min else 1)
                plt.text(mid, y_pos, f'N={int(r["measure_Support"])}', fontsize=7, ha='center', va='bottom')
        label_y = (y_max - y_min) * 0.08 if y_max != y_min else 0.1
        xs, xe = df["start_time"].min(), df["end_time"].max()
        for x, lab in [(xs,"start"), (xe,"end")]:
            plt.axvline(x, linestyle='--', linewidth=1)
            plt.text(x, y_max + label_y, label_map[lab], fontsize=8, ha='left', va='bottom', rotation=45)
        # CPs (works for cp & fixed)
        if drift_info_by_id:
            for _, info in drift_info_by_id.items():
                if _ == "na": continue
                x = pd.to_datetime(info["calc_change_moment"])
                lab = label_map.get(info["calc_change_type"], info["calc_change_type"])
                plt.axvline(x=x, color='red', linestyle='--', alpha=0.5)
                plt.text(x, y_max + label_y, lab, rotation=45, fontsize=8, ha='left', va='bottom')
        plt.xlabel("Time"); plt.ylim(bottom=0); plt.ylabel(mname); plt.xticks(rotation=45); plt.grid(True); plt.tight_layout()
        out = RESULTS_DIR / dataset_key / configuration_name / f"{mname}_over_time.png"
        out.parent.mkdir(parents=True, exist_ok=True); plt.savefig(out, dpi=600); plt.close()

def plot_delta_measures(dataset_key, configuration_name, paired_df, drift_info_by_id):
    import matplotlib.pyplot as plt
    df = pd.DataFrame(paired_df).copy()
    for c in ["w1_start_time","w1_end_time","w2_start_time","w2_end_time"]:
        if c in df.columns: df[c] = pd.to_datetime(df[c])
    df["start_time"] = df[["w1_start_time","w2_start_time"]].min(axis=1)
    df["end_time"]   = df[["w1_end_time","w2_end_time"]].max(axis=1)
    delta_cols = [c for c in df.columns if c.startswith("delta_")]
    for dcol in delta_cols:
        dname = dcol.removeprefix("delta_")
        plt.figure(figsize=(12,5))
        y_min, y_max = df[dcol].min(), df[dcol].max()
        for _, r in df.iterrows():
            plt.plot([r["start_time"], r["end_time"]], [r[dcol], r[dcol]])
        label_y = (y_max - y_min) * 0.08 if y_max != y_min else 0.1
        xs, xe = df["start_time"].min(), df["end_time"].max()
        for x, lab in [(xs,"Start"), (xe,"End")]:
            plt.axvline(x, linestyle='--', linewidth=1); plt.text(x, y_max + label_y, lab, fontsize=8, rotation=45)
        if drift_info_by_id:
            for k, info in drift_info_by_id.items():
                if k == "na": continue
                x = pd.to_datetime(info["calc_change_moment"])
                plt.axvline(x=x, color='red', linestyle='--', alpha=0.5)
                plt.text(x, y_max + label_y, info.get("calc_change_type","cp"), rotation=45, fontsize=8)
        plt.xlabel("Time"); plt.ylabel(f"Δ {dname}"); plt.grid(True); plt.tight_layout()
        out = RESULTS_DIR / dataset_key / configuration_name / f"delta_{dname}_over_time.png"
        out.parent.mkdir(parents=True, exist_ok=True); plt.savefig(out, dpi=600); plt.close()
