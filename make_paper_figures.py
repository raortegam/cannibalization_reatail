import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Try seaborn for nicer plots; fall back to matplotlib only
try:
    import seaborn as sns  # type: ignore
    HAS_SNS = True
except Exception:
    HAS_SNS = False
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _std_model_type(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    def _map(x: str) -> str:
        if not isinstance(x, str):
            return str(x)
        z = x.lower().replace("_", "-")
        if z.startswith("meta-x"): return "meta-x"
        if z.startswith("meta-t"): return "meta-t"
        if z.startswith("meta-s"): return "meta-s"
        if z.startswith("gsc"): return "gsc"
        return x
    return s.apply(_map)


def _has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(c in df.columns for c in cols)


def _save(fig, outpath: Path) -> None:
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"saved: {outpath}")


def load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Input not found: {path}")
        sys.exit(1)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix.lower() in (".csv", ".parquet"):
        df = pd.read_parquet(path) if path.suffix.lower()==".parquet" else pd.read_csv(path)
    else:
        # try excel as default
        df = pd.read_excel(path)
    # normalize model_type
    if "model_type" in df.columns:
        df.loc[:, "model_type"] = _std_model_type(df["model_type"])  # type: ignore
    # coerce dates if any present
    for c in ("treat_start", "post_end"):
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
    return df


# ================ Section 1: Pre-fit & Causal metrics =================

def fig1_scatter_rmspe_vs_r2(df: pd.DataFrame, out: Path) -> None:
    req = ["rmspe_pre", "pred_r2_pre", "model_type"]
    if not _has_cols(df, req):
        print("skip fig1_scatter_rmspe_vs_r2 (missing columns)")
        return
    d = df.loc[np.isfinite(df["rmspe_pre"]) & np.isfinite(df["pred_r2_pre"])].copy()
    if d.empty:
        print("skip fig1_scatter_rmspe_vs_r2 (no data)")
        return
    fig, ax = plt.subplots(figsize=(8,6))
    if HAS_SNS:
        sns.scatterplot(data=d, x="rmspe_pre", y="pred_r2_pre", hue="model_type", ax=ax, alpha=0.7)
    else:
        for mt, sub in d.groupby("model_type"):
            ax.scatter(sub["rmspe_pre"], sub["pred_r2_pre"], label=str(mt), alpha=0.7)
        ax.legend()
    ax.set_xlabel("RMSPE (pre)")
    ax.set_ylabel("R^2 (pre)")
    ax.set_title("Pre-fit: RMSPE vs R^2")
    _save(fig, out / "1_prefit_scatter_rmspe_vs_r2.png")


def fig1_violin_corr_r2(df: pd.DataFrame, out: Path) -> None:
    req1 = ["pred_corr_pre", "model_type"]
    req2 = ["pred_r2_pre", "model_type"]
    if not (_has_cols(df, req1) and _has_cols(df, req2)):
        print("skip fig1_violin_corr_r2 (missing columns)")
        return
    d1 = df.loc[np.isfinite(df["pred_corr_pre"])].copy()
    d2 = df.loc[np.isfinite(df["pred_r2_pre"])].copy()
    if d1.empty and d2.empty:
        print("skip fig1_violin_corr_r2 (no data)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12,5), sharex=False)
    if HAS_SNS:
        if not d1.empty:
            sns.violinplot(data=d1, x="model_type", y="pred_corr_pre", ax=axes[0], cut=0)
        if not d2.empty:
            sns.violinplot(data=d2, x="model_type", y="pred_r2_pre", ax=axes[1], cut=0)
    else:
        # fallback to boxplot
        if not d1.empty:
            axes[0].boxplot([g["pred_corr_pre"].dropna().values for _, g in d1.groupby("model_type")])
            axes[0].set_xticks(range(1, len(d1["model_type"].unique())+1))
            axes[0].set_xticklabels([str(k) for k,_ in d1.groupby("model_type")])
        if not d2.empty:
            axes[1].boxplot([g["pred_r2_pre"].dropna().values for _, g in d2.groupby("model_type")])
            axes[1].set_xticks(range(1, len(d2["model_type"].unique())+1))
            axes[1].set_xticklabels([str(k) for k,_ in d2.groupby("model_type")])
    axes[0].set_title("pred_corr_pre")
    axes[1].set_title("pred_r2_pre")
    _save(fig, out / "1_prefit_violin_corr_r2.png")


def _topN_by_abs(df: pd.DataFrame, value_col: str, by: str, n: int) -> pd.DataFrame:
    rows = []
    for mt, sub in df.groupby(by):
        s = sub.loc[np.isfinite(sub[value_col])]
        if s.empty:
            continue
        s = s.reindex(s[value_col].abs().sort_values(ascending=False).index).head(n)
        s = s.assign(_rank=np.arange(1, len(s)+1))
        s = s.assign(_mt=mt)
        rows.append(s)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def fig1_lollipop_att_sum(df: pd.DataFrame, out: Path, max_eps_per_model: int = 20) -> None:
    req = ["episode_id", "model_type", "att_sum"]
    if not _has_cols(df, req):
        print("skip fig1_lollipop_att_sum (missing columns)")
        return
    d = df.loc[np.isfinite(df["att_sum"])].copy()
    if d.empty:
        print("skip fig1_lollipop_att_sum (no data)")
        return
    t = _topN_by_abs(d, "att_sum", "model_type", max_eps_per_model)
    if t.empty:
        print("skip fig1_lollipop_att_sum (no top data)")
        return
    # plot horizontally per model_type
    fig, axes = plt.subplots(max(1, t["_mt"].nunique()), 1, figsize=(12, max(4, 2* t["_mt"].nunique())), squeeze=False)
    for ax, (mt, sub) in zip(axes.flatten(), t.groupby("_mt")):
        sub = sub.sort_values("att_sum")
        y = np.arange(len(sub))
        ax.hlines(y=y, xmin=0, xmax=sub["att_sum"].values, color="gray")
        ax.plot(sub["att_sum"].values, y, "o", color="tab:blue")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["episode_id"].astype(str).tolist())
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"att_sum (top {max_eps_per_model}) - {mt}")
    _save(fig, out / "1_effect_lollipop_att_sum_by_model.png")


def fig1_lollipop_rel_att(df: pd.DataFrame, out: Path, max_eps_per_model: int = 20) -> None:
    candidates = ["rel_att_vs_pre_mean", "rel_att"]
    rel_col = next((c for c in candidates if c in df.columns), None)
    req = ["episode_id", "model_type"]
    if rel_col is None or not _has_cols(df, req + [rel_col]):
        print("skip fig1_lollipop_rel_att (missing columns)")
        return
    d = df.loc[np.isfinite(df[rel_col])].copy()
    if d.empty:
        print("skip fig1_lollipop_rel_att (no data)")
        return
    t = _topN_by_abs(d, rel_col, "model_type", max_eps_per_model)
    if t.empty:
        print("skip fig1_lollipop_rel_att (no top data)")
        return
    fig, axes = plt.subplots(max(1, t["_mt"].nunique()), 1, figsize=(12, max(4, 2* t["_mt"].nunique())), squeeze=False)
    for ax, (mt, sub) in zip(axes.flatten(), t.groupby("_mt")):
        sub = sub.sort_values(rel_col)
        y = np.arange(len(sub))
        ax.hlines(y=y, xmin=0, xmax=sub[rel_col].values, color="gray")
        ax.plot(sub[rel_col].values, y, "o", color="tab:green")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["episode_id"].astype(str).tolist())
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{rel_col} (top {max_eps_per_model}) - {mt}")
    _save(fig, out / "1_effect_lollipop_rel_att_by_model.png")


def fig1_heterogeneity_violin(df: pd.DataFrame, out: Path) -> None:
    req_std = ["het_tau_std", "model_type"]
    req_cv = ["het_tau_cv", "model_type"]
    if not (_has_cols(df, req_std) and _has_cols(df, req_cv)):
        print("skip fig1_heterogeneity_violin (missing columns)")
        return
    d1 = df.loc[np.isfinite(df["het_tau_std"])].copy()
    d2 = df.loc[np.isfinite(df["het_tau_cv"])].copy()
    if d1.empty and d2.empty:
        print("skip fig1_heterogeneity_violin (no data)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    if HAS_SNS:
        if not d1.empty:
            sns.violinplot(data=d1, x="model_type", y="het_tau_std", ax=axes[0], cut=0)
        if not d2.empty:
            sns.violinplot(data=d2, x="model_type", y="het_tau_cv", ax=axes[1], cut=0)
    else:
        if not d1.empty:
            axes[0].boxplot([g["het_tau_std"].dropna().values for _, g in d1.groupby("model_type")])
            axes[0].set_xticks(range(1, len(d1["model_type"].unique())+1))
            axes[0].set_xticklabels([str(k) for k,_ in d1.groupby("model_type")])
        if not d2.empty:
            axes[1].boxplot([g["het_tau_cv"].dropna().values for _, g in d2.groupby("model_type")])
            axes[1].set_xticks(range(1, len(d2["model_type"].unique())+1))
            axes[1].set_xticklabels([str(k) for k,_ in d2.groupby("model_type")])
    axes[0].set_title("het_tau_std")
    axes[1].set_title("het_tau_cv")
    _save(fig, out / "1_heterogeneity_violin_tau_std_cv.png")


def fig1_coverage_stacked_bars(df: pd.DataFrame, out: Path, max_eps_per_model: int = 20) -> None:
    req = ["episode_id", "model_type", "n_pre", "n_post"]
    if not _has_cols(df, req):
        print("skip fig1_coverage_stacked_bars (missing columns)")
        return
    d = df.loc[np.isfinite(df["n_pre"]) & np.isfinite(df["n_post"])].copy()
    if d.empty:
        print("skip fig1_coverage_stacked_bars (no data)")
        return
    t = _topN_by_abs(d.assign(_sum=d["n_pre"].values + d["n_post"].values), "_sum", "model_type", max_eps_per_model)
    if t.empty:
        print("skip fig1_coverage_stacked_bars (no top data)")
        return
    fig, axes = plt.subplots(max(1, t["_mt"].nunique()), 1, figsize=(12, max(4, 2* t["_mt"].nunique())), squeeze=False)
    for ax, (mt, sub) in zip(axes.flatten(), t.groupby("_mt")):
        sub = sub.sort_values("_sum", ascending=True)
        y = np.arange(len(sub))
        ax.barh(y, sub["n_pre"].values, color="#9ecae1", label="n_pre")
        ax.barh(y, sub["n_post"].values, left=sub["n_pre"].values, color="#3182bd", label="n_post")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["episode_id"].astype(str).tolist())
        ax.set_title(f"Cobertura temporal (top {max_eps_per_model}) - {mt}")
        ax.legend()
    _save(fig, out / "1_coverage_stacked_bars_topN.png")


# ================ Section 2: Robustness & Falsification =================

def fig2_scatter_ratio_vs_p(df: pd.DataFrame, out: Path) -> None:
    req = ["plac_effect_ratio", "p_value_placebo_space", "model_type"]
    if not _has_cols(df, req):
        print("skip fig2_scatter_ratio_vs_p (missing columns)")
        return
    d = df.loc[np.isfinite(df["plac_effect_ratio"]) & np.isfinite(df["p_value_placebo_space"])].copy()
    if d.empty:
        print("skip fig2_scatter_ratio_vs_p (no data)")
        return
    fig, ax = plt.subplots(figsize=(8,6))
    if HAS_SNS:
        sns.scatterplot(data=d, x="plac_effect_ratio", y="p_value_placebo_space", hue="model_type", ax=ax, alpha=0.7)
    else:
        for mt, sub in d.groupby("model_type"):
            ax.scatter(sub["plac_effect_ratio"], sub["p_value_placebo_space"], label=str(mt), alpha=0.7)
        ax.legend()
    ax.axhline(0.10, color="red", linestyle="--", linewidth=1)
    ax.axhline(0.05, color="orange", linestyle="--", linewidth=1)
    ax.set_xlabel("|ATT| / mean(|placebo|)")
    ax.set_ylabel("p-value (placebo espacio)")
    ax.set_title("Robustez: Ratio efecto vs p-value (espacio)")
    _save(fig, out / "2_robustness_scatter_ratio_vs_p.png")


def fig2_scatter_placeboN_vs_p(df: pd.DataFrame, out: Path) -> None:
    req = ["placebo_n", "p_value_placebo_space", "model_type"]
    if not _has_cols(df, req):
        print("skip fig2_scatter_placeboN_vs_p (missing columns)")
        return
    d = df.loc[np.isfinite(df["placebo_n"]) & np.isfinite(df["p_value_placebo_space"])].copy()
    if d.empty:
        print("skip fig2_scatter_placeboN_vs_p (no data)")
        return
    fig, ax = plt.subplots(figsize=(8,6))
    if HAS_SNS:
        sns.scatterplot(data=d, x="placebo_n", y="p_value_placebo_space", hue="model_type", ax=ax, alpha=0.7)
    else:
        for mt, sub in d.groupby("model_type"):
            ax.scatter(sub["placebo_n"], sub["p_value_placebo_space"], label=str(mt), alpha=0.7)
        ax.legend()
    # p_min curve = 1/(1+N)
    x = np.linspace(1, max(2, float(np.nanmax(d["placebo_n"]))), 200)
    y = 1.0/(1.0 + x)
    ax.plot(x, y, color="black", linestyle=":", linewidth=1, label="p_min=1/(1+N)")
    ax.set_xlabel("placebo_n (donantes)")
    ax.set_ylabel("p-value (placebo espacio)")
    ax.set_title("Robustez: p-value vs cantidad de placebos")
    _save(fig, out / "2_robustness_scatter_placeboN_vs_p.png")


def fig2_bars_placebo_abs(df: pd.DataFrame, out: Path) -> None:
    req = ["model_type", "placebo_mean_abs", "placebo_q95_abs"]
    if not _has_cols(df, req):
        print("skip fig2_bars_placebo_abs (missing columns)")
        return
    g = df.groupby("model_type", as_index=False).agg(
        placebo_mean_abs_mean=("placebo_mean_abs", "mean"),
        placebo_q95_abs_median=("placebo_q95_abs", "median"),
        count=("model_type", "size")
    )
    if g.empty:
        print("skip fig2_bars_placebo_abs (no data)")
        return
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(g["model_type"], g["placebo_mean_abs_mean"], color="#a1d99b", label="mean(|placebo|)")
    ax.plot(g["model_type"], g["placebo_q95_abs_median"], "o", color="#31a354", label="median q95(|placebo|)")
    ax.set_ylabel("magnitude")
    ax.set_title("Ruido placebo en espacio (absoluto)")
    ax.legend()
    _save(fig, out / "2_robustness_bars_placebo_abs.png")


def fig2_ecdf_pvalues(df: pd.DataFrame, out: Path) -> None:
    req = ["p_value_placebo_space", "model_type"]
    if not _has_cols(df, req):
        print("skip fig2_ecdf_pvalues (missing columns)")
        return
    d = df.loc[np.isfinite(df["p_value_placebo_space"])].copy()
    if d.empty:
        print("skip fig2_ecdf_pvalues (no data)")
        return
    fig, ax = plt.subplots(figsize=(8,6))
    for mt, sub in d.groupby("model_type"):
        v = np.sort(sub["p_value_placebo_space"].to_numpy(dtype=float))
        if v.size == 0:
            continue
        x = v
        y = np.arange(1, v.size+1) / v.size
        ax.step(x, y, where="post", label=str(mt))
    ax.set_xlabel("p-value (placebo espacio)")
    ax.set_ylabel("eCDF")
    ax.set_title("Curva de significancia (eCDF)")
    ax.legend(loc="lower right")
    _save(fig, out / "2_robustness_ecdf_p_value_space.png")


def fig2_violin_time_placebo(df: pd.DataFrame, out: Path) -> None:
    if "time_placebo_mean" not in df.columns or "model_type" not in df.columns:
        print("skip fig2_violin_time_placebo (missing columns)")
        return
    d = df.loc[np.isfinite(df["time_placebo_mean"])].copy()
    if d.empty:
        print("skip fig2_violin_time_placebo (no data)")
        return
    fig, ax = plt.subplots(figsize=(8,5))
    if HAS_SNS:
        sns.violinplot(data=d, x="model_type", y="time_placebo_mean", ax=ax, cut=0)
    else:
        ax.boxplot([g["time_placebo_mean"].dropna().values for _, g in d.groupby("model_type")])
        ax.set_xticks(range(1, len(d["model_type"].unique())+1))
        ax.set_xticklabels([str(k) for k,_ in d.groupby("model_type")])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Placebo en tiempo (media pseudo-post)")
    _save(fig, out / "2_robustness_violin_time_placebo_mean.png")


# ================ Section 3: Global Sensitivity (GSC) =================

def fig3_scatter_att_mean_vs_params_gsc(df: pd.DataFrame, out: Path) -> None:
    # Only GSC rows
    if "model_type" not in df.columns:
        print("skip fig3_scatter_att_mean_vs_params_gsc (no model_type)")
        return
    gsc = df.loc[df["model_type"].astype(str).str.lower().str.startswith("gsc")].copy()
    if gsc.empty:
        print("skip fig3_scatter_att_mean_vs_params_gsc (no gsc rows)")
        return
    params = [c for c in ["rank", "tau", "alpha"] if c in gsc.columns]
    if not params:
        print("skip fig3_scatter_att_mean_vs_params_gsc (no sensitivity params)")
        return
    if "att_mean" not in gsc.columns:
        print("skip fig3_scatter_att_mean_vs_params_gsc (missing att_mean)")
        return
    n = len(params)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), squeeze=False)
    for ax, pcol in zip(axes.flatten(), params):
        sub = gsc.loc[np.isfinite(gsc[pcol]) & np.isfinite(gsc["att_mean"])].copy()
        if sub.empty:
            continue
        if HAS_SNS:
            sns.scatterplot(data=sub, x=pcol, y="att_mean", ax=ax, alpha=0.7)
        else:
            ax.scatter(sub[pcol], sub["att_mean"], alpha=0.7)
        ax.set_xlabel(pcol)
        ax.set_ylabel("att_mean")
        ax.set_title(f"GSC: att_mean vs {pcol}")
    _save(fig, out / "3_sensitivity_scatter_att_mean_vs_rank_tau_alpha_gsc.png")


def fig3_violin_att_mean_by_model(df: pd.DataFrame, out: Path) -> None:
    if not _has_cols(df, ["att_mean", "model_type"]):
        print("skip fig3_violin_att_mean_by_model (missing columns)")
        return
    d = df.loc[np.isfinite(df["att_mean"])].copy()
    if d.empty:
        print("skip fig3_violin_att_mean_by_model (no data)")
        return
    fig, ax = plt.subplots(figsize=(8,5))
    if HAS_SNS:
        sns.violinplot(data=d, x="model_type", y="att_mean", ax=ax, cut=0)
    else:
        ax.boxplot([g["att_mean"].dropna().values for _, g in d.groupby("model_type")])
        ax.set_xticks(range(1, len(d["model_type"].unique())+1))
        ax.set_xticklabels([str(k) for k,_ in d.groupby("model_type")])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Distribución de att_mean por modelo")
    _save(fig, out / "3_sensitivity_violin_att_mean_by_model.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(Path("all_metrics_merged_by_episode_final.xlsx")))
    ap.add_argument("--outdir", default=str(Path("paper_figures")))
    ap.add_argument("--max_eps_per_model", type=int, default=20)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.outdir)
    _ensure_dir(out)

    if HAS_SNS:
        try:
            sns.set_theme(style="whitegrid")
        except Exception:
            pass

    df = load_input(inp)

    # Section 1
    fig1_scatter_rmspe_vs_r2(df, out)
    fig1_violin_corr_r2(df, out)
    fig1_lollipop_att_sum(df, out, args.max_eps_per_model)
    fig1_lollipop_rel_att(df, out, args.max_eps_per_model)
    fig1_heterogeneity_violin(df, out)
    fig1_coverage_stacked_bars(df, out, args.max_eps_per_model)

    # Section 2
    fig2_scatter_ratio_vs_p(df, out)
    fig2_scatter_placeboN_vs_p(df, out)
    fig2_bars_placebo_abs(df, out)
    fig2_ecdf_pvalues(df, out)
    fig2_violin_time_placebo(df, out)

    # Section 3
    fig3_scatter_att_mean_vs_params_gsc(df, out)
    fig3_violin_att_mean_by_model(df, out)

    print("Done.")


if __name__ == "__main__":
    main()
