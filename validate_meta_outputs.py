# validate_meta_outputs.py
from pathlib import Path
import argparse
import pandas as pd

def _read_df_any(p_csv: Path, p_parq: Path | None = None) -> pd.DataFrame:
    try:
        if p_csv and p_csv.exists():
            return pd.read_csv(p_csv)
    except Exception:
        pass
    if p_parq and p_parq.exists():
        try:
            return pd.read_parquet(p_parq)
        except Exception:
            pass
    return pd.DataFrame()

def print_placebo_counts(meta_root: Path, learners: list[str]) -> None:
    print("learner,n_space,n_nonempty_space,n_time")
    for lr in learners:
        d = meta_root / lr / "placebos"
        if not d.exists():
            print(f"{lr},0,0,0")
            continue
        space_files = list(d.glob("*_space.parquet"))
        time_files = list(d.glob("*_time.parquet"))
        n_space = len(space_files)
        n_nonempty = 0
        for p in space_files:
            try:
                df = pd.read_parquet(p, columns=None)
                if df.shape[0] > 0:
                    n_nonempty += 1
            except Exception:
                pass
        n_time = len(time_files)
        print(f"{lr},{n_space},{n_nonempty},{n_time}")

def print_loo_sens_counts(meta_root: Path, learners: list[str]) -> None:
    print("learner,loo_files,sens_files")
    for lr in learners:
        loo_dir = meta_root / lr / "loo"
        sens_dir = meta_root / lr / "sensitivity"
        n_loo = len(list(loo_dir.glob("*_loo.parquet"))) if loo_dir.exists() else 0
        n_sens = len(list(sens_dir.glob("*_sens.parquet"))) if sens_dir.exists() else 0
        print(f"{lr},{n_loo},{n_sens}")

def summarize_robustness(fig_tables: Path) -> None:
    df = _read_df_any(fig_tables / "robustness_combined_by_episode.csv",
                      fig_tables / "robustness_combined_by_episode.parquet")
    if df.empty or "model_type" not in df.columns:
        print("robustness_combined_by_episode no encontrado o sin columnas esperadas.")
        return
    df = df[df["model_type"].astype(str).str.startswith("meta-")]
    if df.empty:
        print("No hay filas meta-* en robustness_combined_by_episode.")
        return
    out = df.groupby("model_type").apply(
        lambda g: pd.Series({
            "placebo_space_ep": int((g["placebo_n"].fillna(0) > 0).sum()) if "placebo_n" in g.columns else 0,
            "placebo_time_ep": int(g["time_placebo_mean"].notna().sum()) if "time_placebo_mean" in g.columns else 0,
            "loo_ep": int((g["loo_n"].fillna(0) > 0).sum()) if "loo_n" in g.columns else 0,
            "sens_ep": int((g["sens_n"].fillna(0) > 0).sum()) if "sens_n" in g.columns else 0,
        })
    )
    print("\nResumen por model_type (episodios con artefactos presentes):")
    print(out)

def show_placebo_examples(fig_tables: Path, learners: list[str], n: int = 10) -> None:
    print("\nEjemplos de placebos en espacio (por learner):")
    for lr in learners:
        df = _read_df_any(fig_tables / f"robustness_meta_{lr}_by_episode.csv",
                          fig_tables / f"robustness_meta_{lr}_by_episode.parquet")
        if df.empty:
            print(lr, "-> sin tabla de robustez")
            continue
        if "placebo_n" not in df.columns:
            print(lr, "-> tabla sin columna placebo_n")
            continue
        sub_cols = [
            "episode_id","placebo_n","placebo_mean","placebo_std","placebo_q05","placebo_q50","placebo_q95",
            "placebo_mean_abs","placebo_q95_abs","placebo_p_value_two_sided","effect_to_placebo_ratio_abs"
        ]
        cols = [c for c in sub_cols if c in df.columns]
        sub = df[df["placebo_n"].fillna(0) > 0][cols].head(n)
        print(lr)
        print(sub.to_string(index=False) if not sub.empty else "sin episodios con placebos en espacio")

def show_loo_sens_examples(fig_tables: Path, n: int = 10) -> None:
    df = _read_df_any(fig_tables / "robustness_combined_by_episode.csv",
                      fig_tables / "robustness_combined_by_episode.parquet")
    if df.empty:
        print("\nNo hay robustness_combined_by_episode.")
        return
    df = df[df["model_type"].astype(str).str.startswith("meta-")]
    print("\nLOO ejemplos:")
    loo_cols = ["episode_id","model_type","loo_n","loo_sd","loo_min","loo_max","loo_range","loo_sign_flip_rate"]
    if "loo_n" in df.columns:
        sub = df[df["loo_n"].fillna(0) > 0][[c for c in loo_cols if c in df.columns]].head(n)
        print(sub.to_string(index=False) if not sub.empty else "sin ejemplos LOO")
    else:
        print("sin columna LOO")

    print("\nSens ejemplos:")
    sens_cols = ["episode_id","model_type","sens_n","sens_sd","sens_min","sens_max","sens_range","sens_relative_std"]
    if "sens_n" in df.columns:
        sub = df[df["sens_n"].fillna(0) > 0][[c for c in sens_cols if c in df.columns]].head(n)
        print(sub.to_string(index=False) if not sub.empty else "sin ejemplos Sens")
    else:
        print("sin columna Sens")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_tag", required=True)
    ap.add_argument("--learners", nargs="*", default=["x","t","s"])
    args = ap.parse_args()

    meta_root = Path(".data/processed_data") / "meta_outputs" / args.exp_tag
    fig_tables = Path("figures") / args.exp_tag / "tables"

    print(f"Meta root: {meta_root}")
    print(f"Figures tables: {fig_tables}\n")

    print_placebo_counts(meta_root, args.learners)
    print()
    print_loo_sens_counts(meta_root, args.learners)
    summarize_robustness(fig_tables)
    show_placebo_examples(fig_tables, args.learners, n=10)
    show_loo_sens_examples(fig_tables, n=10)

if __name__ == "__main__":
    main()