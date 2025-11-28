#!/usr/bin/env python3
import argparse, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "EDA"))
from EDA_algorithms import EDAConfig, run as eda_run  # noqa: E402

def main():
    p = argparse.ArgumentParser(description="Regenera figuras EDA (sin franja) para un experimento.")
    p.add_argument("--exp_tag", required=True)
    p.add_argument("--learners", nargs="+", default=["t","s","x"], choices=["t","s","x"])
    p.add_argument("--orientation", default="landscape", choices=["landscape","portrait"])
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--max_episodes_gsc", type=int, default=None)
    p.add_argument("--max_episodes_meta", type=int, default=None)
    p.add_argument("--no_pdf", action="store_true")
    args = p.parse_args()

    base = Path(".data/processed_data")
    episodes_index = base / args.exp_tag / "episodes_index.parquet"
    gsc_out_dir = base / args.exp_tag / "gsc"
    meta_out_root = base / "meta_outputs" / args.exp_tag
    figures_dir = Path("figures") / args.exp_tag

    if not episodes_index.exists():
        print(f"episodes_index no existe: {episodes_index}", file=sys.stderr)
        sys.exit(1)

    cfg = EDAConfig(
        episodes_index=episodes_index,
        gsc_out_dir=gsc_out_dir,
        meta_out_root=meta_out_root,
        meta_learners=tuple(args.learners),
        figures_dir=figures_dir,
        orientation=args.orientation,
        dpi=args.dpi,
        style="academic",
        font_size=10,
        grid=True,
        max_episodes_gsc=args.max_episodes_gsc,
        max_episodes_meta=args.max_episodes_meta,
        export_pdf=not args.no_pdf,
    )
    eda_run(cfg=cfg)

if __name__ == "__main__":
    main()