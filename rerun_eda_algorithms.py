#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script temporal para re-ejecutar EDA_algorithms para A_base
"""

from pathlib import Path
from EDA.EDA_algorithms import EDAConfig, run

cfg = EDAConfig(
    episodes_index=Path('.data/processed_data/A_base/episodes_index.parquet'),
    gsc_out_dir=Path('.data/processed_data/A_base/gsc'),
    meta_out_root=Path('.data/processed_data/A_base'),
    meta_learners=('t', 's', 'x'),
    figures_dir=Path('figures/A_base'),
    orientation='landscape',
    dpi=300,
    style='academic',
    font_size=10,
    grid=True,
    max_episodes_gsc=None,
    max_episodes_meta=None,
    export_pdf=True,
)

print("Ejecutando EDA_algorithms para A_base...")
run(cfg)
print("✅ Completado!")
