#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_eda_only.py
===============
Script para ejecutar SOLO el EDA de algoritmos (GSC + Meta-learners) sin correr el pipeline completo.

Uso:
    python run_eda_only.py --exp_tag A_quick_smoke
    python run_eda_only.py --exp_tag A_base --max_episodes_gsc 20 --max_episodes_meta 20
    python run_eda_only.py --exp_tag A_quick_smoke --learners x s t
"""

import argparse
import logging
import sys
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("eda_only")

# Rutas base del proyecto
REPO_ROOT = Path(__file__).resolve().parent
EDA_DIR = REPO_ROOT / "EDA"

# Añadir EDA al path
sys.path.insert(0, str(EDA_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta solo el EDA de algoritmos (GSC + Meta-learners)"
    )
    parser.add_argument(
        "--exp_tag",
        type=str,
        required=True,
        help="Tag del experimento (e.g., A_quick_smoke, A_base)"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".data/processed_data",
        help="Directorio base de datos procesados (default: .data/processed_data)"
    )
    parser.add_argument(
        "--figures_dir",
        type=str,
        default="figures",
        help="Directorio base para figuras (default: figures)"
    )
    parser.add_argument(
        "--learners",
        nargs="+",
        default=["x"],
        choices=["t", "s", "x"],
        help="Meta-learners a incluir en el EDA (default: x)"
    )
    parser.add_argument(
        "--max_episodes_gsc",
        type=int,
        default=None,
        help="Máximo de episodios a renderizar para GSC (default: todos)"
    )
    parser.add_argument(
        "--max_episodes_meta",
        type=int,
        default=None,
        help="Máximo de episodios a renderizar para Meta-learners (default: todos)"
    )
    parser.add_argument(
        "--orientation",
        type=str,
        default="landscape",
        choices=["landscape", "portrait"],
        help="Orientación de las figuras (default: landscape)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI de las figuras (default: 300)"
    )
    parser.add_argument(
        "--no_pdf",
        action="store_true",
        help="No exportar PDFs (solo PNGs)"
    )
    
    args = parser.parse_args()
    
    # Construir rutas basadas en exp_tag
    exp_tag = args.exp_tag
    base_dir = Path(args.base_dir)
    figures_base = Path(args.figures_dir)
    
    # Rutas específicas del experimento
    processed_dir = base_dir / exp_tag
    gsc_out_dir = processed_dir / "gsc"
    meta_out_root = base_dir / "meta_outputs" / exp_tag
    figures_dir = figures_base / exp_tag
    episodes_index = processed_dir / "episodes_index.parquet"
    
    logger.info("=" * 80)
    logger.info("EDA de Algoritmos - Solo visualización")
    logger.info("=" * 80)
    logger.info(f"Experimento: {exp_tag}")
    logger.info(f"Episodes index: {episodes_index}")
    logger.info(f"GSC output: {gsc_out_dir}")
    logger.info(f"Meta output: {meta_out_root}")
    logger.info(f"Figuras: {figures_dir}")
    logger.info(f"Learners: {args.learners}")
    logger.info("=" * 80)
    
    # Validar que existan los archivos necesarios
    if not episodes_index.exists():
        logger.error(f"❌ No existe episodes_index: {episodes_index}")
        logger.error("   Ejecuta primero el pipeline completo o al menos hasta Step 4")
        sys.exit(1)
    
    # Verificar que existan outputs de algoritmos
    gsc_exists = (gsc_out_dir / "gsc_metrics.parquet").exists()
    meta_exists = any((meta_out_root / lr).exists() for lr in args.learners)
    
    if not gsc_exists and not meta_exists:
        logger.warning("⚠️  No se encontraron outputs de GSC ni Meta-learners")
        logger.warning(f"   GSC: {gsc_out_dir / 'gsc_metrics.parquet'}")
        logger.warning(f"   Meta: {meta_out_root}")
        logger.warning("   El EDA puede estar vacío o incompleto")
    
    if gsc_exists:
        logger.info("✓ Encontrados outputs de GSC")
    else:
        logger.warning("⚠️  No se encontraron outputs de GSC")
    
    if meta_exists:
        logger.info(f"✓ Encontrados outputs de Meta-learners: {args.learners}")
    else:
        logger.warning("⚠️  No se encontraron outputs de Meta-learners")
    
    # Importar EDA_algorithms
    try:
        from EDA_algorithms import EDAConfig, run as eda_run
        logger.info("✓ Módulo EDA_algorithms importado correctamente")
    except ImportError as e:
        logger.error(f"❌ No se pudo importar EDA_algorithms: {e}")
        logger.error("   Verifica que el archivo EDA/EDA_algorithms.py exista")
        sys.exit(1)
    
    # Configurar EDA
    eda_cfg = EDAConfig(
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
    
    logger.info("Iniciando EDA de algoritmos...")
    logger.info("-" * 80)
    
    try:
        eda_run(cfg=eda_cfg)
        logger.info("-" * 80)
        logger.info("✅ EDA completado exitosamente")
        logger.info(f"📁 Figuras guardadas en: {figures_dir}")
        
        # Listar algunos archivos generados
        if figures_dir.exists():
            png_files = list(figures_dir.glob("*.png"))
            pdf_files = list(figures_dir.glob("*.pdf"))
            logger.info(f"   - {len(png_files)} archivos PNG generados")
            logger.info(f"   - {len(pdf_files)} archivos PDF generados")
            
            # Mostrar algunos ejemplos
            if png_files:
                logger.info("   Ejemplos de PNGs:")
                for f in png_files[:5]:
                    logger.info(f"     • {f.name}")
                if len(png_files) > 5:
                    logger.info(f"     ... y {len(png_files) - 5} más")
        
    except Exception as e:
        logger.exception("❌ Error durante la ejecución del EDA")
        sys.exit(1)


if __name__ == "__main__":
    main()
