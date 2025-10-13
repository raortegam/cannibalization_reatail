# src/treatment/build_treatment_gpu.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Union, Optional

import pandas as pd
import numpy as np


def build_treatment_design_gpu(
    panel: Union[pd.DataFrame, "cudf.DataFrame"],
    treatment_col: str,
    groupby: Iterable[str] = ("store_nbr", "item_nbr"),
    *,
    add_group_tuple: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Versión GPU (cuDF/RAPIDS) para detectar episodios de tratamiento y etiquetar el panel.

    Parámetros
    ----------
    panel : pandas.DataFrame o cudf.DataFrame
        Debe contener columnas: *groupby, "w_idx", treatment_col*.
        Si viene en pandas, se convertirá a cuDF internamente.
    treatment_col : str
        Nombre de la columna binaria de tratamiento (0/1).
    groupby : Iterable[str]
        Columnas por las cuales se definen los grupos (p.ej., tienda, item).
    add_group_tuple : bool, default True
        Añade columna 'group' (tupla de claves) a 'episodes' para compatibilidad.

    Retorna
    -------
    panel_labeled : pandas.DataFrame
        Panel original con columnas añadidas: 'role' y 'episode_id'.
    episodes : pandas.DataFrame
        Lista de episodios con columnas: groupby..., 's_w', 'e_w', 'episode_id' (+ 'group' opcional).
    windows_long : pandas.DataFrame
        Filas por semana tratada con columnas: groupby..., 'w_idx', 'episode_id', 'role'.

    Notas
    -----
    - Implementación 100% vectorizada sobre GPU (cuDF). Sin bucles Python por fila.
    - Define episodios como rachas contiguas de 1s en 'treatment_col' dentro de cada grupo.
    """
    try:
        import cudf  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Se requiere RAPIDS cuDF para la versión GPU. Instálalo según tu entorno CUDA."
        ) from e

    gb_cols: List[str] = list(groupby)

    # Validaciones mínimas
    for c in [*gb_cols, "w_idx", treatment_col]:
        if c not in panel.columns:
            raise ValueError(f"Falta la columna requerida '{c}' en 'panel'.")

    # Conserva el panel original en pandas para devolver todas las columnas
    if isinstance(panel, pd.DataFrame):
        panel_pd = panel.copy()
        gdf = cudf.from_pandas(panel_pd[gb_cols + ["w_idx", treatment_col]])
    else:
        # Si ya es cuDF, también construye un pandas para el merge final
        gdf = panel[gb_cols + ["w_idx", treatment_col]].copy()
        panel_pd = panel.to_pandas()

    # Tipos compactos
    gdf["w_idx"] = gdf["w_idx"].astype("int32")
    # Asegura binario (NaN -> 0)
    gdf["treatment_used"] = gdf[treatment_col].fillna(0).astype("int8")

    # Orden por grupo y tiempo
    gdf = gdf.sort_values(by=gb_cols + ["w_idx"])

    # Marca de inicio de episodio: transición 0 -> 1
    prev_t = gdf.groupby(gb_cols).treatment_used.shift(1).fillna(0).astype("int8")
    gdf["start"] = ((gdf["treatment_used"] == 1) & (prev_t == 0)).astype("int8")

    # ID incremental de episodio *dentro del grupo*
    gdf["ep_id_in_group"] = gdf.groupby(gb_cols).start.cumsum().astype("int32")

    # Filas tratadas
    treat_mask = gdf["treatment_used"] == 1

    # ---- EPISODIOS: min y max de w_idx por (grupo, ep_id_in_group)
    eps_min = (
        gdf.loc[treat_mask]
        .groupby(gb_cols + ["ep_id_in_group"])
        .agg({"w_idx": "min"})
        .reset_index()
        .rename(columns={"w_idx": "s_w"})
    )
    eps_max = (
        gdf.loc[treat_mask]
        .groupby(gb_cols + ["ep_id_in_group"])
        .agg({"w_idx": "max"})
        .reset_index()
        .rename(columns={"w_idx": "e_w"})
    )
    episodes_g = eps_min.merge(eps_max, on=gb_cols + ["ep_id_in_group"])
    episodes_g = episodes_g.sort_values(by=gb_cols + ["s_w"]).reset_index(drop=True)
    episodes_g["episode_id"] = cudf.Series(range(len(episodes_g)), dtype="int64")

    # ---- WINDOWS_LONG: todas las semanas tratadas con su episode_id
    windows_long_g = (
        gdf.loc[treat_mask, gb_cols + ["w_idx", "ep_id_in_group"]]
        .merge(
            episodes_g[gb_cols + ["ep_id_in_group", "episode_id"]],
            on=gb_cols + ["ep_id_in_group"],
            how="left",
        )
        .drop(columns=["ep_id_in_group"])
    )
    windows_long_g["role"] = "treat"

    # ---- PANEL_LABELED: merge de etiquetas sobre el panel original (pandas)
    windows_long_pd = windows_long_g.to_pandas()
    episodes_pd = episodes_g.drop(columns=["ep_id_in_group"]).to_pandas()

    # Asegura tipos en pandas
    windows_long_pd["episode_id"] = windows_long_pd["episode_id"].astype("Int64")
    episodes_pd["episode_id"] = episodes_pd["episode_id"].astype("int64")

    # Añade columna 'group' (tupla) para compatibilidad con tu función original, si se desea
    if add_group_tuple:
        episodes_pd["group"] = list(zip(*[episodes_pd[c] for c in gb_cols]))

    # Etiquetado del panel manteniendo TODAS las columnas originales
    panel_labeled = panel_pd.merge(
        windows_long_pd[gb_cols + ["w_idx", "role", "episode_id"]],
        on=gb_cols + ["w_idx"],
        how="left",
        copy=False,
    )

    return panel_labeled, episodes_pd, windows_long_pd


# ----------------------------- CLI opcional ----------------------------- #
def _read_any_to_pandas(path: Union[str, Path]) -> pd.DataFrame:
    """
    Lee CSV o Parquet en pandas. Si hay cuDF disponible, intenta usar IO GPU y devuelve pandas.
    """
    import os
    p = Path(path)
    ext = p.suffix.lower()
    try:
        import cudf  # type: ignore
        if ext in (".parquet", ".pq"):
            return cudf.read_parquet(str(p)).to_pandas()
        elif ext == ".csv":
            return cudf.read_csv(str(p)).to_pandas()
    except Exception:
        pass  # fallback a pandas

    if ext in (".parquet", ".pq"):
        return pd.read_parquet(p)
    elif ext == ".csv":
        return pd.read_csv(p)
    else:
        raise ValueError("Formato no soportado. Usa .csv o .parquet")


def _save_df(df: pd.DataFrame, path: Union[str, Path]) -> None:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        df.to_parquet(p, index=False)
    elif p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    else:
        # por defecto parquet
        p = p.with_suffix(".parquet")
        df.to_parquet(p, index=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Construye diseño de tratamiento (GPU/cuDF) y guarda outputs."
    )
    parser.add_argument("--input", required=True, help="Ruta a CSV/Parquet con el panel.")
    parser.add_argument(
        "--treatment-col", required=True, help="Nombre de la columna binaria de tratamiento."
    )
    parser.add_argument(
        "--groupby",
        nargs="+",
        default=["store_nbr", "item_nbr"],
        help="Columnas de agrupación (por defecto: store_nbr item_nbr).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directorio de salida. Por defecto, el del archivo de entrada.",
    )
    parser.add_argument(
        "--prefix",
        default="treatment_design",
        help="Prefijo para los archivos de salida (sin extensión).",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Formato de salida (parquet recomendado).",
    )

    args = parser.parse_args()

    panel_pd = _read_any_to_pandas(args.input)

    panel_labeled, episodes, windows_long = build_treatment_design_gpu(
        panel_pd, args.treatment_col, args.groupby
    )

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.input).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = ".parquet" if args.format == "parquet" else ".csv"
    _save_df(panel_labeled, out_dir / f"{args.prefix}_panel_labeled{ext}")
    _save_df(episodes, out_dir / f"{args.prefix}_episodes{ext}")
    _save_df(windows_long, out_dir / f"{args.prefix}_windows_long{ext}")

    print("✅ Hecho.")
    print(f"  panel_labeled -> {out_dir / f'{args.prefix}_panel_labeled{ext}'}")
    print(f"  episodes      -> {out_dir / f'{args.prefix}_episodes{ext}'}")
    print(f"  windows_long  -> {out_dir / f'{args.prefix}_windows_long{ext}'}")


if __name__ == "__main__":
    main()