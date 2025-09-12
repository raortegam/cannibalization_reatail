import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.utils import add_week_start, safe_bool_to_int, ensure_columns, cast_if_exists
from src.conf import config

# feature_engineering.py
from typing import List, Dict

# ====================================================
# 1) Agregación semanal de ventas y variables clave
# ====================================================

def aggregate_weekly_sales(
    ventas: pd.DataFrame,
    theta_semana: float = config.THETA_DIAS_PROMO_SEMANA
) -> pd.DataFrame:
    """
    Crea Y_{isw}, y_{isw}, δ_{isw} y ˜P_{isw} a partir de ventas diarias.
    """
    # Tipos eficientes
    ventas = ventas.copy()
    ventas["store_nbr"] = ventas["store_nbr"].astype("int32")
    ventas["item_nbr"]  = ventas["item_nbr"].astype("int32")
    ventas["unit_sales"] = pd.to_numeric(ventas["unit_sales"], errors="coerce").astype("float32")

    # onpromotion → {0,1}
    ventas["onpromotion"] = safe_bool_to_int(ventas["onpromotion"])

    # Semana ISO (lunes a domingo)
    ventas = add_week_start(ventas, "date", "week_start")

    # Agregación semanal por (tienda, ítem, semana)
    agg_week = (
        ventas
        .groupby(["store_nbr", "item_nbr", "week_start"], as_index=False)
        .agg(
            Y_isw=("unit_sales", "sum"),          # ∑_t Y_{ist}
            dias_semana=("date", "nunique"),      # |ι(w)|
            promo_dias=("onpromotion", "sum")     # ∑_t P_{ist}
        )
    )

    # δ_{isw} y ˜P_{isw}
    agg_week["delta_isw"] = (agg_week["promo_dias"] / agg_week["dias_semana"]).astype("float32")
    agg_week["Ptilde_isw"] = (agg_week["delta_isw"] >= theta_semana).astype("int8")

    # y_{isw} = log(1 + Y_{isw})
    agg_week["y_isw"] = np.log1p(agg_week["Y_isw"]).astype("float32")

    return agg_week


# ====================================================
# 2) Metadatos de ítems y tiendas
# ====================================================

def merge_item_store_meta(agg_week: pd.DataFrame, items: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    items = items.copy()
    stores = stores.copy()

    items["item_nbr"] = items["item_nbr"].astype("int32")
    items["perishable"] = items["perishable"].astype("int8")
    stores["store_nbr"] = stores["store_nbr"].astype("int32")

    panel = (
        agg_week
        .merge(items, on="item_nbr", how="left")
        .merge(stores, on="store_nbr", how="left")
    )
    return panel


# ====================================================
# 3) Festivos por tienda-semana: H_nat, H_reg, H_loc
# ====================================================

def build_holiday_features(hol: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """
    Construye H_nat, H_reg, H_loc (conteo de eventos no 'Work Day' por tienda-semana).
    - National: se cruza con todas las tiendas (aplican a todas).
    - Regional: join por estado.
    - Local:    join por ciudad.
    """
    hol = hol.copy()
    stores = stores.copy()

    # Nos quedamos con no-transferidos (coherente con tu código original)
    if "transferred" in hol.columns:
        hol = hol[hol["transferred"] == False].copy()

    # Opcional: excluir 'Work Day' (días laborables insertados)
    if "type" in hol.columns:
        hol = hol[hol["type"] != "Work Day"].copy()

    # Semana ISO
    hol = add_week_start(hol, "date", "week_start")

    # National → cross-join a todas las tiendas
    hol_nat = (
        hol[hol["locale"] == "National"]
        .assign(key=1)
        .merge(stores[["store_nbr"]].assign(key=1), on="key")
        .drop(columns="key")
    )
    h_nat = (
        hol_nat.groupby(["store_nbr", "week_start"], as_index=False)
        .size().rename(columns={"size": "H_nat"})
    )

    # Regional → join por estado
    if {"locale_name", "state"}.issubset(hol.columns.union(stores.columns)):
        hol_reg = hol[hol["locale"] == "Regional"].merge(
            stores[["state", "store_nbr"]],
            left_on="locale_name",
            right_on="state",
            how="left"
        )
        h_reg = (
            hol_reg.groupby(["store_nbr", "week_start"], as_index=False)
            .size().rename(columns={"size": "H_reg"})
        )
    else:
        h_reg = pd.DataFrame(columns=["store_nbr", "week_start", "H_reg"])

    # Local → join por ciudad
    if {"locale_name", "city"}.issubset(hol.columns.union(stores.columns)):
        hol_loc = hol[hol["locale"] == "Local"].merge(
            stores[["city", "store_nbr"]],
            left_on="locale_name",
            right_on="city",
            how="left"
        )
        h_loc = (
            hol_loc.groupby(["store_nbr", "week_start"], as_index=False)
            .size().rename(columns={"size": "H_loc"})
        )
    else:
        h_loc = pd.DataFrame(columns=["store_nbr", "week_start", "H_loc"])

    # Unión de features
    H = (
        h_nat
        .merge(h_reg, on=["store_nbr", "week_start"], how="outer")
        .merge(h_loc, on=["store_nbr", "week_start"], how="outer")
        .fillna(0.0)
    )
    # Tipos
    H["store_nbr"] = H["store_nbr"].astype("int32")
    H["H_nat"] = H["H_nat"].astype("float32")
    H["H_reg"] = H["H_reg"].astype("float32")
    H["H_loc"] = H["H_loc"].astype("float32")
    return H


# ====================================================
# 4) Controles: tráfico (transacciones) y petróleo
# ====================================================

def build_transactions_features(trans: pd.DataFrame) -> pd.DataFrame:
    trans = trans.copy()
    trans = add_week_start(trans, "date", "week_start")
    trans["store_nbr"] = trans["store_nbr"].astype("int32")
    trans_week = (
        trans
        .groupby(["store_nbr", "week_start"], as_index=False)
        .agg(F_sw=("transactions", "sum"))
    )
    trans_week["log_F_sw"] = np.log1p(trans_week["F_sw"]).astype("float32")
    return trans_week

def build_oil_features(oil: pd.DataFrame) -> pd.DataFrame:
    oil = oil.sort_values("date").copy()
    oil["dcoilwtico"] = pd.to_numeric(oil["dcoilwtico"], errors="coerce")
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill()
    oil = add_week_start(oil, "date", "week_start")
    oil_week = (
        oil.groupby("week_start", as_index=False)
           .agg(O_w=("dcoilwtico", "mean"))
    )
    oil_week["O_w"] = oil_week["O_w"].astype("float32")
    return oil_week


# ====================================================
# 5) Choque terremoto, Fourier y lags
# ====================================================

def add_earthquake_shock(panel: pd.DataFrame, eq_date: pd.Timestamp) -> pd.DataFrame:
    panel = panel.copy()
    eq_week = eq_date - pd.to_timedelta(eq_date.weekday(), unit="D")
    panel["eq_shock"] = (panel["week_start"] == eq_week).astype("int8")
    return panel

def add_fourier_terms(panel: pd.DataFrame, k: int = config.FOURIER_K) -> pd.DataFrame:
    panel = panel.copy()
    # Índice semanal global
    weeks_sorted = np.sort(panel["week_start"].unique())
    week_to_index = {w: i for i, w in enumerate(weeks_sorted)}
    panel["w_idx"] = panel["week_start"].map(week_to_index).astype("int32")

    for kk in range(1, k + 1):
        panel[f"fourier_s{kk}"] = np.sin(2 * np.pi * kk * panel["w_idx"] / 52).astype("float32")
        panel[f"fourier_c{kk}"] = np.cos(2 * np.pi * kk * panel["w_idx"] / 52).astype("float32")
    return panel

def add_lags(panel: pd.DataFrame, lags: List[int] = config.LAGS) -> pd.DataFrame:
    panel = panel.sort_values(["store_nbr", "item_nbr", "week_start"]).copy()
    for L in lags:
        panel[f"lag{L}"] = (
            panel.groupby(["store_nbr", "item_nbr"], sort=False)["y_isw"].shift(L)
        ).astype("float32")
    return panel


# ====================================================
# 6) Orquestador y columnas finales
# ====================================================

def finalize_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena columnas, asegura presencia y castea tipos razonables.
    """
    cols_order = [
        "store_nbr", "item_nbr", "week_start",
        "Y_isw", "y_isw", "dias_semana", "promo_dias", "delta_isw", "Ptilde_isw",
        "family", "class", "perishable", "city", "state", "type", "cluster",
        "F_sw", "log_F_sw", "O_w", "H_nat", "H_reg", "H_loc", "eq_shock",
        "w_idx"
    ] + [f"fourier_s{k}" for k in range(1, config.FOURIER_K + 1)] \
      + [f"fourier_c{k}" for k in range(1, config.FOURIER_K + 1)] \
      + [f"lag{L}" for L in config.LAGS]

    panel = ensure_columns(panel, cols_order)
    panel = panel[cols_order].reset_index(drop=True)

    # Casts ligeros cuando existan
    panel = cast_if_exists(panel, {
        "store_nbr": "int32",
        "item_nbr": "int32",
        "perishable": "int8",
        "Ptilde_isw": "int8",
    })
    # Flotantes clave
    for c in ["Y_isw", "y_isw", "delta_isw", "log_F_sw", "O_w"]:
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")
    return panel

def build_full_panel(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Ejecuta de punta a punta la construcción del panel semanal con controles y features.
    """
    ventas = raw["ventas"]
    items = raw["items"]
    stores = raw["stores"]
    hol = raw["hol"]
    trans = raw["trans"]
    oil = raw["oil"]

    # 1) Agregación semanal de ventas
    agg_week = aggregate_weekly_sales(ventas, theta_semana=config.THETA_DIAS_PROMO_SEMANA)

    # 2) Metadatos item & store
    panel = merge_item_store_meta(agg_week, items, stores)

    # 3) Festivos por tienda-semana
    H = build_holiday_features(hol, stores)
    panel = panel.merge(H, on=["store_nbr", "week_start"], how="left")

    # 4) Controles de tráfico y petróleo
    trans_week = build_transactions_features(trans)
    panel = panel.merge(trans_week, on=["store_nbr", "week_start"], how="left")

    oil_week = build_oil_features(oil)
    panel = panel.merge(oil_week, on="week_start", how="left")

    # 5) Terremoto, Fourier, Lags
    panel = add_earthquake_shock(panel, pd.to_datetime(config.EQ_DATE))
    panel = add_fourier_terms(panel, k=config.FOURIER_K)
    panel = add_lags(panel, lags=config.LAGS)

    # 6) Columnas finales
    panel = finalize_columns(panel)
    return panel