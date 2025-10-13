# src/treatment/bluid_treatment_debug.py
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ajustar el sys.path para encontrar config y utils
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.conf import config
from src.utils import utils


def build_treatment_design(panel: pd.DataFrame, treatment_col: str, groupby=["store_nbr", "item_nbr"]):
    """
    Construye episodios, ventanas y etiqueta el panel con roles de tratamiento.
    """

    print(f"\n[DEBUG] >>> Entrando a build_treatment_design <<<")
    print(f"[DEBUG] panel shape={panel.shape}, cols={list(panel.columns)[:10]}...")
    print(f"[DEBUG] treatment_col usado: {treatment_col}")

    # Copia panel para evitar mutaciones
    panel = panel.copy()
    panel["treatment_used"] = panel[treatment_col]

    # Variables de salida
    episodes = []
    windows_long = []

    # Iterar por cada grupo (tienda, item)
    for keys, df_g in panel.groupby(groupby):
        print(f"\n[DEBUG] Grupo={keys}, df_g.shape={df_g.shape}")
        print(f"[DEBUG] w_idx head={df_g['w_idx'].tolist()[:5]}")
        print(f"[DEBUG] treatment_used sum={df_g['treatment_used'].sum()}")

        w = df_g["w_idx"].astype(int).to_numpy()
        z = df_g["treatment_used"].to_numpy()

        # Sanity check
        if len(w) != len(z):
            print(f"[ERROR] Desajuste grupo={keys}, len(w)={len(w)}, len(z)={len(z)}")

        # Detectar episodios de tratamiento
        in_episode = False
        s_w = None
        for idx in range(len(df_g)):
            if z[idx] == 1 and not in_episode:
                s_w = w[idx]
                in_episode = True
            if z[idx] == 0 and in_episode:
                e_w = w[idx - 1]
                ep = {"episode_id": len(episodes), "group": keys, "s_w": s_w, "e_w": e_w}
                episodes.append(ep)
                print(f"[DEBUG] Episodio detectado: {ep}")
                in_episode = False

        # Si terminó en episodio abierto
        if in_episode:
            e_w = w[-1]
            ep = {"episode_id": len(episodes), "group": keys, "s_w": s_w, "e_w": e_w}
            episodes.append(ep)
            print(f"[DEBUG] Episodio final detectado: {ep}")

        # Construir ventanas por episodio
        for ep in episodes:
            mask_ep = (df_g["w_idx"] >= ep["s_w"]) & (df_g["w_idx"] <= ep["e_w"])
            df_ep = df_g[mask_ep]

            w = df_ep["w_idx"].astype(int).to_numpy()
            z = df_ep["treatment_used"].to_numpy()

            print(f"[DEBUG] Ventana episodio {ep['episode_id']} - df_ep.shape={df_ep.shape}")
            print(f"[DEBUG] w.shape={w.shape}, z.shape={z.shape}")

            if len(w) != len(z):
                print(f"[ERROR] Desajuste episodio={ep['episode_id']} len(w)={len(w)}, len(z)={len(z)}")
                continue

            # Línea problemática vigilada
            mask = (z == 1)
            if len(mask) != len(w):
                print(f"[ERROR] Boolean mask mismatch episodio={ep['episode_id']}: len(w)={len(w)}, len(mask)={len(mask)}")
                treat_weeks_set = set()
            else:
                treat_weeks_set = set(w[mask])

            wins = {"treat": (ep["s_w"], ep["e_w"])}
            wins_sets = {role: utils.expand_range_to_set(rng) for role, rng in wins.items()}
            wins_sets["treat"] = wins_sets["treat"].intersection(treat_weeks_set)

            for role, weeks in wins_sets.items():
                for wi in weeks:
                    windows_long.append(
                        {"episode_id": ep["episode_id"], "group": ep["group"], "w_idx": wi, "role": role}
                    )

    # Convertir salidas a DataFrame
    episodes = pd.DataFrame(episodes)
    windows_long = pd.DataFrame(windows_long)

    print(f"\n[DEBUG] Total episodios detectados: {len(episodes)}")
    print(f"[DEBUG] windows_long shape={windows_long.shape}")
    print(windows_long.head())

    # Etiquetar panel
    panel_labeled = panel.merge(
        windows_long, how="left", on=groupby + ["w_idx"], suffixes=("", "_role")
    )
    print(f"[DEBUG] panel_labeled shape={panel_labeled.shape}")
    print(panel_labeled.head())

    return panel_labeled, episodes, windows_long
