#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import re
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "wiki_images"
CACHE_PATH = BASE_DIR / "ngc_resolved_cache.json"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]


# ============================================================
# Texte / réponses
# ============================================================

def normalize_name(text):
    s = str(text).strip().upper()
    s = s.replace("-", " ").replace("_", " ")
    s = " ".join(s.split())

    if s.startswith("NGC") and not s.startswith("NGC "):
        s = s.replace("NGC", "NGC ", 1)

    parts = s.split()

    if len(parts) == 2 and parts[0] == "NGC" and parts[1].isdigit():
        return f"NGC {int(parts[1])}"

    if s.isdigit():
        return str(int(s))

    return s


def object_number(text):
    s = normalize_name(text)
    parts = s.split()

    if len(parts) == 2 and parts[0] == "NGC" and parts[1].isdigit():
        return str(int(parts[1]))

    if s.isdigit():
        return str(int(s))

    return None


def answer_matches(answer, true_name):
    answer_norm = normalize_name(answer)
    true_norm = normalize_name(true_name)

    if answer_norm == true_norm:
        return True

    answer_num = object_number(answer_norm)
    true_num = object_number(true_norm)

    if answer_num is not None and true_num is not None and answer_num == true_num:
        return True

    return SequenceMatcher(None, answer_norm, true_norm).ratio() > 0.88


def ngc_number_from_name(name):
    match = re.search(r"NGC\s*0*([0-9]+)", str(name).upper())
    if match:
        return str(int(match.group(1)))
    return None


def find_image_for_ngc(ngc_num):
    for ext in IMAGE_EXTS:
        path = IMAGE_DIR / f"NGC{ngc_num}{ext}"
        if path.exists():
            return path
    return None


def format_value(value, ndigits=2):
    if value is None:
        return "non disponible"

    try:
        if pd.isna(value):
            return "non disponible"
    except Exception:
        pass

    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return str(value)


# ============================================================
# Données
# ============================================================

@st.cache_data
def load_objects():
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    rows = []

    for entry in cache:
        name = entry.get("name")
        ngc_num = entry.get("ngc") or ngc_number_from_name(name)

        if ngc_num is None:
            continue

        ngc_num = str(int(ngc_num))
        image_path = find_image_for_ngc(ngc_num)

        if image_path is None:
            continue

        constellation = (
            entry.get("constellation_catalog")
            or entry.get("constellation")
            or "non disponible"
        )

        rows.append({
            "name": f"NGC {ngc_num}",
            "ngc": ngc_num,
            "image_path": str(image_path),
            "ra_deg": entry.get("ra_deg"),
            "dec_deg": entry.get("dec_deg"),
            "mag": entry.get("mag"),
            "size_arcmin": entry.get("size_arcmin"),
            "type": entry.get("type"),
            "description": entry.get("description"),
            "constellation": constellation,
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df

    df["mag_numeric"] = pd.to_numeric(df["mag"], errors="coerce")
    df["ra_numeric"] = pd.to_numeric(df["ra_deg"], errors="coerce")
    df["dec_numeric"] = pd.to_numeric(df["dec_deg"], errors="coerce")
    df["size_numeric"] = pd.to_numeric(df["size_arcmin"], errors="coerce")

    return df


# ============================================================
# État Streamlit
# ============================================================

def init_state(df):
    defaults = {
        "score": 0,
        "played": 0,
        "current_idx": random.randrange(len(df)) if len(df) else 0,
        "answered": False,
        "last_result": None,
        "revealed": False,
        "answer": "",
        "mistakes": {},
        "corrects": {},
        "attempts_by_object": {},
        "recent_errors": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_stats():
    st.session_state.score = 0
    st.session_state.played = 0
    st.session_state.mistakes = {}
    st.session_state.corrects = {}
    st.session_state.attempts_by_object = {}
    st.session_state.recent_errors = []
    st.session_state.last_result = None
    st.session_state.answered = False
    st.session_state.revealed = False
    st.session_state.answer = ""


def next_object(df):
    if len(df) == 0:
        return

    st.session_state.current_idx = random.randrange(len(df))
    st.session_state.answered = False
    st.session_state.last_result = None
    st.session_state.revealed = False
    st.session_state.answer = ""


def register_answer(name, is_correct, user_answer):
    st.session_state.played += 1
    st.session_state.attempts_by_object[name] = (
        st.session_state.attempts_by_object.get(name, 0) + 1
    )

    if is_correct:
        st.session_state.score += 1
        st.session_state.corrects[name] = st.session_state.corrects.get(name, 0) + 1
        st.session_state.last_result = "correct"
    else:
        st.session_state.mistakes[name] = st.session_state.mistakes.get(name, 0) + 1
        st.session_state.last_result = "wrong"

        st.session_state.recent_errors.insert(
            0,
            {
                "object": name,
                "answer": user_answer,
            }
        )
        st.session_state.recent_errors = st.session_state.recent_errors[:20]

    st.session_state.answered = True


# ============================================================
# Carte mentale
# ============================================================

def wrap_ra_deg_to_180(ra_deg):
    wrapped = ((ra_deg + 180.0) % 360.0) - 180.0
    return -wrapped


def point_size_from_mag(mag):
    if mag is None or not np.isfinite(mag):
        return 12.0

    m = min(max(float(mag), 6.0), 14.0)
    size = 44.0 - (m - 6.0) * (34.0 / 8.0)
    return max(size, 8.0)


def make_sky_map(df, current_name=None, annotate=15):
    valid = df.dropna(subset=["ra_numeric", "dec_numeric"]).copy()

    if len(valid) == 0:
        return None

    x = np.deg2rad([wrap_ra_deg_to_180(v) for v in valid["ra_numeric"]])
    y = np.deg2rad(valid["dec_numeric"].to_numpy())
    sizes = np.array([point_size_from_mag(m) for m in valid["mag_numeric"]])

    fig = plt.figure(figsize=(10, 5.8))
    ax = fig.add_subplot(111, projection="aitoff")

    ax.scatter(x, y, s=sizes, alpha=0.65)
    ax.grid(True, alpha=0.45)

    ax.set_title(
        f"Carte mentale NGC — {len(valid)} objets",
        fontsize=12,
    )

    tick_positions_deg = np.array(
        [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]
    )
    tick_positions_rad = np.deg2rad(tick_positions_deg)
    tick_labels = [
        "150°", "120°", "90°", "60°", "30°", "0°",
        "330°", "300°", "270°", "240°", "210°"
    ]

    ax.set_xticks(tick_positions_rad)
    ax.set_xticklabels(tick_labels)

    if current_name is not None:
        current = valid[valid["name"] == current_name]
        if len(current) > 0:
            row = current.iloc[0]
            xc = np.deg2rad(wrap_ra_deg_to_180(row["ra_numeric"]))
            yc = np.deg2rad(row["dec_numeric"])

            ax.scatter([xc], [yc], s=160, marker="*", zorder=10)
            ax.text(xc, yc, " " + current_name, fontsize=8, weight="bold")

    if annotate > 0:
        brightest = valid.sort_values("mag_numeric", na_position="last").head(annotate)

        for _, row in brightest.iterrows():
            xa = np.deg2rad(wrap_ra_deg_to_180(row["ra_numeric"]))
            ya = np.deg2rad(row["dec_numeric"])

            ax.text(xa, ya, " " + row["name"], fontsize=7, alpha=0.85)

    plt.tight_layout()
    return fig


# ============================================================
# Filtres
# ============================================================

def build_filtered_df(df, mag_limit, constellation_choice, only_mistakes):
    filtered = df.copy()

    filtered = filtered[
        filtered["mag_numeric"].isna()
        | (filtered["mag_numeric"] <= mag_limit)
    ]

    if constellation_choice != "Toutes":
        filtered = filtered[filtered["constellation"] == constellation_choice]

    if only_mistakes:
        mistake_names = set(st.session_state.mistakes.keys())
        filtered = filtered[filtered["name"].isin(mistake_names)]

    filtered = filtered.reset_index(drop=True)

    return filtered


# ============================================================
# Interface
# ============================================================

def main():
    st.set_page_config(
        page_title="NGC Quiz",
        page_icon="🌌",
        layout="centered",
    )

    st.title("NGC Quiz")
    st.caption("Images Wikipedia · réponse par nom NGC ou numéro seul")

    df = load_objects()

    if len(df) == 0:
        st.error("Aucune image trouvée. Vérifie wiki_images/ et ngc_resolved_cache.json.")
        return

    init_state(df)

    constellations = sorted(
        c for c in df["constellation"].dropna().unique()
        if str(c).strip() and str(c) != "non disponible"
    )

    with st.sidebar:
        st.header("Réglages")

        mode = st.radio(
            "Mode",
            ["Quiz", "Révision"],
            index=0,
        )

        show_hints = st.checkbox("Afficher les indices", value=True)

        reveal_after_answer = st.checkbox(
            "Révéler automatiquement après validation",
            value=True,
        )

        mag_limit = st.slider(
            "Magnitude max",
            min_value=8.0,
            max_value=14.0,
            value=14.0,
            step=0.5,
        )

        constellation_choice = st.selectbox(
            "Constellation",
            ["Toutes"] + constellations,
            index=0,
        )

        only_mistakes = st.checkbox(
            "Réviser seulement les erreurs",
            value=False,
        )

        st.divider()

        st.write(f"Score : {st.session_state.score}/{st.session_state.played}")

        if st.session_state.played > 0:
            accuracy = 100.0 * st.session_state.score / st.session_state.played
            st.write(f"Réussite : {accuracy:.1f} %")

        if st.button("Réinitialiser les stats"):
            reset_stats()
            st.rerun()

    filtered = build_filtered_df(
        df,
        mag_limit=mag_limit,
        constellation_choice=constellation_choice,
        only_mistakes=only_mistakes,
    )

    if len(filtered) == 0:
        st.warning("Aucun objet avec ces filtres.")
        return

    if st.session_state.current_idx >= len(filtered):
        st.session_state.current_idx = random.randrange(len(filtered))

    row = filtered.iloc[st.session_state.current_idx]

    with st.sidebar:
        st.write(f"Objets disponibles : {len(filtered)}")

        if st.button("Nouvel objet"):
            next_object(filtered)
            st.rerun()

    image_path = Path(row["image_path"])

    st.image(str(image_path), use_container_width=True)

    if mode == "Révision" or st.session_state.revealed:
        st.subheader(row["name"])

    if show_hints:
        c1, c2 = st.columns(2)

        with c1:
            st.write(f"**RA** : {format_value(row['ra_deg'], 4)}°")
            st.write(f"**Dec** : {format_value(row['dec_deg'], 4)}°")
            st.write(f"**Magnitude** : {format_value(row['mag'], 2)}")

        with c2:
            st.write(f"**Taille** : {format_value(row['size_arcmin'], 2)} arcmin")
            st.write(f"**Type** : {row['type'] if row['type'] else 'non disponible'}")
            st.write(f"**Constellation** : {row['constellation']}")

    answer = st.text_input(
        "Nom de l’objet",
        key="answer",
        placeholder="Ex : NGC 5775 ou 5775",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Valider", type="primary"):
            is_correct = answer_matches(answer, row["name"])
            register_answer(row["name"], is_correct, answer)

            if reveal_after_answer:
                st.session_state.revealed = True

            st.rerun()

    with col2:
        if st.button("Révéler"):
            st.session_state.revealed = True
            st.rerun()

    with col3:
        if st.button("Suivant"):
            next_object(filtered)
            st.rerun()

    if st.session_state.answered:
        if st.session_state.last_result == "correct":
            st.success(f"Correct : {row['name']}")
        else:
            st.error(f"Incorrect. Réponse : {row['name']}")

        if row["description"]:
            st.write(f"Description : {row['description']}")

    if st.session_state.revealed and not st.session_state.answered:
        st.info(f"Réponse : {row['name']}")

    st.divider()

    with st.expander("Statistiques des erreurs", expanded=False):
        total_errors = sum(st.session_state.mistakes.values())

        st.write(f"Erreurs totales : {total_errors}")

        if total_errors == 0:
            st.write("Aucune erreur pour l’instant.")
        else:
            mistake_df = pd.DataFrame(
                [
                    {
                        "Objet": name,
                        "Erreurs": count,
                        "Tentatives": st.session_state.attempts_by_object.get(name, 0),
                        "Corrects": st.session_state.corrects.get(name, 0),
                    }
                    for name, count in st.session_state.mistakes.items()
                ]
            ).sort_values("Erreurs", ascending=False)

            st.dataframe(mistake_df, use_container_width=True, hide_index=True)

            if len(st.session_state.recent_errors) > 0:
                st.write("Erreurs récentes :")
                recent_df = pd.DataFrame(st.session_state.recent_errors)
                st.dataframe(recent_df, use_container_width=True, hide_index=True)

    with st.expander("Carte mentale du ciel", expanded=False):
        annotate = st.slider(
            "Nombre de labels sur la carte",
            min_value=0,
            max_value=60,
            value=15,
            step=5,
        )

        fig = make_sky_map(
            filtered,
            current_name=row["name"],
            annotate=annotate,
        )

        if fig is None:
            st.write("Pas assez de coordonnées pour tracer la carte.")
        else:
            st.pyplot(fig)
            plt.close(fig)

    with st.expander("Liste des objets disponibles", expanded=False):
        cols = [
            "name",
            "constellation",
            "mag",
            "size_arcmin",
            "ra_deg",
            "dec_deg",
            "image_path",
        ]

        st.dataframe(
            filtered[cols],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()

