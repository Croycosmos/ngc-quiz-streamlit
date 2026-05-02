#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import re
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
IMAGE_DIR = BASE_DIR / "wiki_images"
CACHE_PATH = BASE_DIR / "ngc_resolved_cache.json"

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]


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
            "constellation_catalog": entry.get("constellation_catalog"),
        })

    return pd.DataFrame(rows)


def init_state(df):
    if "score" not in st.session_state:
        st.session_state.score = 0

    if "played" not in st.session_state:
        st.session_state.played = 0

    if "current_idx" not in st.session_state:
        st.session_state.current_idx = random.randrange(len(df))

    if "answered" not in st.session_state:
        st.session_state.answered = False

    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def next_object(df):
    st.session_state.current_idx = random.randrange(len(df))
    st.session_state.answered = False
    st.session_state.last_result = None
    st.session_state.answer = ""


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


def main():
    st.set_page_config(
        page_title="NGC Quiz",
        page_icon="🌌",
        layout="centered",
    )

    st.title("NGC Quiz")
    st.caption("Images Wikipedia locales · réponse par nom NGC ou numéro seul")

    df = load_objects()

    if len(df) == 0:
        st.error("Aucune image trouvée. Vérifie wiki_images/ et ngc_resolved_cache.json.")
        return

    init_state(df)

    with st.sidebar:
        st.header("Réglages")

        mode = st.radio(
            "Mode",
            ["Quiz", "Révision"],
            index=0,
        )

        show_hints = st.checkbox("Afficher les indices", value=True)

        mag_limit = st.slider(
            "Magnitude max",
            min_value=8.0,
            max_value=14.0,
            value=14.0,
            step=0.5,
        )

        filtered = df.copy()
        filtered["mag_numeric"] = pd.to_numeric(filtered["mag"], errors="coerce")
        filtered = filtered[
            filtered["mag_numeric"].isna()
            | (filtered["mag_numeric"] <= mag_limit)
        ]

        st.write(f"Objets disponibles : {len(filtered)}")
        st.write(f"Score : {st.session_state.score}/{st.session_state.played}")

        if st.button("Nouvel objet"):
            next_object(filtered)
            st.rerun()

    if len(filtered) == 0:
        st.warning("Aucun objet avec ce filtre de magnitude.")
        return

    if st.session_state.current_idx >= len(filtered):
        st.session_state.current_idx = random.randrange(len(filtered))

    row = filtered.iloc[st.session_state.current_idx]

    image_path = Path(row["image_path"])

    st.image(str(image_path), use_container_width=True)

    if mode == "Révision":
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
            st.write(
                f"**Constellation** : "
                f"{row['constellation_catalog'] if row['constellation_catalog'] else 'non disponible'}"
            )

    answer = st.text_input(
        "Nom de l’objet",
        key="answer",
        placeholder="Ex : NGC 5775 ou 5775",
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Valider", type="primary"):
            if answer_matches(answer, row["name"]):
                st.session_state.score += 1
                st.session_state.last_result = "correct"
            else:
                st.session_state.last_result = "wrong"

            st.session_state.played += 1
            st.session_state.answered = True
            st.rerun()

    with col2:
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

    with st.expander("Liste des objets disponibles"):
        st.dataframe(
            filtered[["name", "mag", "size_arcmin", "ra_deg", "dec_deg", "image_path"]],
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
