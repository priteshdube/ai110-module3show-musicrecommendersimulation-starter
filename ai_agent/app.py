

import os
import sys

import streamlit as st
import pandas as pd

# Ensure project root is on the path so we can import src/ and ai_agent/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.recommender import load_songs
from ai_agent.agent import plan_playlist

# ── Load .env if python-dotenv is installed (optional convenience) ─────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Playlist Planner",
    layout="wide",
)


# ── Load catalog once and cache it ────────────────────────────────────────────
@st.cache_data
def load_catalog() -> list:
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
    return load_songs(os.path.normpath(csv_path))


songs = load_catalog()


# ── Sidebar — catalog viewer ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Song Catalog")
    st.caption(f"{len(songs)} songs available")

    df = pd.DataFrame(songs)
    display_cols = ["title", "artist", "genre", "mood", "energy", "acousticness"]
    st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.caption(
        "The agent scores songs using: genre match, mood match, "
        "energy proximity, and acousticness proximity."
    )


# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("AI Playlist Planner")
st.write(
    "Describe what you're doing or how you're feeling — the AI agent will "
    "reason through it step by step and build you a personalized playlist."
)

# Example prompts
st.caption("Try: *late-night study session* · *morning run* · *rainy Sunday* · *pre-game hype*")

situation = st.text_area(
    "What's the vibe?",
    placeholder=(
        'e.g. "I need music for a late-night coding session" or '
        '"Plan an energetic morning workout playlist"'
    ),
    height=110,
)

run_button = st.button(
    "Plan My Playlist",
    type="primary",
    disabled=not situation.strip(),
)

if run_button and situation.strip():
    with st.spinner("Agent is thinking..."):
        try:
            result = plan_playlist(situation.strip(), songs)
            st.divider()
            st.markdown(result)

        except ValueError as exc:
            # Missing API key
            st.error(str(exc))

        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
