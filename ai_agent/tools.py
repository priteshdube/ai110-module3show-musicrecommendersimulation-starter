"""
tools.py — Tool functions and Gemini tool schema definitions.

Each function here is callable by the Gemini agent during the agentic loop.
The TOOL_DEFINITIONS object at the bottom is passed directly to the Gemini API
so Gemini knows what tools exist, what parameters they take, and when to use them.
"""

import os
import sys
import json
from typing import List, Dict, Optional

from google.genai import types

# Allow imports from the project root (src/recommender.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.recommender import recommend_songs


# ── Tool functions ─────────────────────────────────────────────────────────────

def get_catalog_summary(songs: List[Dict]) -> Dict:
    """
    Returns all unique genres, moods, and total count from the song catalog.
    Gemini calls this first to know what values are valid before choosing parameters.
    """
    genres = sorted(set(s["genre"] for s in songs))
    moods  = sorted(set(s["mood"]  for s in songs))
    return {
        "genres":      genres,
        "moods":       moods,
        "total_songs": len(songs),
    }


def get_recommendations(
    songs: List[Dict],
    favorite_genre: str,
    favorite_mood: str,
    target_energy: float,
    likes_acoustic: bool,
    k: int = 5,
) -> List[Dict]:
    """
    Scores every song against the given preferences and returns the top k matches.
    Wraps the existing recommend_songs() function from src/recommender.py.
    """
    # likes_acoustic maps to a target_acousticness value (same as Recommender class)
    target_acousticness = 0.75 if likes_acoustic else 0.25

    user_prefs = {
        "favorite_genre":      favorite_genre,
        "favorite_mood":       favorite_mood,
        "target_energy":       target_energy,
        "target_acousticness": target_acousticness,
    }

    results = recommend_songs(user_prefs, songs, k=k)

    return [
        {
            "rank":        i + 1,
            "id":          song["id"],
            "title":       song["title"],
            "artist":      song["artist"],
            "genre":       song["genre"],
            "mood":        song["mood"],
            "energy":      song["energy"],
            "acousticness": song["acousticness"],
            "score":       round(score, 3),
            "explanation": explanation,
        }
        for i, (song, score, explanation) in enumerate(results)
    ]


def filter_songs_by_attribute(
    songs: List[Dict],
    attribute: str,
    value: Optional[str] = None,
    energy_min: Optional[float] = None,
    energy_max: Optional[float] = None,
) -> List[Dict]:
    """
    Filters the catalog by a single attribute so Gemini can explore subsets
    before deciding on final recommendation parameters.

    attribute can be:
        "genre"        — exact genre match (case-insensitive)
        "mood"         — exact mood match (case-insensitive)
        "energy_range" — songs whose energy is between energy_min and energy_max
    """
    if attribute == "genre" and value:
        filtered = [s for s in songs if s["genre"].lower() == value.lower()]
    elif attribute == "mood" and value:
        filtered = [s for s in songs if s["mood"].lower() == value.lower()]
    elif attribute == "energy_range":
        lo = energy_min if energy_min is not None else 0.0
        hi = energy_max if energy_max is not None else 1.0
        filtered = [s for s in songs if lo <= s["energy"] <= hi]
    else:
        filtered = songs

    # Return a compact view — Gemini doesn't need every field to make decisions
    return [
        {
            "id":     s["id"],
            "title":  s["title"],
            "artist": s["artist"],
            "genre":  s["genre"],
            "mood":   s["mood"],
            "energy": s["energy"],
        }
        for s in filtered
    ]


def get_song_details(songs: List[Dict], song_id: int) -> Dict:
    """Returns the full record for a single song by its integer ID (1–20)."""
    for song in songs:
        if int(song["id"]) == int(song_id):
            return dict(song)
    return {"error": f"No song found with id={song_id}"}


# ── Gemini tool schema definitions ────────────────────────────────────────────
# This is passed to the Gemini API via GenerateContentConfig(tools=[TOOL_DEFINITIONS]).
# Gemini reads the name + description of each FunctionDeclaration to decide
# which tool to call, and reads the parameters schema to know what args to pass.

TOOL_DEFINITIONS = types.Tool(
    function_declarations=[

        types.FunctionDeclaration(
            name="get_catalog_summary",
            description=(
                "Returns all unique genres and moods available in the song catalog, "
                "plus the total number of songs. ALWAYS call this first before calling "
                "get_recommendations, so you know which genre and mood values are valid."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),

        types.FunctionDeclaration(
            name="get_recommendations",
            description=(
                "Scores every song in the catalog against the given user preferences "
                "and returns the top k matches. Uses a weighted scoring algorithm: "
                "genre match (+2.0), mood match (+1.5), energy proximity (0–4.0), "
                "acousticness proximity (0–2.0). Max possible score is 9.5."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "favorite_genre": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "The genre that best fits the situation. Must be one of the "
                            "genres returned by get_catalog_summary."
                        ),
                    ),
                    "favorite_mood": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "The mood that best fits the situation. Must be one of the "
                            "moods returned by get_catalog_summary."
                        ),
                    ),
                    "target_energy": types.Schema(
                        type=types.Type.NUMBER,
                        description=(
                            "Target energy level from 0.0 (very calm, background music) "
                            "to 1.0 (very intense, workout music)."
                        ),
                    ),
                    "likes_acoustic": types.Schema(
                        type=types.Type.BOOLEAN,
                        description=(
                            "True if the situation calls for acoustic/organic sounds "
                            "(e.g. studying, coffee shop, nature). False for electronic/"
                            "produced sounds (e.g. gym, clubbing, gaming)."
                        ),
                    ),
                    "k": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of songs to return. Defaults to 5.",
                    ),
                },
                required=["favorite_genre", "favorite_mood", "target_energy", "likes_acoustic"],
            ),
        ),

        types.FunctionDeclaration(
            name="filter_songs_by_attribute",
            description=(
                "Filters the catalog by a specific attribute. Use this to explore "
                "what songs exist within a genre, mood, or energy range before "
                "deciding on final recommendation parameters."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "attribute": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "Which attribute to filter by. One of: "
                            "'genre', 'mood', or 'energy_range'."
                        ),
                    ),
                    "value": types.Schema(
                        type=types.Type.STRING,
                        description="The value to match when filtering by 'genre' or 'mood'.",
                    ),
                    "energy_min": types.Schema(
                        type=types.Type.NUMBER,
                        description="Minimum energy (0.0–1.0) when attribute is 'energy_range'.",
                    ),
                    "energy_max": types.Schema(
                        type=types.Type.NUMBER,
                        description="Maximum energy (0.0–1.0) when attribute is 'energy_range'.",
                    ),
                },
                required=["attribute"],
            ),
        ),

        types.FunctionDeclaration(
            name="get_song_details",
            description=(
                "Returns the full details of a single song by its integer ID (1–20). "
                "Use this to inspect a specific song's energy, acousticness, valence, "
                "tempo, and danceability before including it in a playlist."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "song_id": types.Schema(
                        type=types.Type.INTEGER,
                        description="The integer ID of the song to retrieve (1–20).",
                    ),
                },
                required=["song_id"],
            ),
        ),

    ]
)
