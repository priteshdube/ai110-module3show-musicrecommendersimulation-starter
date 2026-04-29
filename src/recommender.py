import csv
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

# ------------------------------------------------------------
# Taste profiles: the target values the recommender scores
# each song against. Categorical fields are used as filters;
# numerical fields drive the Gaussian proximity score.
# ------------------------------------------------------------

# Profile 1 — Chill Lofi
USER_TASTE_PROFILE: Dict = {
    # --- categorical (used as filters) ---
    "favorite_genre": "lofi",       # primary genre anchor
    "favorite_mood":  "chill",      # primary mood anchor
    "likes_acoustic": True,         # prefers organic/acoustic over electronic

    # --- numerical targets (all 0.0–1.0) ---
    "target_energy":       0.40,    # low energy, background-friendly
    "target_valence":      0.60,    # mildly positive, not too dark or upbeat
    "target_danceability": 0.60,    # light groove, not dancefloor
    "target_acousticness": 0.75,    # strong lean toward acoustic sound

    # --- tempo (raw BPM, normalized before scoring) ---
    "target_tempo_bpm":    80,      # slow to mid tempo
}

# Profile 2 — High-Energy Pop
HIGH_ENERGY_POP_PROFILE: Dict = {
    # --- categorical (used as filters) ---
    "favorite_genre": "pop",        # mainstream pop anchor
    "favorite_mood":  "energetic",  # upbeat, hype mood
    "likes_acoustic": False,        # prefers produced/electronic sound

    # --- numerical targets (all 0.0–1.0) ---
    "target_energy":       0.90,    # very high energy
    "target_valence":      0.85,    # bright and positive
    "target_danceability": 0.90,    # highly danceable
    "target_acousticness": 0.10,    # minimal acoustic, heavily produced

    # --- tempo (raw BPM, normalized before scoring) ---
    "target_tempo_bpm":    128,     # fast, club/dance tempo
}

# Profile 3 — Deep Intense Rock
DEEP_INTENSE_ROCK_PROFILE: Dict = {
    # --- categorical (used as filters) ---
    "favorite_genre": "rock",       # rock/metal anchor
    "favorite_mood":  "intense",    # dark and driving mood
    "likes_acoustic": False,        # prefers electric/distorted sound

    # --- numerical targets (all 0.0–1.0) ---
    "target_energy":       0.88,    # high, aggressive energy
    "target_valence":      0.25,    # dark and serious tone
    "target_danceability": 0.35,    # not focused on danceability
    "target_acousticness": 0.15,    # very little acoustic texture

    # --- tempo (raw BPM, normalized before scoring) ---
    "target_tempo_bpm":    145,     # fast and driving
}

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        """Initializes the recommender with a catalog of Song objects."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Returns the top k Song objects best matching the given UserProfile."""
        user_prefs = {
            "favorite_genre":    user.favorite_genre,
            "favorite_mood":     user.favorite_mood,
            "target_energy":     user.target_energy,
            "target_acousticness": 0.75 if user.likes_acoustic else 0.25,
        }
        scored = []
        for song in self.songs:
            song_dict = {
                "genre":        song.genre,
                "mood":         song.mood,
                "energy":       song.energy,
                "acousticness": song.acousticness,
            }
            score, _ = score_song(user_prefs, song_dict)
            scored.append((song, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Returns a human-readable score breakdown for a single song against the given UserProfile."""
        user_prefs = {
            "favorite_genre":    user.favorite_genre,
            "favorite_mood":     user.favorite_mood,
            "target_energy":     user.target_energy,
            "target_acousticness": 0.75 if user.likes_acoustic else 0.25,
        }
        song_dict = {
            "genre":        song.genre,
            "mood":         song.mood,
            "energy":       song.energy,
            "acousticness": song.acousticness,
        }
        score, reasons = score_song(user_prefs, song_dict)
        return f"Score {score:.3f}: " + "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """Reads songs.csv and returns a list of dicts with numerical fields cast to float."""
    print(f"Loading songs from {csv_path}...")
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["energy"]       = float(row["energy"])
            row["tempo_bpm"]    = float(row["tempo_bpm"])
            row["valence"]      = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            songs.append(row)
    return songs

def _gaussian(song_val: float, target_val: float, sigma: float, max_pts: float) -> float:
    """Returns a proximity score that peaks at max_pts when song_val equals target_val and decays with distance."""
    return max_pts * math.exp(-((song_val - target_val) ** 2) / (2 * sigma ** 2))

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Scores a song against user preferences and returns a (score, reasons) tuple."""
    score = 0.0
    reasons = []

    if song["genre"].lower() == user_prefs["favorite_genre"].lower():
        score += 2.0
        reasons.append(f"genre match (+2.0)")
    else:
        reasons.append(f"genre mismatch ({song['genre']} != {user_prefs['favorite_genre']}, +0.0)")

    if song["mood"].lower() == user_prefs["favorite_mood"].lower():
        score += 1.5
        reasons.append(f"mood match (+1.5)")
    else:
        reasons.append(f"mood mismatch ({song['mood']} != {user_prefs['favorite_mood']}, +0.0)")

    energy_pts = _gaussian(song["energy"], user_prefs["target_energy"], sigma=0.20, max_pts=4.0)
    score += energy_pts
    reasons.append(f"energy proximity (+{energy_pts:.2f}/4.00)")

    acoustic_pts = _gaussian(song["acousticness"], user_prefs["target_acousticness"], sigma=0.25, max_pts=2.0)
    score += acoustic_pts
    reasons.append(f"acousticness proximity (+{acoustic_pts:.2f}/2.00)")

    return (round(score, 3), reasons)

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Scores all songs, sorts by score descending, and returns the top k as (song, score, explanation) tuples."""
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        scored.append((song, score, "; ".join(reasons)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
