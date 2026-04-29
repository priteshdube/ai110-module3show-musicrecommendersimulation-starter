"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

try:
    from recommender import (
        load_songs, recommend_songs,
        USER_TASTE_PROFILE, HIGH_ENERGY_POP_PROFILE, DEEP_INTENSE_ROCK_PROFILE,
    )
except ModuleNotFoundError:
    from src.recommender import (
        load_songs, recommend_songs,
        USER_TASTE_PROFILE, HIGH_ENERGY_POP_PROFILE, DEEP_INTENSE_ROCK_PROFILE,
    )

PROFILES = [
    ("Chill Lofi",        USER_TASTE_PROFILE),
    ("High-Energy Pop",   HIGH_ENERGY_POP_PROFILE),
    ("Deep Intense Rock", DEEP_INTENSE_ROCK_PROFILE),
]


def main() -> None:
    songs = load_songs("data/songs.csv")

    for profile_name, user_prefs in PROFILES:
        recommendations = recommend_songs(user_prefs, songs, k=5)

        print("\n" + "=" * 50)
        print(f"  TOP 5 RECOMMENDATIONS — {profile_name.upper()}")
        print("=" * 50)

        for i, (song, score, explanation) in enumerate(recommendations, start=1):
            print(f"\n#{i}  {song['title']}  —  {song['artist']}")
            print(f"    Genre: {song['genre']}  |  Mood: {song['mood']}  |  Score: {score:.2f} / 9.50")
            print(f"    Why:")
            for reason in explanation.split("; "):
                print(f"      • {reason}")

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
