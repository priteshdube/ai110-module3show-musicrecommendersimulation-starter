"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs, USER_TASTE_PROFILE


def main() -> None:
    songs = load_songs("data/songs.csv")

    user_prefs = USER_TASTE_PROFILE

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\n" + "=" * 50)
    print("  TOP 5 SONG RECOMMENDATIONS")
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
