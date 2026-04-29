import math
import pytest
from src.recommender import Song, UserProfile, Recommender, score_song, _gaussian


# ─────────────────────────────────────────────
# Fixtures & helpers
# ─────────────────────────────────────────────

def make_song(**overrides) -> Song:
    """Build a Song with sensible defaults; override any field via kwargs."""
    defaults = dict(
        id=1,
        title="Default Track",
        artist="Test Artist",
        genre="pop",
        mood="happy",
        energy=0.8,
        tempo_bpm=120.0,
        valence=0.9,
        danceability=0.8,
        acousticness=0.2,
    )
    defaults.update(overrides)
    return Song(**defaults)


def make_user(**overrides) -> UserProfile:
    """Build a UserProfile with sensible defaults; override any field via kwargs."""
    defaults = dict(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    defaults.update(overrides)
    return UserProfile(**defaults)


def make_recommender(*songs: Song) -> Recommender:
    return Recommender(list(songs))


POP_SONG   = make_song(id=1, genre="pop",  mood="happy",  energy=0.8, acousticness=0.2)
LOFI_SONG  = make_song(id=2, genre="lofi", mood="chill",  energy=0.4, acousticness=0.9)
ROCK_SONG  = make_song(id=3, genre="rock", mood="intense", energy=0.9, acousticness=0.1)


# ─────────────────────────────────────────────
# 1. Gaussian helper
# ─────────────────────────────────────────────

class TestGaussian:
    def test_exact_match_returns_max_pts(self):
        assert _gaussian(0.5, 0.5, sigma=0.2, max_pts=4.0) == pytest.approx(4.0)

    def test_far_value_returns_near_zero(self):
        result = _gaussian(0.0, 1.0, sigma=0.2, max_pts=4.0)
        assert result < 0.001

    def test_one_sigma_away_returns_approx_60_pct(self):
        # e^(-0.5) ≈ 0.6065
        result = _gaussian(0.6, 0.8, sigma=0.2, max_pts=4.0)
        assert result == pytest.approx(4.0 * math.exp(-0.5), rel=1e-5)

    def test_symmetric_around_target(self):
        above = _gaussian(0.7, 0.5, sigma=0.2, max_pts=2.0)
        below = _gaussian(0.3, 0.5, sigma=0.2, max_pts=2.0)
        assert above == pytest.approx(below, rel=1e-9)


# ─────────────────────────────────────────────
# 2. score_song — normal cases
# ─────────────────────────────────────────────

class TestScoreSongNormal:
    def _prefs(self, genre="pop", mood="happy", energy=0.8, acousticness=0.25):
        return {
            "favorite_genre": genre,
            "favorite_mood": mood,
            "target_energy": energy,
            "target_acousticness": acousticness,
        }

    def _song(self, genre="pop", mood="happy", energy=0.8, acousticness=0.2):
        return {"genre": genre, "mood": mood, "energy": energy, "acousticness": acousticness}

    def test_perfect_genre_and_mood_match_adds_3_5(self):
        score, _ = score_song(self._prefs(), self._song())
        # genre (+2.0) + mood (+1.5) = 3.5, plus gaussian terms > 0
        assert score > 3.5

    def test_genre_mismatch_no_genre_pts(self):
        score_match, _   = score_song(self._prefs(genre="pop"),  self._song(genre="pop"))
        score_miss, _    = score_song(self._prefs(genre="rock"), self._song(genre="pop"))
        assert score_match - score_miss == pytest.approx(2.0, rel=1e-3)

    def test_mood_mismatch_no_mood_pts(self):
        score_match, _ = score_song(self._prefs(mood="happy"), self._song(mood="happy"))
        score_miss, _  = score_song(self._prefs(mood="sad"),   self._song(mood="happy"))
        assert score_match - score_miss == pytest.approx(1.5, rel=1e-3)

    def test_returns_tuple_of_float_and_list(self):
        result = score_song(self._prefs(), self._song())
        assert isinstance(result, tuple) and len(result) == 2
        score, reasons = result
        assert isinstance(score, float)
        assert isinstance(reasons, list) and len(reasons) == 4

    def test_score_is_non_negative(self):
        score, _ = score_song(self._prefs(genre="jazz", mood="sad"), self._song(genre="pop", mood="happy"))
        assert score >= 0.0

    def test_max_possible_score_near_9_5(self):
        prefs = self._prefs(genre="pop", mood="happy", energy=0.8, acousticness=0.2)
        song  = self._song(genre="pop", mood="happy", energy=0.8, acousticness=0.2)
        score, _ = score_song(prefs, song)
        assert score == pytest.approx(9.5, rel=1e-3)


# ─────────────────────────────────────────────
# 3. score_song — case sensitivity
# ─────────────────────────────────────────────

class TestScoreSongCaseSensitivity:
    def _base_prefs(self):
        return {"favorite_genre": "POP", "favorite_mood": "Happy",
                "target_energy": 0.8, "target_acousticness": 0.25}

    def test_genre_match_is_case_insensitive(self):
        song = {"genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.2}
        _, reasons = score_song(self._base_prefs(), song)
        assert any("genre match" in r for r in reasons)

    def test_mood_match_is_case_insensitive(self):
        song = {"genre": "pop", "mood": "HAPPY", "energy": 0.8, "acousticness": 0.2}
        _, reasons = score_song(self._base_prefs(), song)
        assert any("mood match" in r for r in reasons)


# ─────────────────────────────────────────────
# 4. Edge cases — empty / blank strings
# ─────────────────────────────────────────────

class TestEmptyStrings:
    def test_empty_genre_matches_empty_genre(self):
        """Two empty strings are 'equal' — guard against accidental matches."""
        prefs = {"favorite_genre": "", "favorite_mood": "happy",
                 "target_energy": 0.5, "target_acousticness": 0.5}
        song  = {"genre": "", "mood": "happy", "energy": 0.5, "acousticness": 0.5}
        score, reasons = score_song(prefs, song)
        # Empty string matches empty string; system does not raise
        assert score >= 0.0
        assert any("genre match" in r for r in reasons), (
            "Empty genre vs empty genre should still register as a match — "
            "document this known behavior"
        )

    def test_empty_genre_does_not_match_real_genre(self):
        prefs = {"favorite_genre": "", "favorite_mood": "happy",
                 "target_energy": 0.5, "target_acousticness": 0.5}
        song  = {"genre": "pop", "mood": "happy", "energy": 0.5, "acousticness": 0.5}
        _, reasons = score_song(prefs, song)
        assert any("genre mismatch" in r for r in reasons)


# ─────────────────────────────────────────────
# 5. Adversarial profiles — conflicting preferences
# ─────────────────────────────────────────────

class TestAdversarialProfiles:
    """
    Users with internally contradictory preference combinations.
    The system should not crash; results should be deterministic
    even if the recommendations are semantically strange.
    """

    def test_high_energy_user_who_likes_lofi_genre(self):
        """
        Contradictory: user says favorite_genre='lofi' (typically low energy)
        but target_energy=0.95 (maximum energy).
        No lofi song should fully satisfy this profile.
        """
        user = make_user(favorite_genre="lofi", favorite_mood="chill", target_energy=0.95)
        rec  = make_recommender(POP_SONG, LOFI_SONG, ROCK_SONG)
        results = rec.recommend(user, k=3)
        assert len(results) == 3
        # Lofi song gets genre/mood bonus but huge energy penalty;
        # top result should NOT be a runaway perfect score.
        scores = []
        for song in results:
            user_prefs = {
                "favorite_genre": user.favorite_genre,
                "favorite_mood":  user.favorite_mood,
                "target_energy":  user.target_energy,
                "target_acousticness": 0.75 if user.likes_acoustic else 0.25,
            }
            song_dict = {"genre": song.genre, "mood": song.mood,
                         "energy": song.energy, "acousticness": song.acousticness}
            s, _ = score_song(user_prefs, song_dict)
            scores.append(s)
        assert max(scores) < 9.5, "No song should achieve a perfect score with a contradictory profile"

    def test_acoustic_lover_who_wants_zero_energy(self):
        """target_energy=0.0 is a valid boundary; system should not crash."""
        user = make_user(target_energy=0.0, likes_acoustic=True)
        rec  = make_recommender(POP_SONG, LOFI_SONG)
        results = rec.recommend(user, k=2)
        assert len(results) == 2

    def test_mismatched_genre_and_mood(self):
        """
        'lofi' genre paired with 'aggressive' mood — no real song fits both.
        The recommender should return results without raising.
        """
        user = make_user(favorite_genre="lofi", favorite_mood="aggressive", target_energy=0.5)
        rec  = make_recommender(POP_SONG, LOFI_SONG, ROCK_SONG)
        results = rec.recommend(user, k=3)
        assert len(results) == 3

    def test_all_songs_same_genre_different_mood(self):
        """When genre always matches, mood becomes the differentiator."""
        happy_pop  = make_song(id=10, genre="pop", mood="happy",  energy=0.7, acousticness=0.2)
        sad_pop    = make_song(id=11, genre="pop", mood="sad",    energy=0.7, acousticness=0.2)
        chill_pop  = make_song(id=12, genre="pop", mood="chill",  energy=0.7, acousticness=0.2)
        user = make_user(favorite_genre="pop", favorite_mood="happy", target_energy=0.7)
        rec  = Recommender([happy_pop, sad_pop, chill_pop])
        results = rec.recommend(user, k=3)
        assert results[0].mood == "happy", "Song with matching mood should rank first"

    def test_out_of_range_energy_above_1(self):
        """
        target_energy=2.5 is outside [0, 1] but must not raise an exception.
        All songs score low on energy; the function should still return results.
        """
        user = make_user(target_energy=2.5)
        rec  = make_recommender(POP_SONG, LOFI_SONG)
        results = rec.recommend(user, k=2)
        assert len(results) == 2

    def test_out_of_range_energy_below_0(self):
        """target_energy=-1.0 is outside [0, 1]; must not raise."""
        user = make_user(target_energy=-1.0)
        rec  = make_recommender(POP_SONG, LOFI_SONG)
        results = rec.recommend(user, k=2)
        assert len(results) == 2

    def test_identical_preference_to_existing_profile(self):
        """User whose prefs exactly mirror a song's features should rank that song first."""
        perfect_song = make_song(id=99, genre="pop", mood="happy", energy=0.8, acousticness=0.25)
        other_song   = make_song(id=98, genre="rock", mood="intense", energy=0.1, acousticness=0.9)
        user = make_user(favorite_genre="pop", favorite_mood="happy",
                         target_energy=0.8, likes_acoustic=False)
        rec = Recommender([other_song, perfect_song])
        results = rec.recommend(user, k=2)
        assert results[0].id == 99, "Perfect-match song should be ranked first"


# ─────────────────────────────────────────────
# 6. Recommender — k parameter edge cases
# ─────────────────────────────────────────────

class TestRecommendKEdgeCases:
    def test_k_equals_zero_returns_empty_list(self):
        rec = make_recommender(POP_SONG, LOFI_SONG)
        assert rec.recommend(make_user(), k=0) == []

    def test_k_larger_than_catalog_returns_all_songs(self):
        rec     = make_recommender(POP_SONG, LOFI_SONG)
        results = rec.recommend(make_user(), k=100)
        assert len(results) == 2

    def test_k_equals_1_returns_single_best(self):
        rec     = make_recommender(POP_SONG, LOFI_SONG)
        results = rec.recommend(make_user(favorite_genre="pop", favorite_mood="happy"), k=1)
        assert len(results) == 1
        assert results[0].genre == "pop"

    def test_k_equals_catalog_size_returns_all(self):
        rec     = make_recommender(POP_SONG, LOFI_SONG, ROCK_SONG)
        results = rec.recommend(make_user(), k=3)
        assert len(results) == 3


# ─────────────────────────────────────────────
# 7. Recommender — empty catalog
# ─────────────────────────────────────────────

class TestEmptyCatalog:
    def test_empty_catalog_returns_empty_list(self):
        rec = Recommender([])
        assert rec.recommend(make_user(), k=5) == []

    def test_empty_catalog_k_zero_returns_empty_list(self):
        rec = Recommender([])
        assert rec.recommend(make_user(), k=0) == []


# ─────────────────────────────────────────────
# 8. Recommender — ranking & ordering
# ─────────────────────────────────────────────

class TestRanking:
    def test_results_are_sorted_descending_by_score(self):
        user = make_user(favorite_genre="pop", favorite_mood="happy", target_energy=0.8)
        rec  = make_recommender(POP_SONG, LOFI_SONG, ROCK_SONG)
        results = rec.recommend(user, k=3)

        # Re-score each result and verify descending order
        user_prefs = {
            "favorite_genre": user.favorite_genre,
            "favorite_mood":  user.favorite_mood,
            "target_energy":  user.target_energy,
            "target_acousticness": 0.25,
        }
        scores = []
        for song in results:
            s, _ = score_song(user_prefs, {"genre": song.genre, "mood": song.mood,
                                            "energy": song.energy, "acousticness": song.acousticness})
            scores.append(s)
        assert scores == sorted(scores, reverse=True)

    def test_recommend_is_deterministic(self):
        """Same inputs always produce the same ranking."""
        user = make_user()
        rec  = make_recommender(POP_SONG, LOFI_SONG, ROCK_SONG)
        first  = [s.id for s in rec.recommend(user, k=3)]
        second = [s.id for s in rec.recommend(user, k=3)]
        assert first == second

    def test_tied_scores_return_consistent_order(self):
        """Songs with identical features should appear in a consistent order."""
        song_a = make_song(id=10, genre="jazz", mood="sad", energy=0.5, acousticness=0.5)
        song_b = make_song(id=11, genre="jazz", mood="sad", energy=0.5, acousticness=0.5)
        user   = make_user(favorite_genre="jazz", favorite_mood="sad", target_energy=0.5)
        rec    = Recommender([song_a, song_b])
        r1 = [s.id for s in rec.recommend(user, k=2)]
        r2 = [s.id for s in rec.recommend(user, k=2)]
        assert r1 == r2


# ─────────────────────────────────────────────
# 9. likes_acoustic mapping
# ─────────────────────────────────────────────

class TestAcousticMapping:
    def test_likes_acoustic_true_targets_0_75(self):
        """A song at 0.75 acousticness should score maximum acoustic pts for acoustic lovers."""
        high_acoustic = make_song(acousticness=0.75)
        low_acoustic  = make_song(id=2, acousticness=0.25)
        user = make_user(likes_acoustic=True, favorite_genre="pop", favorite_mood="happy", target_energy=0.8)
        rec  = Recommender([high_acoustic, low_acoustic])
        results = rec.recommend(user, k=2)
        assert results[0].id == high_acoustic.id

    def test_likes_acoustic_false_targets_0_25(self):
        """A song at 0.25 acousticness should score maximum acoustic pts for non-acoustic users."""
        high_acoustic = make_song(acousticness=0.75)
        low_acoustic  = make_song(id=2, acousticness=0.25)
        user = make_user(likes_acoustic=False, favorite_genre="pop", favorite_mood="happy", target_energy=0.8)
        rec  = Recommender([high_acoustic, low_acoustic])
        results = rec.recommend(user, k=2)
        assert results[0].id == low_acoustic.id

    def test_mid_range_acoustic_song_scores_equally_for_both_preferences(self):
        """
        A song at acousticness=0.5 is equidistant from both targets (0.25 and 0.75),
        so both user types should receive the same acousticness score component.
        """
        prefs_acoustic     = {"favorite_genre": "zzz", "favorite_mood": "zzz",
                               "target_energy": 0.5, "target_acousticness": 0.75}
        prefs_non_acoustic = {"favorite_genre": "zzz", "favorite_mood": "zzz",
                               "target_energy": 0.5, "target_acousticness": 0.25}
        song_dict = {"genre": "zzz", "mood": "zzz", "energy": 0.5, "acousticness": 0.5}
        score_a, _ = score_song(prefs_acoustic,     song_dict)
        score_b, _ = score_song(prefs_non_acoustic, song_dict)
        assert score_a == pytest.approx(score_b, rel=1e-9)


# ─────────────────────────────────────────────
# 10. explain_recommendation
# ─────────────────────────────────────────────

class TestExplainRecommendation:
    def test_returns_non_empty_string(self):
        rec  = make_recommender(POP_SONG)
        user = make_user()
        explanation = rec.explain_recommendation(user, POP_SONG)
        assert isinstance(explanation, str) and explanation.strip()

    def test_contains_score_value(self):
        rec  = make_recommender(POP_SONG)
        user = make_user()
        explanation = rec.explain_recommendation(user, POP_SONG)
        assert "Score" in explanation

    def test_explanation_mentions_genre_match(self):
        rec  = make_recommender(POP_SONG)
        user = make_user(favorite_genre="pop")
        explanation = rec.explain_recommendation(user, POP_SONG)
        assert "genre match" in explanation

    def test_explanation_mentions_genre_mismatch(self):
        rec  = make_recommender(LOFI_SONG)
        user = make_user(favorite_genre="rock")
        explanation = rec.explain_recommendation(user, LOFI_SONG)
        assert "genre mismatch" in explanation

    def test_explanation_works_for_non_recommended_song(self):
        """explain_recommendation should work on any song, not just top-k results."""
        rec  = make_recommender(POP_SONG, LOFI_SONG)
        user = make_user(favorite_genre="pop")
        # Explain a song that would rank last
        explanation = rec.explain_recommendation(user, LOFI_SONG)
        assert isinstance(explanation, str) and explanation.strip()


# ─────────────────────────────────────────────
# 11. Single-song catalog
# ─────────────────────────────────────────────

class TestSingleSongCatalog:
    def test_single_song_is_always_top_result(self):
        rec     = make_recommender(POP_SONG)
        results = rec.recommend(make_user(), k=5)
        assert len(results) == 1
        assert results[0].id == POP_SONG.id

    def test_single_song_k_zero_returns_empty(self):
        rec = make_recommender(POP_SONG)
        assert rec.recommend(make_user(), k=0) == []
