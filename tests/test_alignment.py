import unittest

from storymem_agentic.alignment import analyze_whisperx_alignment, line_timestamps, transcript_from_words, word_error_rate


class AlignmentTests(unittest.TestCase):
    def test_continuous_song_does_not_fail_scene_window_drift(self):
        result = analyze_whisperx_alignment(
            ["Hush now", "Sleep now"],
            [(0.0, 2.0), (2.0, 4.0)],
            [
                {"word": "Hush", "start": 1.0, "end": 1.2},
                {"word": "now", "start": 1.3, "end": 1.5},
                {"word": "Sleep", "start": 5.0, "end": 5.2},
                {"word": "now", "start": 5.3, "end": 5.5},
            ],
            enforce_scene_windows=False,
        )
        self.assertTrue(result["passed"])
        self.assertFalse(result["enforce_scene_windows"])

    def test_continuous_song_rejects_excessive_initial_lyric_delay(self):
        result = analyze_whisperx_alignment(
            ["Hush now", "Sleep now"],
            [(0.0, 2.0), (2.0, 4.0)],
            [
                {"word": "Hush", "start": 10.8, "end": 11.2},
                {"word": "now", "start": 11.3, "end": 11.6},
                {"word": "Sleep", "start": 12.0, "end": 12.4},
                {"word": "now", "start": 12.5, "end": 12.8},
            ],
            enforce_scene_windows=False,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["initial_lyric_start_seconds"], 10.8)
        self.assertIn("lyrics_start_too_late", result["failure_reasons"])

    def test_word_error_rate_exact_match(self):
        self.assertEqual(word_error_rate("Twinkle little star", "twinkle little star"), 0)

    def test_transcript_from_words(self):
        self.assertEqual(transcript_from_words([{"word": "Twinkle"}, {"word": "star"}]), "Twinkle star")

    def test_line_timestamps_keeps_compound_token_timing(self):
        lines = line_timestamps(
            ["hush a bye"],
            [{"word": "hush-a-bye", "start": 1.0, "end": 1.8}],
        )

        self.assertEqual(lines[0]["matched_word_count"], 3)
        self.assertEqual(lines[0]["observed_start_seconds"], 1.0)
        self.assertEqual(lines[0]["observed_end_seconds"], 1.8)

    def test_line_timestamps_scopes_repeated_lines_to_planned_windows(self):
        aligned = [
            {"word": "Twinkle", "start": 0.2, "end": 0.6},
            {"word": "twinkle", "start": 0.7, "end": 1.1},
            {"word": "little", "start": 1.2, "end": 1.5},
            {"word": "star", "start": 1.6, "end": 2.0},
            {"word": "Twinkle", "start": 10.2, "end": 10.6},
            {"word": "twinkle", "start": 10.7, "end": 11.1},
            {"word": "little", "start": 11.2, "end": 11.5},
            {"word": "star", "start": 11.6, "end": 12.0},
        ]
        lines = line_timestamps(
            ["Twinkle twinkle little star", "Twinkle twinkle little star"],
            aligned,
            [(0.0, 4.0), (10.0, 14.0)],
        )

        self.assertEqual(lines[0]["observed_start_seconds"], 0.2)
        self.assertEqual(lines[1]["observed_start_seconds"], 10.2)

    def test_repeated_word_omission_fails_exact_alignment(self):
        result = analyze_whisperx_alignment(
            ["Twinkle, twinkle, little star"],
            [(0.0, 4.0)],
            [
                {"word": "Twinkle", "start": 0.2, "end": 0.6},
                {"word": "little", "start": 0.7, "end": 1.0},
                {"word": "star", "start": 1.1, "end": 1.5},
            ],
        )

        self.assertFalse(result["passed"])
        self.assertIn("line_1_omitted_repeated_twinkle_1_of_2", result["failure_reasons"])
        self.assertIn("line_1_final_line_incomplete", result["failure_reasons"])

    def test_one_observation_cannot_satisfy_two_overlapping_lines(self):
        lines = line_timestamps(
            ["star", "star"],
            [{"word": "star", "start": 1.0, "end": 1.4}],
            [(0.0, 2.0), (1.0, 3.0)],
        )

        self.assertEqual(lines[0]["matched_word_count"], 1)
        self.assertEqual(lines[1]["matched_word_count"], 0)

    def test_empty_alignment_is_pending_failure(self):
        result = analyze_whisperx_alignment(["Hush now"], [(0.0, 2.0)], [])

        self.assertFalse(result["passed"])
        self.assertIn("missing_observed_lyrics", result["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
