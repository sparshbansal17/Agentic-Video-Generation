import unittest

from storymem_agentic.alignment import line_timestamps, transcript_from_words, word_error_rate


class AlignmentTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
