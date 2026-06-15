import unittest

from storymem_agentic.alignment import transcript_from_words, word_error_rate


class AlignmentTests(unittest.TestCase):
    def test_word_error_rate_exact_match(self):
        self.assertEqual(word_error_rate("Twinkle little star", "twinkle little star"), 0)

    def test_transcript_from_words(self):
        self.assertEqual(transcript_from_words([{"word": "Twinkle"}, {"word": "star"}]), "Twinkle star")


if __name__ == "__main__":
    unittest.main()
