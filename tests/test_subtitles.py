import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storymem_agentic.subtitles import (
    add_whisperx_subtitles,
    default_subtitled_video_path,
    whisperx_cues,
)


class WhisperXSubtitleTests(unittest.TestCase):
    def test_cues_use_observed_whisperx_words_not_planner_lyrics(self):
        alignment = {
            "language": "en",
            "word_segments": [
                {"word": "Actually", "start": 0.2, "end": 0.7},
                {"word": "spoken", "start": 0.8, "end": 1.2},
                {"word": "words", "start": 1.3, "end": 1.8},
                {"word": "Second", "start": 2.4, "end": 2.9},
                {"word": "phrase", "start": 3.0, "end": 3.5},
            ],
        }

        cues = whisperx_cues(alignment)

        self.assertEqual([cue.text for cue in cues], ["Actually spoken words", "Second phrase"])
        self.assertNotIn("planner", " ".join(cue.text for cue in cues).lower())
        self.assertAlmostEqual(cues[0].start_seconds, 0.2)
        self.assertLess(cues[0].end_seconds, cues[1].start_seconds)

    def test_single_word_timing_fragment_is_merged_into_readable_cue(self):
        alignment = {
            "segments": [
                {
                    "words": [
                        {"word": "Little", "start": 1.0, "end": 1.1},
                        {"word": "children", "start": 2.4, "end": 2.9},
                        {"word": "want", "start": 3.0, "end": 3.3},
                        {"word": "to", "start": 3.4, "end": 3.5},
                        {"word": "play", "start": 3.6, "end": 4.0},
                    ]
                }
            ]
        }

        cues = whisperx_cues(alignment)

        self.assertEqual([cue.text for cue in cues], ["Little children want to play"])

    def test_postprocess_writes_ass_and_provenance_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "generated_with_music.mp4"
            video.touch()
            alignment = root / "whisperx.json"
            alignment.write_text(
                json.dumps(
                    {
                        "language": "en",
                        "word_segments": [
                            {"word": "Observed", "start": 0.1, "end": 0.5},
                            {"word": "speech", "start": 0.6, "end": 1.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            subtitle_file = root / "subtitles.ass"
            result_file = root / "subtitle_postprocess_result.json"
            output_file = root / "generated_subtitled_with_music.mp4"

            with patch("storymem_agentic.subtitles.burn_subtitles", return_value=output_file):
                result = add_whisperx_subtitles(
                    video_file=video,
                    whisperx_json=alignment,
                    subtitle_file=subtitle_file,
                    output_file=output_file,
                    result_file=result_file,
                )

            self.assertIn("Observed speech", subtitle_file.read_text(encoding="utf-8"))
            self.assertFalse(result["planner_lyrics_used"])
            self.assertEqual(result["source"], "whisperx_observed_speech")
            self.assertTrue(result_file.exists())

    def test_default_name_inserts_subtitled_before_with_music(self):
        self.assertEqual(
            default_subtitled_video_path("generated_with_music.mp4").name,
            "generated_subtitled_with_music.mp4",
        )


if __name__ == "__main__":
    unittest.main()
