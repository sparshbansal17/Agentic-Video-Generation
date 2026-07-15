import unittest

from storymem_agentic.audio_director import build_audio_plan
from storymem_agentic.audio_quality import evaluate_audio_metrics
from storymem_agentic.schemas import AudioCandidate
from storymem_agentic.song_pipeline import (
    build_song_spec,
    candidate_manifest,
    repair_action_for_candidate,
    select_passing_candidate,
    validate_song_feasibility,
)
from storymem_agentic.mixer import build_mix_manifest


class SongPipelineTests(unittest.TestCase):
    def setUp(self):
        self.plan = build_audio_plan(
            "Twinkle, twinkle, little star\nHow I wonder what you are",
            target_duration_seconds=10,
            mode="full_song",
            voice_backend="ace_step_full_song",
            music_backend="ace_step_full_song",
        )
        self.spec = build_song_spec(self.plan, story_context="stars appear above a warm bedroom")

    def test_song_spec_separates_caption_from_immutable_lyrics(self):
        self.assertEqual(self.spec.timing_authority, "video")
        self.assertEqual(self.spec.lyrics, self.plan.lyrics)
        self.assertNotIn("Twinkle", self.spec.caption)
        self.assertTrue(validate_song_feasibility(self.spec)["passed"])

    def test_selection_never_returns_a_failing_candidate(self):
        failed = AudioCandidate("bad", "ace", "1.5", 1, "bad.wav", passed=False)
        self.assertIsNone(select_passing_candidate([failed]))
        self.assertEqual(candidate_manifest(self.spec, [failed])["status"], "audio_generation_failed")

    def test_candidate_ranking_prioritizes_lyrics(self):
        complete = AudioCandidate(
            "complete", "ace", "1.5", 1, "a.wav",
            alignment={"word_error_rate": 0.0, "lines": [], "repeated_word_omission_count": 0},
            technical_metrics={"passed": True}, context_scores={"fit": 0.5}, passed=True,
        )
        incomplete = AudioCandidate(
            "pretty", "ace", "1.5", 2, "b.wav",
            alignment={"word_error_rate": 0.2, "lines": [{"missing_word_count": 1}], "repeated_word_omission_count": 0},
            technical_metrics={"passed": True}, context_scores={"fit": 1.0}, passed=True,
        )
        self.assertEqual(select_passing_candidate([incomplete, complete]).candidate_id, "complete")

    def test_failed_lines_produce_contextual_repaint_then_hard_fail(self):
        candidate = AudioCandidate(
            "bad", "ace", "1.5", 1, "bad.wav",
            alignment={"lines": [{"line_index": 1, "missing_word_count": 1}]},
        )
        repaint = repair_action_for_candidate(self.spec, candidate, repaint_round=0)
        self.assertEqual(repaint.kind, "repaint_region")
        self.assertEqual(repaint.target_lines, [1])
        self.assertEqual(repair_action_for_candidate(self.spec, candidate, repaint_round=2).kind, "hard_fail")

    def test_technical_gate_rejects_silence_and_duration_error(self):
        result = evaluate_audio_metrics({
            "has_audio": True,
            "duration_seconds": 9.5,
            "expected_duration_seconds": 10.0,
            "mean_volume_db": -60.0,
            "max_volume_db": -1.0,
        })
        self.assertFalse(result["passed"])
        self.assertIn("audio_duration_out_of_tolerance", result["failure_reasons"])
        self.assertIn("audio_effectively_silent", result["failure_reasons"])

    def test_full_song_manifest_names_real_song_and_backing_artifacts(self):
        manifest = build_mix_manifest(self.plan)

        self.assertEqual(manifest["voice_stems"], ["song.wav"])
        self.assertEqual(manifest["music_bed"], "backing.wav")
        self.assertEqual(manifest["output_file"], "mixed_song.wav")
        self.assertTrue(manifest["ducking"]["enabled"])


if __name__ == "__main__":
    unittest.main()
