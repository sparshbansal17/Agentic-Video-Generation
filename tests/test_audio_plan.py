import unittest

from story_audio import AudioConfig, _audio_style_prompt
from storymem_agentic.audio_director import build_audio_plan
from storymem_agentic.evaluation import evaluate_manifest
from storymem_agentic.feedback import build_revision_plan
from storymem_agentic.schemas import EvaluationReport, ProductionPlan, ReviewerReport


class AudioPlanTests(unittest.TestCase):
    def test_build_audio_plan_allocates_lines(self):
        plan = build_audio_plan("A line\nSecond line", target_duration_seconds=10)
        data = plan.to_dict()
        self.assertEqual(data["mode"], "voice_bed")
        self.assertEqual(len(data["lines"]), 2)
        self.assertLess(data["lines"][0]["start_seconds"], data["lines"][0]["end_seconds"])
        self.assertEqual(data["lines"][1]["scene_num"], 2)

    def test_audio_evaluation_fails_when_lyrics_are_missing_from_alignment(self):
        plan = build_audio_plan("Twinkle, twinkle, little star", target_duration_seconds=5)
        report = evaluate_manifest(
            plan,
            {
                "target_duration_seconds": 5,
            },
            aligned_words=[],
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["alignment"]["passes_wer"])
        self.assertIn("missing_observed_lyrics", report["alignment"]["failure_reasons"])

    def test_audio_prompt_preserves_exact_lyrics_and_repeated_warning_under_truncation(self):
        story = {
            "story_name": "twinkle",
            "story_overview": "x" * 5000,
            "agentic_metadata": {"music_prompt": "plinky_unique. plinky_unique. gentle."},
            "scenes": [
                {
                    "lyric_line": "Twinkle, twinkle, little star",
                    "planned_start_seconds": 20.0,
                    "planned_end_seconds": 25.0,
                    "video_prompts": ["A very long scene " + ("sparkle " * 1000)],
                }
            ],
        }

        prompt = _audio_style_prompt(story, AudioConfig("story.json", "out"))

        self.assertIn("1. Twinkle, twinkle, little star", prompt)
        self.assertIn("20.000-25.000s", prompt)
        self.assertIn('"Twinkle" appears 2 times', prompt)
        self.assertEqual(prompt.count("plinky_unique"), 1)

    def test_repeated_word_omission_fails_manifest_evaluation(self):
        plan = build_audio_plan("Twinkle, twinkle, little star", target_duration_seconds=5)
        report = evaluate_manifest(
            plan,
            {"target_duration_seconds": 5},
            aligned_words=[
                {"word": "Twinkle"},
                {"word": "little"},
                {"word": "star"},
            ],
        )

        self.assertFalse(report["passed"])
        self.assertIn("line_1_omitted_repeated_twinkle_1_of_2", report["alignment"]["failure_reasons"])

    def test_revision_guidance_deduplicates_repeated_audio_suggestions(self):
        plan = ProductionPlan.from_dict(
            {
                "version": "1.0",
                "rhyme": {"topic_or_name": "twinkle"},
                "target_fps": 24,
                "clip_count": 1,
                "audio_mode": "full_song",
                "music_prompt": "soft",
                "evaluation_rubric": {},
                "character_bank": [],
                "lyric_segments": [
                    {"index": 1, "text": "Twinkle, twinkle, little star", "start_seconds": 0.0, "end_seconds": 5.0}
                ],
                "scenes": [
                    {
                        "scene_num": 1,
                        "lyric_segment_index": 1,
                        "start_seconds": 0.0,
                        "end_seconds": 5.0,
                        "description": "star scene",
                        "subtitle_text": "Twinkle, twinkle, little star",
                        "video_prompt": "No generated text.",
                        "first_frame_prompt": "No generated text.",
                        "audio_description": "soft",
                        "cut": True,
                    }
                ],
            }
        )
        report = EvaluationReport(
            version="1.0",
            passed=False,
            artifact_checks={
                "clip_count": True,
                "final_video_exists": True,
                "has_video_stream": True,
                "has_audio_stream": True,
                "has_subtitles": True,
                "duration_match": True,
            },
            scene_reports=[],
            reviewer_reports=[
                ReviewerReport(
                    reviewer="AudioReviewAgent",
                    passed=False,
                    failure_reasons=["lyric_missing"],
                    evidence={"audio_prompt_revision": "Make diction clearer. Make diction clearer."},
                ),
                ReviewerReport(
                    reviewer="WhisperXLyricTimingAgent",
                    passed=False,
                    failure_reasons=["line_1_omitted_repeated_twinkle_1_of_2"],
                    evidence={},
                ),
            ],
            whisperx_alignment={
                "lines": [
                    {
                        "line_index": 1,
                        "text": "Twinkle, twinkle, little star",
                        "matched_word_count": 3,
                        "expected_word_count": 4,
                        "repeated_word_omissions": {"twinkle": {"matched": 1, "expected": 2}},
                    }
                ]
            },
            failure_reasons=["line_1_omitted_repeated_twinkle_1_of_2"],
        )

        revision = build_revision_plan(plan, report)

        self.assertEqual(revision.audio_prompt_revision.count("Make diction clearer"), 1)
        self.assertIn('line 1 omitted repeated "twinkle"', revision.audio_prompt_revision)


if __name__ == "__main__":
    unittest.main()
