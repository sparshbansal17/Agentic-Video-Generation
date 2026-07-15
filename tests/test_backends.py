import unittest
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from story_audio import AudioConfig, _mix_stems, _render_command, _validate_voice_reference, generate_audio_for_story
from storymem_agentic.backends import default_backends
from storymem_agentic.cli import build_parser
from storymem_agentic.media_evaluator import accompaniment_artifact_report
from storymem_agentic.orchestrator import _alignment_rank, _bounded_candidate_count, _retry_seed_offsets


class BackendTests(unittest.TestCase):
    def test_backend_command_rendering_quotes_arguments(self):
        command = default_backends()["musicgen"]
        rendered = command.render({"music_prompt": "soft lullaby", "duration": 12, "output_file": "out.wav"})
        self.assertEqual(rendered[0], "musicgen-generate")
        self.assertIn("soft lullaby", rendered)
        self.assertIn("out.wav", rendered)

    def test_default_backends_include_hybrid_music_backend(self):
        backends = default_backends()

        self.assertIn("stable_audio", backends)
        self.assertEqual(backends["stable_audio"].kind, "music")

    def test_exact_voice_backends_require_reference_audio_and_text(self):
        with self.assertRaisesRegex(RuntimeError, "voice-ref-audio"):
            _validate_voice_reference(AudioConfig("story.json", "out", vocal_backend="f5_tts"))

        with TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref.wav"
            ref.write_bytes(b"fake wav")
            with self.assertRaisesRegex(RuntimeError, "voice-ref-text"):
                _validate_voice_reference(
                    AudioConfig(
                        "story.json",
                        "out",
                        vocal_backend="cosyvoice",
                        voice_ref_audio=str(ref),
                    )
                )

            _validate_voice_reference(
                AudioConfig(
                    "story.json",
                    "out",
                    vocal_backend="f5_tts",
                    voice_ref_audio=str(ref),
                    voice_ref_text="gentle reference line",
                )
            )

    def test_voice_template_values_cover_f5_tts_and_cosyvoice(self):
        values = {
            "text": "Twinkle, twinkle",
            "gen_text": "Twinkle, twinkle",
            "ref_audio": "/tmp/ref voice.wav",
            "ref_text": "Reference words",
            "output_file": "/tmp/out.wav",
        }

        rendered = _render_command(
            "f5-tts_infer-cli --ref_audio ${ref_audio} --ref_text ${ref_text} "
            "--gen_text ${gen_text} --output_file ${output_file}",
            values,
        )

        self.assertIn("/tmp/ref voice.wav", rendered)
        self.assertIn("Reference words", rendered)
        self.assertIn("Twinkle, twinkle", rendered)

    def test_workflow_cli_accepts_hybrid_voice_bed_backends(self):
        args = build_parser().parse_args(
            [
                "iterate",
                "--topic",
                "twinkle twinkle little star",
                "--output-dir",
                "out",
                "--allow-mock-review",
                "--media-audio-mode",
                "hybrid_voice_bed",
                "--voice-backend",
                "f5_tts",
                "--music-backend",
                "stable_audio",
                "--voice-ref-audio",
                "/tmp/ref.wav",
                "--voice-ref-text",
                "soft reference voice",
                "--allow-scene-mix-debug",
                "--full-song-candidates",
                "3",
            ]
        )

        self.assertEqual(args.media_audio_mode, "hybrid_voice_bed")
        self.assertEqual(args.voice_backend, "f5_tts")
        self.assertEqual(args.music_backend, "stable_audio")
        self.assertEqual(args.voice_ref_text, "soft reference voice")
        self.assertTrue(args.allow_scene_mix_debug)
        self.assertEqual(args.full_song_candidates, 3)

    def test_scene_retry_candidate_limits_count_initial_render(self):
        self.assertEqual(_bounded_candidate_count(4, 1), 1)
        self.assertEqual(_bounded_candidate_count(4, 3), 3)
        self.assertEqual(_bounded_candidate_count(0, 3), 1)
        self.assertEqual(_retry_seed_offsets(1, [1601, 3203, 4801]), [])
        self.assertEqual(_retry_seed_offsets(3, [1601, 3203, 4801]), [1601, 3203])

    def test_candidate_rank_prefers_complete_final_lyrics_over_lower_wer(self):
        complete = {
            "passed": False,
            "word_error_rate": 0.2,
            "repeated_word_omission_count": 0,
            "failure_reasons": ["line_1_starts_before_scene"],
            "lines": [
                {"expected_word_count": 4, "matched_word_count": 4, "missing_word_count": 0},
            ],
        }
        incomplete_lower_wer = {
            "passed": False,
            "word_error_rate": 0.05,
            "repeated_word_omission_count": 1,
            "failure_reasons": ["line_1_omitted_repeated_twinkle_1_of_2"],
            "lines": [
                {"expected_word_count": 4, "matched_word_count": 3, "missing_word_count": 1},
            ],
        }

        self.assertLess(_alignment_rank(complete), _alignment_rank(incomplete_lower_wer))

    def test_stem_mix_graph_splits_vocal_sidechain_and_mix_labels(self):
        source = inspect.getsource(_mix_stems)

        self.assertIn("asplit=2[voc_sc][voc_mix]", source)
        self.assertIn("[back][voc_sc]sidechaincompress", source)
        self.assertIn("[ducked][voc_mix]amix", source)

    def test_full_song_generates_and_mixes_explicit_instrumental_backing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            story = root / "story.json"
            story.write_text(json.dumps({"story_name": "Test", "scenes": []}), encoding="utf-8")
            lyrics = root / "lyrics.txt"
            lyrics.write_text("Sleep softly\n", encoding="utf-8")
            video = root / "generated.mp4"
            video.write_bytes(b"video")
            config = AudioConfig(
                story_script_path=str(story),
                output_dir=str(root),
                final_video=str(video),
                lyrics_file=str(lyrics),
                vocal_backend="ace_step_full_song",
                ace_step_cmd="fake ${output_file}",
            )

            invocations = []

            def fake_run(_template, values, label):
                invocations.append((dict(values), label))
                Path(values["output_file"]).write_bytes(b"RIFF" + b"audio" * 20)

            with (
                patch("story_audio._video_duration", return_value=20.0),
                patch("story_audio._find_ffmpeg", return_value="ffmpeg"),
                patch("story_audio._run_template", side_effect=fake_run),
                patch("story_audio._mix_stems") as mix,
                patch("story_audio._mux_video"),
            ):
                generate_audio_for_story(config)

            self.assertEqual(len(invocations), 2)
            backing_values, backing_label = invocations[1]
            self.assertEqual(backing_label, "instrumental backing-track generation")
            self.assertEqual(Path(backing_values["lyrics_file"]).read_text().strip(), "[instrumental]")
            self.assertIn("instrumental accompaniment only", backing_values["music_prompt"])
            self.assertIn("no singing", backing_values["music_prompt"])
            mix.assert_called_once_with(
                "ffmpeg",
                root / "audio" / "song.wav",
                root / "audio" / "backing.wav",
                root / "audio" / "mixed_song.wav",
                20.0,
            )

    def test_accompaniment_gate_rejects_missing_backing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "audio"
            audio.mkdir()
            (audio / "audio_prompt.json").write_text(
                json.dumps({"audio_mode": "full_song"}),
                encoding="utf-8",
            )

            report = accompaniment_artifact_report(root)

            self.assertIsNotNone(report)
            self.assertFalse(report.passed)
            self.assertIn("missing_instrumental_backing", report.failure_reasons)

    def test_accompaniment_gate_accepts_audible_backing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "audio"
            audio.mkdir()
            (audio / "audio_prompt.json").write_text(
                json.dumps({"audio_mode": "full_song"}),
                encoding="utf-8",
            )
            backing = audio / "backing.wav"
            backing.write_bytes(b"RIFF" + b"audio" * 20)
            quality = {
                "passed": True,
                "failure_reasons": [],
                "metrics": {"has_audio": True, "mean_volume_db": -20.0},
            }

            with patch("storymem_agentic.media_evaluator.probe_audio_quality", return_value=quality):
                report = accompaniment_artifact_report(root)

            self.assertIsNotNone(report)
            self.assertTrue(report.passed)
            self.assertEqual(report.evidence["backing_path"], str(backing))


if __name__ == "__main__":
    unittest.main()
