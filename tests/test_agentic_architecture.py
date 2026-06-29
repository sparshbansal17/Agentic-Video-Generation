import json
import sys
import tempfile
import unittest
from pathlib import Path

from storymem_agentic.cli import main
from storymem_agentic.alignment import analyze_whisperx_alignment
from storymem_agentic.agents import MockAgentBackend
from storymem_agentic.feedback import apply_revision_plan, build_revision_plan
from storymem_agentic.media_evaluator import evaluate_iteration
from storymem_agentic.planner import PromptPlannerAgent, build_production_plan
from storymem_agentic.orchestrator import run_workflow
from storymem_agentic.schemas import EvaluationReport, NurseryRhymeInput, ReviewerReport, RevisionPlan
from storymem_agentic.story_writer import story_from_plan


class AgenticArchitectureTests(unittest.TestCase):
    def _plan(self):
        return build_production_plan(
            NurseryRhymeInput(
                rhyme_text="Twinkle, twinkle, little star,\nHow I wonder what you are!\nUp above the world so high,\n",
                target_duration_seconds=12,
                clip_count=3,
                seed=7,
            )
        )

    def test_planner_quantizes_segments_and_keeps_character_bank_out_of_prompts(self):
        plan = self._plan()
        self.assertEqual(plan.clip_count, 3)
        self.assertEqual(len(plan.character_bank), 3)
        for segment in plan.lyric_segments:
            self.assertAlmostEqual(segment.start_seconds * 24, round(segment.start_seconds * 24), places=5)
            self.assertAlmostEqual(segment.end_seconds * 24, round(segment.end_seconds * 24), places=5)
        self.assertIn("Wan scene-clip prompt using the Advanced Formula", plan.scenes[0].video_prompt)
        self.assertIn("Subject:", plan.scenes[0].video_prompt)
        self.assertIn("Scene:", plan.scenes[0].video_prompt)
        self.assertIn("Motion:", plan.scenes[0].video_prompt)
        self.assertIn("Aesthetic Control:", plan.scenes[0].video_prompt)
        self.assertIn("Stylization:", plan.scenes[0].video_prompt)
        self.assertIn("no picture-in-picture", plan.scenes[0].video_prompt)
        self.assertNotIn("pajama_child", plan.scenes[0].video_prompt)
        self.assertNotIn("smiling_star", plan.scenes[0].video_prompt)

    def test_visual_prompts_do_not_embed_literal_lyrics(self):
        plan = self._plan()

        self.assertEqual(plan.scenes[0].subtitle_text, "Twinkle, twinkle, little star,")
        self.assertNotIn("Twinkle, twinkle, little star", plan.scenes[0].video_prompt)
        self.assertIn("lyric is for meaning only", plan.scenes[0].video_prompt)
        self.assertIn("no written words", plan.scenes[0].first_frame_prompt)

    def test_story_writer_outputs_pipeline_compatible_json(self):
        story = story_from_plan(self._plan())
        self.assertEqual(len(story["scenes"]), 3)
        first = story["scenes"][0]
        self.assertEqual(first["scene_num"], 1)
        self.assertIn("video_prompts", first)
        self.assertIn("first_frame_prompt", first)
        self.assertIn("cut", first)
        self.assertEqual(first["subtitle_text"], "Twinkle, twinkle, little star,")

    def test_evaluator_flags_missing_media_and_feedback_targets_scenes(self):
        plan = self._plan()
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate_iteration(plan, Path(tmp), dry_run=False)
            self.assertFalse(report.passed)
            self.assertFalse(report.artifact_checks["clip_count"])
            revision = build_revision_plan(plan, report)
            self.assertEqual(revision.status, "needs_iteration")
            self.assertEqual(revision.target_scenes, [1, 2, 3])
            self.assertTrue(revision.regenerate_audio)

    def test_feedback_preserves_video_for_audio_only_failure(self):
        plan = self._plan()
        report = EvaluationReport(
            version="1.0",
            passed=False,
            artifact_checks={
                "storymem_compatible_plan": True,
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
                    reviewer="WhisperXLyricTimingAgent",
                    passed=False,
                    failure_reasons=["wer_above_threshold"],
                    evidence={"lines": []},
                )
            ],
            failure_reasons=["wer_above_threshold"],
        )

        revision = build_revision_plan(plan, report)

        self.assertEqual(revision.target_scenes, [])
        self.assertEqual(revision.preserve_scenes, [1, 2, 3])
        self.assertTrue(revision.regenerate_audio)

    def test_feedback_ignores_non_mapping_model_suggestions(self):
        plan = self._plan()
        report = EvaluationReport(
            version="1.0",
            passed=False,
            artifact_checks={
                "storymem_compatible_plan": True,
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
                    reviewer="ContinuityReviewAgent",
                    passed=False,
                    failure_reasons=["character_drift"],
                    evidence={
                        "target_scenes": [2],
                        "prompt_revisions": "met",
                        "first_frame_prompt_revisions": "met",
                        "subtitle_timing_adjustments": "met",
                        "mix_adjustments": "met",
                    },
                )
            ],
            regeneration_targets=[],
            failure_reasons=["character_drift"],
        )

        revision = build_revision_plan(plan, report)

        self.assertEqual(revision.target_scenes, [2])
        self.assertIn("2", revision.prompt_revisions)

    def test_cli_dry_run_writes_iteration_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rhyme = tmp_path / "rhyme.txt"
            rhyme.write_text("Twinkle star\nWonder far\nUp high\n", encoding="utf-8")
            out = tmp_path / "agentic"
            code = main(
                [
                    "dry-run",
                    "--rhyme-file",
                    str(rhyme),
                    "--output-dir",
                    str(out),
                    "--target-duration",
                    "12",
                    "--clip-count",
                    "3",
                ]
            )
            self.assertEqual(code, 0)
            iteration = out / "iterations" / "001"
            self.assertTrue((iteration / "production_plan.json").exists())
            self.assertTrue((iteration / "story.json").exists())
            self.assertTrue((iteration / "audio_plan.json").exists())
            self.assertTrue((iteration / "evaluation_report.json").exists())
            self.assertTrue((iteration / "revision_plan.json").exists())
            manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["passed"])

    def test_topic_only_input_creates_fallback_lullaby_plan(self):
        plan = build_production_plan(
            NurseryRhymeInput(topic_or_name="moon bedtime lullaby", target_duration_seconds=16, clip_count=4)
        )
        self.assertEqual(plan.clip_count, 4)
        self.assertIn("moon bedtime lullaby", plan.lyric_segments[0].text)
        self.assertTrue(all(scene.subtitle_text for scene in plan.scenes))

    def test_prompt_planner_agent_uses_backend_decision_for_topic_only_plan(self):
        backend = MockAgentBackend(
            responses={
                "planner": {
                    "lyrics": ["Line one from model", "Line two from model", "Line three from model"],
                    "clip_count": 3,
                    "target_duration_seconds": 18,
                    "characters": [
                        {
                            "label": "cloud_child",
                            "description": "same sleepy cloud child with a silver blanket",
                            "continuity_constraints": ["silver blanket stays visible"],
                            "negative_constraints": ["no in-frame text"],
                        }
                    ],
                    "scenes": [
                        {"description": "cloud child watches a quiet moon", "camera": "slow push in"},
                        {"description": "moon hums over soft rooftops", "camera": "gentle pan"},
                        {"description": "cloud child falls asleep", "camera": "soft fade"},
                    ],
                    "music_prompt": "quiet celesta lullaby generated by backend",
                }
            }
        )
        plan = PromptPlannerAgent(backend).plan(NurseryRhymeInput(topic_or_name="any lullaby prompt"))
        self.assertEqual(plan.clip_count, 3)
        self.assertGreater(plan.lyric_segments[-1].end_seconds, 14.0)
        self.assertLessEqual(plan.lyric_segments[-1].end_seconds, 15.0)
        self.assertEqual([segment.text for segment in plan.lyric_segments], backend.responses["planner"]["lyrics"])
        self.assertEqual(plan.character_bank[0].label, "cloud_child")
        self.assertIn("Opening shot:", plan.scenes[0].description)
        self.assertIn("Camera", plan.scenes[0].description)
        self.assertIn("Advanced Formula", plan.scenes[0].video_prompt)
        self.assertIn("hard cut transition", plan.scenes[0].video_prompt)
        self.assertTrue(all(scene.cut for scene in plan.scenes))
        self.assertEqual(plan.music_prompt, "quiet celesta lullaby generated by backend")

    def test_planner_preserves_rich_backend_scene_descriptions(self):
        rich_description = (
            "Opening shot: a cozy moonlit nursery with layered curtains in the foreground and a round window in "
            "the background. A sleepy cloud child gently lifts a silver blanket while a tiny moonbeam glows beside "
            "the bed. Camera slowly dollies from the bedside toward the window, revealing soft rooftops outside. "
            "Soft pastel 3D animation, warm practical lamp glow, calm blue-and-gold palette, soothing bedtime mood. "
            "Keep the child consistent, no written words, no scary shadows, child-safe magical atmosphere."
        )
        backend = MockAgentBackend(
            responses={
                "planner": {
                    "lyrics": ["Line one", "Line two"],
                    "clip_count": 2,
                    "target_duration_seconds": 10,
                    "characters": [{"label": "cloud_child", "description": "sleepy cloud child"}],
                    "scenes": [
                        {"description": rich_description, "camera": "slow dolly toward window"},
                        {"description": rich_description.replace("nursery", "cloud garden"), "camera": "slow pullback"},
                    ],
                    "music_prompt": "soft lullaby",
                }
            }
        )

        plan = PromptPlannerAgent(backend).plan(NurseryRhymeInput(topic_or_name="moon lullaby"))

        self.assertEqual(plan.scenes[0].description, rich_description)
        self.assertIn(rich_description, plan.scenes[0].video_prompt)

    def test_planner_normalizes_scenes_as_distinct_storyboard_cuts(self):
        backend = MockAgentBackend(
            responses={
                "planner": {
                    "lyrics": ["First line", "Second line", "Third line"],
                    "clip_count": 3,
                    "target_duration_seconds": 15,
                    "characters": [
                        {
                            "label": "sleepy_child",
                            "description": "same sleepy child in soft pajamas",
                        }
                    ],
                    "scenes": [
                        {"description": "child looks at moon", "camera": "medium shot"},
                        {"description": "moon shines over rooftops", "camera": "medium shot"},
                        {"description": "child sleeps", "camera": "medium shot"},
                    ],
                    "music_prompt": "soft lullaby",
                }
            }
        )

        plan = PromptPlannerAgent(backend).plan(NurseryRhymeInput(topic_or_name="moon lullaby"))

        self.assertTrue(all(scene.cut for scene in plan.scenes))
        self.assertEqual([scene.regeneration_dependencies for scene in plan.scenes], [[], [], []])
        self.assertIn("Opening shot:", plan.scenes[0].description)
        self.assertGreaterEqual(len(plan.scenes[0].description.split()), 45)
        self.assertIn("wide establishing shot", plan.scenes[0].video_prompt)
        self.assertIn("intimate close-up", plan.scenes[1].video_prompt)
        self.assertIn("low-angle wonder shot", plan.scenes[2].video_prompt)
        self.assertTrue(all("hard cut transition" in scene.video_prompt for scene in plan.scenes))
        self.assertTrue(all("No dialogue. No background music." in scene.video_prompt for scene in plan.scenes))

    def test_character_db_profiles_are_preserved_without_prompt_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "characters.json"
            db.write_text(
                json.dumps(
                    {
                        "characters": [
                            {
                                "id": "moon_bear",
                                "visual_description": "same plush moon bear with silver cap",
                                "continuity_constraints": ["silver cap stays visible"],
                                "negative_constraints": ["no text on cap"],
                                "reference_image_paths": ["refs/moon_bear.png"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            plan = build_production_plan(
                NurseryRhymeInput(
                    lyrics="Hush now\nDream now\n",
                    target_duration_seconds=8,
                    clip_count=2,
                    character_db_path=str(db),
                )
            )
            self.assertEqual(plan.character_bank[0].label, "moon_bear")
            self.assertNotIn("moon_bear", plan.scenes[0].video_prompt)
            self.assertEqual(plan.character_bank[0].reference_image_paths, ["refs/moon_bear.png"])

    def test_whisperx_alignment_flags_timing_drift(self):
        result = analyze_whisperx_alignment(
            ["Hush now"],
            [(2.0, 3.0)],
            [{"word": "Hush", "start": 0.8, "end": 1.0}, {"word": "now", "start": 1.05, "end": 1.2}],
        )
        self.assertFalse(result["passed"])
        self.assertIn("line_1_starts_before_scene", result["failure_reasons"])

    def test_full_song_timing_failure_triggers_scene_mix_fallback(self):
        from storymem_agentic.orchestrator import _needs_scene_lyrics_audio_fallback

        self.assertTrue(
            _needs_scene_lyrics_audio_fallback(
                {
                    "passed": False,
                    "failure_reasons": [
                        "line_1_missing_words",
                        "line_2_ends_after_scene",
                    ],
                }
            )
        )
        self.assertTrue(
            _needs_scene_lyrics_audio_fallback(
                {
                    "passed": False,
                    "failure_reasons": ["wer_above_threshold"],
                }
            )
        )
        self.assertFalse(_needs_scene_lyrics_audio_fallback({"passed": True, "failure_reasons": []}))

    def test_cli_plan_accepts_topic_and_writes_reviewer_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "topic_plan"
            code = main(
                [
                    "plan",
                    "--topic",
                    "moon bedtime lullaby",
                    "--output-dir",
                    str(out),
                    "--target-duration",
                    "12",
                    "--clip-count",
                    "3",
                ]
            )
            self.assertEqual(code, 0)
            iteration = out / "iterations" / "001"
            self.assertTrue((iteration / "storymem_commands.json").exists())
            self.assertTrue((iteration / "review_reports" / "ArtifactReviewAgent.json").exists())
            plan = json.loads((iteration / "production_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["rhyme"]["topic_or_name"], "moon bedtime lullaby")

    def test_command_review_backend_requires_vlm_command(self):
        report = evaluate_iteration(
            self._plan(),
            Path("/tmp/nonexistent-agentic-review"),
            dry_run=True,
            review_backend="command",
        )

        failed_reviewers = {item.reviewer: item.failure_reasons for item in report.reviewer_reports if not item.passed}
        self.assertIn("VisualSafetyReviewAgent", failed_reviewers)
        self.assertIn("missing_vlm_command", failed_reviewers["VisualSafetyReviewAgent"])

    def test_command_review_backend_uses_model_suggestions_in_revision(self):
        plan = self._plan()
        report = EvaluationReport(
            version="1.0",
            passed=False,
            artifact_checks={
                "storymem_compatible_plan": True,
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
                    reviewer="StoryAlignmentReviewAgent",
                    passed=False,
                    failure_reasons=["scene_2_lyric_mismatch"],
                    evidence={
                        "target_scenes": [2],
                        "prompt_revisions": {"2": "Show the child wondering at the star more clearly."},
                    },
                ),
                ReviewerReport(
                    reviewer="AudioReviewAgent",
                    passed=False,
                    failure_reasons=["music_too_harsh"],
                    evidence={"audio_prompt_revision": "Make the arrangement softer and slower."},
                ),
            ],
            failure_reasons=["scene_2_lyric_mismatch", "music_too_harsh"],
        )

        revision = build_revision_plan(plan, report)

        self.assertEqual(revision.target_scenes, [2])
        self.assertEqual(revision.preserve_scenes, [1, 3])
        self.assertIn("wondering at the star", revision.prompt_revisions["2"])
        self.assertIn("softer and slower", revision.audio_prompt_revision)

    def test_cli_optional_parameters_modify_planner_input_and_are_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "overrides"
            code = main(
                [
                    "plan",
                    "--topic",
                    "rainy nap lullaby",
                    "--output-dir",
                    str(out),
                    "--target-audience",
                    "preschoolers",
                    "--visual-style",
                    "soft watercolor bedtime animation",
                    "--audio-style",
                    "hummed lullaby with quiet piano",
                    "--target-duration",
                    "18",
                    "--clip-count",
                    "3",
                ]
            )
            self.assertEqual(code, 0)
            user_input = json.loads((out / "nursery_rhyme_input.json").read_text(encoding="utf-8"))
            self.assertEqual(user_input["target_audience"], "preschoolers")
            self.assertEqual(user_input["visual_style"], "soft watercolor bedtime animation")
            self.assertEqual(user_input["audio_style"], "hummed lullaby with quiet piano")
            planner_output = json.loads((out / "planner_agent_output.json").read_text(encoding="utf-8"))
            self.assertTrue(planner_output["used_fallback"])
            plan = json.loads((out / "iterations" / "001" / "production_plan.json").read_text(encoding="utf-8"))
            self.assertIn("soft watercolor bedtime animation", plan["scenes"][0]["video_prompt"])
            self.assertIn("hummed lullaby with quiet piano", plan["music_prompt"])

    def test_cli_direct_lyrics_are_preserved_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "direct_lyrics"
            lyrics = "Custom hush line\nCustom dream line\n"
            code = main(
                [
                    "plan",
                    "--topic",
                    "custom lullaby",
                    "--lyrics",
                    lyrics,
                    "--output-dir",
                    str(out),
                ]
            )
            self.assertEqual(code, 0)
            plan = json.loads((out / "iterations" / "001" / "production_plan.json").read_text(encoding="utf-8"))
            self.assertEqual([item["text"] for item in plan["lyric_segments"]], ["Custom hush line", "Custom dream line"])

    def test_command_planner_output_is_normalized_and_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            planner_script = tmp_path / "planner_backend.py"
            planner_script.write_text(
                "import json, sys\n"
                "json.load(sys.stdin)\n"
                "print(json.dumps({"
                "'lyrics':['Backend line one','Backend line two'],"
                "'clip_count':2,"
                "'target_duration_seconds':10,"
                "'characters':[{'label':'night_light','description':'same warm night light','continuity_constraints':['warm glow'],'negative_constraints':['no text']}],"
                "'scenes':[{'description':'night light glows','camera':'slow push'},{'description':'room settles','camera':'gentle hold'}],"
                "'music_prompt':'backend music prompt'"
                "}))\n",
                encoding="utf-8",
            )
            out = tmp_path / "command_plan"
            code = main(
                [
                    "plan",
                    "--topic",
                    "any lullaby",
                    "--output-dir",
                    str(out),
                    "--planner-backend",
                    "command",
                    "--planner-command",
                    f"{sys.executable} {planner_script}",
                ]
            )
            self.assertEqual(code, 0)
            planner_output = json.loads((out / "planner_agent_output.json").read_text(encoding="utf-8"))
            self.assertFalse(planner_output["used_fallback"])
            self.assertEqual(planner_output["response"]["lyrics"][0], "Backend line one")
            plan = json.loads((out / "iterations" / "001" / "production_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["character_bank"][0]["label"], "night_light")
            self.assertIn("Opening shot:", plan["scenes"][0]["description"])
            self.assertIn("Camera", plan["scenes"][0]["description"])
            self.assertGreaterEqual(len(plan["scenes"][0]["description"].split()), 45)
            self.assertNotIn("night light glows", plan["scenes"][0]["description"])

    def test_command_planner_invalid_json_falls_back_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            planner_script = tmp_path / "bad_planner_backend.py"
            planner_script.write_text(
                "import json, sys\n"
                "json.load(sys.stdin)\n"
                "print(json.dumps({'type':'object','properties':{'lyrics':{'type':'array'}}}))\n",
                encoding="utf-8",
            )
            out = tmp_path / "bad_command_plan"
            run_workflow(
                topic_or_name="any lullaby",
                output_dir=out,
                mode="generate",
                planner_backend="command",
                planner_command=f"{sys.executable} {planner_script}",
                generate_audio=False,
                execute_video=False,
            )
            planner_output = json.loads((out / "planner_agent_output.json").read_text(encoding="utf-8"))
            self.assertTrue(planner_output["used_fallback"])
            self.assertEqual(planner_output["fallback_policy"], "continued_with_deterministic_local_plan")
            self.assertIn("planner decision did not provide lyrics", planner_output["error"])
            self.assertTrue((out / "iterations" / "001" / "production_plan.json").exists())

    def test_iterate_applies_revision_plan_to_next_iteration_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "revision_loop"
            result = run_workflow(
                topic_or_name="moon bedtime lullaby",
                output_dir=out,
                mode="iterate",
                target_duration=12,
                clip_count=3,
                max_iterations=2,
                audio_aligner="none",
                generate_audio=False,
            )
            self.assertFalse(result["passed"])
            first = json.loads((out / "iterations" / "001" / "production_plan.json").read_text(encoding="utf-8"))
            second = json.loads((out / "iterations" / "002" / "production_plan.json").read_text(encoding="utf-8"))
            self.assertNotEqual(first["scenes"][0]["video_prompt"], second["scenes"][0]["video_prompt"])
            self.assertIn("Revision guidance", second["scenes"][0]["video_prompt"])
            self.assertTrue((out / "iterations" / "001" / "revision_plan.json").exists())

    def test_partial_whisperx_timing_revision_does_not_break_monotonic_plan(self):
        plan = build_production_plan(
            NurseryRhymeInput(
                topic_or_name="twinkle",
                lyrics="\n".join([f"Line {index}" for index in range(1, 7)]),
                clip_count=6,
            )
        )
        revision = RevisionPlan(
            version="1.0",
            status="needs_iteration",
            lyric_timing_adjustments={
                "1": {"observed_start_seconds": 0.2, "observed_end_seconds": 3.4},
                "2": {"observed_start_seconds": 4.5, "observed_end_seconds": 7.7},
                "3": {"observed_start_seconds": 8.8, "observed_end_seconds": 11.8},
                "4": {"observed_start_seconds": 13.0, "observed_end_seconds": 20.4},
                "5": {"observed_start_seconds": None, "observed_end_seconds": None},
                "6": {"observed_start_seconds": None, "observed_end_seconds": None},
            },
        )

        revised = apply_revision_plan(plan, revision)

        self.assertEqual(
            [(segment.start_seconds, segment.end_seconds) for segment in revised.lyric_segments],
            [(segment.start_seconds, segment.end_seconds) for segment in plan.lyric_segments],
        )

    def test_storymem_commands_split_first_shot_story_from_full_story(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "commands"
            code = main(
                [
                    "generate",
                    "--topic",
                    "moon bedtime lullaby",
                    "--output-dir",
                    str(out),
                    "--storymem-dir",
                    "/tmp/storymem",
                    "--t2v-model-path",
                    "/tmp/t2v",
                    "--i2v-model-path",
                    "/tmp/i2v",
                    "--lora-weight-path",
                    "/tmp/lora",
                    "--no-execute-video",
                    "--no-generate-audio",
                    "--audio-aligner",
                    "none",
                ]
            )
            self.assertEqual(code, 0)
            iteration = out / "iterations" / "001"
            commands = json.loads((iteration / "storymem_commands.json").read_text(encoding="utf-8"))["commands"]
            self.assertTrue(any("story_t2v_first_shot.json" in value for value in commands[0]))
            self.assertTrue(any(value.endswith("story.json") for value in commands[1]))
            self.assertIn("--t2v_first_shot", commands[0])
            self.assertNotIn("--mi2v", commands[0])
            self.assertIn("--mi2v", commands[1])
            first_story = json.loads((iteration / "story_t2v_first_shot.json").read_text(encoding="utf-8"))
            full_story = json.loads((iteration / "story.json").read_text(encoding="utf-8"))
            self.assertEqual(len(first_story["scenes"]), 1)
            self.assertGreater(len(full_story["scenes"]), 1)

    def test_first_shot_keyframe_fallback_uses_last_frame(self) -> None:
        from storymem_agentic.orchestrator import ensure_first_shot_memory_keyframe

        with tempfile.TemporaryDirectory() as tmp:
            generated = Path(tmp)
            last_frame = generated / "last_frame.jpg"
            last_frame.write_bytes(b"fake image bytes")

            fallback = ensure_first_shot_memory_keyframe(generated)

            self.assertEqual(fallback, generated / "01_01_keyframe0.jpg")
            self.assertEqual(fallback.read_bytes(), b"fake image bytes")

    def test_first_shot_keyframe_fallback_preserves_existing_keyframe(self) -> None:
        from storymem_agentic.orchestrator import ensure_first_shot_memory_keyframe

        with tempfile.TemporaryDirectory() as tmp:
            generated = Path(tmp)
            (generated / "01_01_keyframe0.jpg").write_bytes(b"existing")
            (generated / "last_frame.jpg").write_bytes(b"last")

            fallback = ensure_first_shot_memory_keyframe(generated)

            self.assertIsNone(fallback)
            self.assertEqual((generated / "01_01_keyframe0.jpg").read_bytes(), b"existing")

    def test_storymem_commands_resolve_paths_before_cwd_change(self) -> None:
        from storymem_agentic.orchestrator import build_storymem_commands

        commands = build_storymem_commands(
            story_json=Path("relative/out/iterations/001/story.json"),
            first_shot_story_json=Path("relative/out/iterations/001/story_t2v_first_shot.json"),
            output_dir=Path("relative/out/iterations/001/generated"),
            storymem_dir="/tmp/storymem",
            t2v_model_path=Path("models/t2v"),
            i2v_model_path=Path("models/i2v"),
            lora_weight_path=Path("models/lora"),
            nproc_per_node=2,
            t5_cpu=True,
        )

        first = commands[0]
        self.assertTrue(Path(first[first.index("--story_script_path") + 1]).is_absolute())
        self.assertTrue(Path(first[first.index("--output_dir") + 1]).is_absolute())
        self.assertTrue(Path(first[first.index("--t2v_model_path") + 1]).is_absolute())
        self.assertIn("--t5_cpu", first)

    def test_storymem_commands_expose_speed_options(self) -> None:
        from storymem_agentic.orchestrator import build_storymem_commands

        commands = build_storymem_commands(
            story_json=Path("relative/out/iterations/001/story.json"),
            first_shot_story_json=Path("relative/out/iterations/001/story_t2v_first_shot.json"),
            output_dir=Path("relative/out/iterations/001/generated"),
            storymem_dir="/tmp/storymem",
            t2v_model_path=Path("models/t2v"),
            i2v_model_path=Path("models/i2v"),
            lora_weight_path=Path("models/lora"),
            nproc_per_node=8,
            sample_steps=20,
            frame_num=41,
            keyframe_mode="simple",
        )

        first = commands[0]
        second = commands[1]
        self.assertNotIn("--offload_model", first)
        self.assertNotIn("--offload_model", second)
        self.assertEqual(first[first.index("--sample_steps") + 1], "20")
        self.assertEqual(second[second.index("--frame_num") + 1], "41")
        self.assertEqual(first[first.index("--keyframe_mode") + 1], "simple")

    def test_resume_reuses_completed_iteration_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "resume"
            result = run_workflow(
                topic_or_name="moon bedtime lullaby",
                output_dir=out,
                mode="iterate",
                target_duration=10,
                clip_count=2,
                max_iterations=1,
                generate_audio=False,
                execute_video=False,
            )
            self.assertFalse(result["passed"])
            report_path = out / "iterations" / "001" / "evaluation_report.json"
            original_report = json.loads(report_path.read_text(encoding="utf-8"))
            original_report["failure_reasons"] = ["cached_marker"]
            report_path.write_text(json.dumps(original_report, indent=2) + "\n", encoding="utf-8")

            resumed = run_workflow(
                topic_or_name="moon bedtime lullaby",
                output_dir=out,
                mode="iterate",
                target_duration=10,
                clip_count=2,
                max_iterations=1,
                generate_audio=False,
                execute_video=False,
            )

            self.assertFalse(resumed["passed"])
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8"))["failure_reasons"],
                ["cached_marker"],
            )

    def test_storymem_continuation_command_uses_mi2v_only(self) -> None:
        from storymem_agentic.orchestrator import build_storymem_continuation_command

        command = build_storymem_continuation_command(
            story_json=Path("relative/out/iterations/002/story_from_scene_02.json"),
            output_dir=Path("relative/out/iterations/002/generated"),
            storymem_dir="/tmp/storymem",
            t2v_model_path=Path("models/t2v"),
            i2v_model_path=Path("models/i2v"),
            lora_weight_path=Path("models/lora"),
            nproc_per_node=4,
        )

        self.assertIn("--mi2v", command)
        self.assertNotIn("--t2v_first_shot", command)
        self.assertTrue(Path(command[command.index("--story_script_path") + 1]).is_absolute())


if __name__ == "__main__":
    unittest.main()
