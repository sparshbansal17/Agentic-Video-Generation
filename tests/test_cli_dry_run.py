import json
import tempfile
import unittest
from pathlib import Path

from storymem_agentic.cli import main


class CliDryRunTests(unittest.TestCase):
    def test_cli_plan_audio_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rhyme = tmp_path / "rhyme.txt"
            rhyme.write_text("Twinkle star\nWonder far\n", encoding="utf-8")
            out = tmp_path / "out"
            code = main(["plan-audio", "--rhyme-file", str(rhyme), "--output-dir", str(out), "--target-duration", "8"])
            self.assertEqual(code, 0)
            plan = json.loads((out / "audio_plan.json").read_text(encoding="utf-8"))
            self.assertEqual(plan["target_duration_seconds"], 8)
            self.assertTrue((out / "mix_manifest.json").exists())
            self.assertTrue((out / "audio_evaluation_report.json").exists())

    def test_cli_run_writes_agentic_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rhyme = tmp_path / "rhyme.txt"
            rhyme.write_text("Twinkle star\nWonder far\n", encoding="utf-8")
            out = tmp_path / "run"
            code = main(["run", "--rhyme-file", str(rhyme), "--output-dir", str(out), "--target-duration", "8"])
            self.assertEqual(code, 0)
            manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["dry_run"])
            self.assertEqual(manifest["stages"][0]["name"], "audio_plan")
            self.assertEqual(manifest["stages"][1]["status"], "pending")
            self.assertTrue((out / "audio" / "audio_plan.json").exists())

    def test_iterate_requires_command_review_unless_explicitly_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "blocked"
            code = main(["iterate", "--topic", "moon lullaby", "--output-dir", str(out)])
            self.assertEqual(code, 2)
            self.assertFalse(out.exists())

    def test_iterate_allows_explicit_mock_review_for_debugging(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "allowed"
            code = main(
                [
                    "iterate",
                    "--topic",
                    "moon lullaby",
                    "--output-dir",
                    str(out),
                    "--allow-mock-review",
                    "--audio-aligner",
                    "none",
                    "--no-generate-audio",
                ]
            )
            self.assertEqual(code, 3)
            self.assertTrue((out / "iterations" / "001" / "evaluation_report.json").exists())


if __name__ == "__main__":
    unittest.main()
