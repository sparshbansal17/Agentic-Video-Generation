import unittest

from storymem_agentic.audio_director import build_audio_plan


class AudioPlanTests(unittest.TestCase):
    def test_build_audio_plan_allocates_lines(self):
        plan = build_audio_plan("A line\nSecond line", target_duration_seconds=10)
        data = plan.to_dict()
        self.assertEqual(data["mode"], "voice_bed")
        self.assertEqual(len(data["lines"]), 2)
        self.assertLess(data["lines"][0]["start_seconds"], data["lines"][0]["end_seconds"])
        self.assertEqual(data["lines"][1]["scene_num"], 2)


if __name__ == "__main__":
    unittest.main()
