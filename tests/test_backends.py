import unittest

from storymem_agentic.backends import default_backends


class BackendTests(unittest.TestCase):
    def test_backend_command_rendering_quotes_arguments(self):
        command = default_backends()["musicgen"]
        rendered = command.render({"music_prompt": "soft lullaby", "duration": 12, "output_file": "out.wav"})
        self.assertEqual(rendered[0], "musicgen-generate")
        self.assertIn("soft lullaby", rendered)
        self.assertIn("out.wav", rendered)


if __name__ == "__main__":
    unittest.main()
