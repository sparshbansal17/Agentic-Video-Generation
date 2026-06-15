# Dependency Groups

The canonical dependency metadata lives in `pyproject.toml`. These files are convenience entrypoints for plain `python -m venv` + `pip` workflows on clusters where `uv` is not already installed.

- `base.txt`: install the agentic package only
- `dev.txt`: local development, tests, schema tooling
- `video.txt`: StoryMem/Wan GPU stack
- `audio-tts.txt`: F5-TTS and TTS helpers
- `audio-music.txt`: lightweight music helpers; heavyweight engines are external command adapters
- `align.txt`: WhisperX alignment
