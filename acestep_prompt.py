from __future__ import annotations


def structured_lyrics(lyrics: str) -> str:
    """Give ACE-Step an explicit vocal section without altering lyric words."""
    if any(line.strip().startswith("[") and line.strip().endswith("]") for line in lyrics.splitlines()):
        return lyrics
    return f"[verse]\n{lyrics}"
