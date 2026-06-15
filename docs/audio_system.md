# Timing-First Audio System

The audio pipeline is designed around one constraint: exact nursery-rhyme words and stable timing should not depend on a single music model following every instruction perfectly.

## Default: Voice + Instrumental Bed

1. The audio director creates `audio_plan.json` with lyric lines, scene mapping, target timestamps, voice backend, music backend, and mix policy.
2. F5-TTS or CosyVoice generates exact line stems from the planned lyrics.
3. MusicGen, Stable Audio, or ACE-Step generates one continuous instrumental bed from the full story/music prompt.
4. WhisperX aligns the generated voice and produces word timings.
5. The mixer adjusts pauses/time-stretch within tolerance, ducks music under voice, normalizes loudness, and emits a final mux-ready track.

## Sung-Song Mode

Sung songs are generated as one complete track with ACE-Step 1.5 or YuE. Independent sung scene segments are disallowed by policy unless a backend later proves reliable continuation from a shared reference.

## Feedback Actions

The evaluator should pick the smallest useful action:

- regenerate voice only when WER or line timing fails
- adjust pauses or time-stretch when timing is close
- regenerate music only when style/scene fit fails
- regenerate full song only in sung-song mode when lyric drift is structural
