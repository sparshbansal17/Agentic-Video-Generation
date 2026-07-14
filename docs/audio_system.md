# Video-First Song Generation

The video timeline is authoritative. Production audio must contain continuous music and genuine singing, preserve every supplied lyric, and place each line inside its fixed scene window.

## Contracts

Every run writes `audio/song_spec.json` before inference. The v2 contract contains immutable lyrics, exact line windows, total duration, BPM, key, meter, vocal direction, and a short music/story caption. Lyrics and timing never appear in the free-form caption.

Generated attempts are recorded as `AudioCandidate` entries in `audio_candidate_manifest.json`, including backend/model version, seed, parameters, alignment, technical metrics, context scores, and repaint history. A manifest is accepted only when it has a passing candidate; failed candidates are never silently published.

## Generation Policy

1. Reject lyric plans whose syllable density cannot fit the fixed video windows.
2. Generate eight continuous full-song candidates with ACE-Step 1.5 by default.
3. Gate candidates in this order: immutable lyrics, fixed timing, technical quality, then story/music fit.
4. Repaint failed lyric regions with surrounding musical context for at most two rounds.
5. Benchmark SongGeneration 2 through the same backend-neutral contract.
6. If whole-song models cannot satisfy correctness, use note-conditioned singing plus vocal-to-accompaniment generation. Spoken TTS and independently generated song fragments are not production fallbacks.

The legacy scene mix is available only with `--allow-scene-mix-debug`; it is diagnostic and must not be published.

## Review Policy

- `AudioTechnicalGate` deterministically checks the audio stream, duration, audibility, and peak safety.
- `WhisperXLyricTimingAgent` performs globally monotonic lyric matching; one observed word cannot satisfy multiple lines.
- `AudioReviewAgent` must use an audio-capable reviewer supplied with `--audio-review-command`.
- `AudioVisualSyncReviewAgent` consumes the actual audio/video media plus deterministic alignment evidence.
- Visual review remains independent. A reviewer process or JSON failure is recorded as `review_infrastructure_error`, not as a media-quality observation.

For local qualitative review, use `scripts/local_omni_audio_reviewer.py` with Qwen2.5-Omni. Keep it in a separate environment from Qwen2-VL because its Transformers and media dependencies differ.

## Acceptance Defaults

- At least 98% lyric coverage, with no missing content or repeated words.
- Line boundaries within 350 ms of the fixed windows.
- Final duration within 100 ms of the video.
- Audible lead vocal and accompaniment, safe peak level, and a clean ending.
- All deterministic gates pass before qualitative model review.
