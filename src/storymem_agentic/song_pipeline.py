from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from .schemas import AudioCandidate, AudioPlan, AudioRepairAction, SongLineSpec, SongSpec
from .song_backends import SongBackend

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def estimate_syllables(text: str) -> int:
    """Conservative English syllable estimate used only for timing preflight."""
    total = 0
    for raw_word in _WORD_RE.findall(text):
        word = raw_word.lower()
        groups = len(_VOWEL_GROUP_RE.findall(word))
        if word.endswith("e") and not word.endswith(("le", "ye")) and groups > 1:
            groups -= 1
        total += max(groups, 1)
    return max(total, 1)


def _clean_caption(*parts: str, max_chars: int = 600) -> str:
    seen: set[str] = set()
    clean: list[str] = []
    for part in parts:
        for phrase in re.split(r"[.;\n]+", str(part)):
            phrase = " ".join(phrase.split()).strip(" ,")
            key = phrase.lower()
            if not phrase or key in seen or "exact lyric" in key or "timing window" in key:
                continue
            seen.add(key)
            clean.append(phrase)
    return ", ".join(clean)[:max_chars].rstrip(" ,")


def build_song_spec(
    plan: AudioPlan,
    *,
    story_context: str = "",
    bpm: int = 72,
    key_scale: str = "C major",
    time_signature: str = "3/4",
    timing_authority: str = "video",
) -> SongSpec:
    caption = _clean_caption(
        "gentle child-safe nursery song with clear lead singing and continuous accompaniment",
        plan.music_prompt,
        story_context,
    )
    spec = SongSpec(
        version="2.0",
        timing_authority=timing_authority,  # type: ignore[arg-type]
        duration_seconds=plan.target_duration_seconds,
        lyrics=list(plan.lyrics),
        lines=[
            SongLineSpec(
                index=line.index,
                text=line.text,
                scene_num=line.scene_num,
                start_seconds=line.start_seconds,
                end_seconds=line.end_seconds,
                syllable_count=estimate_syllables(line.text),
            )
            for line in plan.lines
        ],
        caption=caption,
        bpm=bpm,
        key_scale=key_scale,
        time_signature=time_signature,
    )
    spec.validate()
    return spec


def validate_song_feasibility(
    spec: SongSpec,
    *,
    minimum_syllables_per_second: float = 0.8,
    maximum_syllables_per_second: float = 4.0,
) -> dict[str, Any]:
    issues = []
    for line in spec.lines:
        duration = line.end_seconds - line.start_seconds
        density = line.syllable_count / duration
        if density > maximum_syllables_per_second:
            issues.append({
                "code": "line_too_dense",
                "line_index": line.index,
                "syllables_per_second": round(density, 3),
            })
        elif density < minimum_syllables_per_second:
            # Sparse lines are musically feasible; record them as arrangement guidance.
            issues.append({
                "code": "line_too_sparse",
                "line_index": line.index,
                "syllables_per_second": round(density, 3),
                "severity": "warning",
            })
    return {
        "passed": not any(issue.get("severity", "error") == "error" for issue in issues),
        "issues": issues,
    }


def candidate_rank(candidate: AudioCandidate) -> tuple[float, float, float, float, str]:
    """Lower is better; correctness always precedes subjective quality."""
    alignment = candidate.alignment or {}
    missing = sum(int(line.get("missing_word_count", 0)) for line in alignment.get("lines", []))
    repeated = int(alignment.get("repeated_word_omission_count", 0))
    lyric_penalty = missing + repeated + float(alignment.get("word_error_rate", 1.0))
    timing_penalty = sum(
        max(abs(float(line.get("start_drift_seconds", 0.0))), abs(float(line.get("end_drift_seconds", 0.0))))
        for line in alignment.get("lines", [])
    )
    technical_penalty = 0.0 if bool(candidate.technical_metrics.get("passed", False)) else 1.0
    context_quality = max(candidate.context_scores.values(), default=0.0)
    return (lyric_penalty, timing_penalty, technical_penalty, -context_quality, candidate.candidate_id)


def select_passing_candidate(candidates: Iterable[AudioCandidate]) -> AudioCandidate | None:
    passing = [candidate for candidate in candidates if candidate.passed]
    return min(passing, key=candidate_rank) if passing else None


def failed_line_indices(alignment: dict[str, Any]) -> list[int]:
    failed = []
    for line in alignment.get("lines", []):
        if (
            int(line.get("missing_word_count", 0)) > 0
            or line.get("observed_start_seconds") is None
            or line.get("observed_end_seconds") is None
            or abs(float(line.get("start_drift_seconds", 0.0))) > 0.35
            or abs(float(line.get("end_drift_seconds", 0.0))) > 0.35
        ):
            failed.append(int(line["line_index"]))
    return failed


def repair_action_for_candidate(
    spec: SongSpec,
    candidate: AudioCandidate,
    *,
    repaint_round: int,
    maximum_repaint_rounds: int = 2,
    context_seconds: float = 1.0,
) -> AudioRepairAction:
    failed = failed_line_indices(candidate.alignment)
    if not failed:
        return AudioRepairAction(kind="adjust_mix", reason="lyrics and timing passed; finalize mix")
    if repaint_round >= maximum_repaint_rounds:
        return AudioRepairAction(
            kind="hard_fail",
            reason="candidate still violates immutable lyric or video timing constraints after repaint budget",
            target_lines=failed,
        )
    by_index = {line.index: line for line in spec.lines}
    target = [by_index[index] for index in failed if index in by_index]
    return AudioRepairAction(
        kind="repaint_region",
        reason="repair only failed lyric windows while preserving whole-song continuity",
        start_seconds=max(0.0, min(line.start_seconds for line in target) - context_seconds),
        end_seconds=min(spec.duration_seconds, max(line.end_seconds for line in target) + context_seconds),
        target_lines=failed,
        parameter_changes={"preserve_lyrics": True, "preserve_surrounding_audio": True},
    )


def candidate_manifest(spec: SongSpec, candidates: Iterable[AudioCandidate]) -> dict[str, Any]:
    items = sorted(candidates, key=candidate_rank)
    selected = select_passing_candidate(items)
    return {
        "version": "2.0",
        "song_spec": spec.to_dict(),
        "status": "accepted" if selected else "audio_generation_failed",
        "selected_candidate_id": selected.candidate_id if selected else None,
        "candidates": [candidate.to_dict() for candidate in items],
        "policy": {
            "publish_failed_candidate": False,
            "ranking": ["lyrics", "timing", "technical_quality", "context_fit"],
            "maximum_repaint_rounds": 2,
            "scene_fragment_fallback": False,
            "tts_song_fallback": False,
        },
    }


class SongCandidatePipeline:
    """Generate, gate, locally repair, and select songs without publishing failures."""

    def __init__(
        self,
        backend: SongBackend,
        *,
        align: Any,
        technical_review: Any,
        context_review: Any | None = None,
        candidate_count: int = 8,
        maximum_repaint_rounds: int = 2,
    ) -> None:
        self.backend = backend
        self.align = align
        self.technical_review = technical_review
        self.context_review = context_review
        self.candidate_count = candidate_count
        self.maximum_repaint_rounds = maximum_repaint_rounds

    def _evaluate(self, spec: SongSpec, path: Path, candidate_id: str, seed: int) -> AudioCandidate:
        alignment = dict(self.align(path, spec))
        technical = dict(self.technical_review(path, spec))
        context = dict(self.context_review(path, spec)) if self.context_review else {}
        passed = bool(alignment.get("passed")) and bool(technical.get("passed"))
        return AudioCandidate(
            candidate_id=candidate_id,
            backend=self.backend.name,
            model_version=self.backend.model_version,
            seed=seed,
            media_path=str(path),
            alignment=alignment,
            technical_metrics=technical,
            context_scores=context,
            passed=passed,
        )

    def run(self, spec: SongSpec, output_dir: str | Path, *, seed: int) -> dict[str, Any]:
        feasibility = validate_song_feasibility(spec)
        if not feasibility["passed"]:
            raise ValueError(f"lyrics cannot fit fixed video timing: {feasibility['issues']}")
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        candidates: list[AudioCandidate] = []
        for index in range(self.candidate_count):
            candidate_seed = seed + index * 1601
            path = root / f"candidate_{index:02d}.wav"
            self.backend.generate(spec, path, seed=candidate_seed)
            candidate = self._evaluate(spec, path, f"candidate_{index:02d}", candidate_seed)
            candidates.append(candidate)
            if candidate.passed:
                break

        selected = select_passing_candidate(candidates)
        for repaint_round in range(self.maximum_repaint_rounds):
            if selected:
                break
            source = min(candidates, key=candidate_rank)
            action = repair_action_for_candidate(
                spec,
                source,
                repaint_round=repaint_round,
                maximum_repaint_rounds=self.maximum_repaint_rounds,
            )
            source.repairs.append(action)
            if action.kind != "repaint_region":
                break
            path = root / f"{source.candidate_id}_repaint_{repaint_round + 1:02d}.wav"
            repaint_seed = seed + 50021 + repaint_round * 1601
            self.backend.repaint(spec, Path(source.media_path), path, action, seed=repaint_seed)
            repaired = self._evaluate(
                spec,
                path,
                f"{source.candidate_id}_repaint_{repaint_round + 1:02d}",
                repaint_seed,
            )
            repaired.repairs = [*source.repairs]
            candidates.append(repaired)
            selected = select_passing_candidate(candidates)

        manifest = candidate_manifest(spec, candidates)
        if selected:
            accepted = root / "accepted_song.wav"
            shutil.copy2(selected.media_path, accepted)
            manifest["accepted_media_path"] = str(accepted)
        return manifest
