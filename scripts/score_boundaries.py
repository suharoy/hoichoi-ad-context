"""Fit development-only calibration and compute EABS break suitability."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from src.break_scoring.calibration import (
    EmpiricalCDF,
    harmonic_mean,
    percentile_summary,
)
from src.break_scoring.evidence_score import (
    acoustic_pause_margin,
    evidence_aware_break_score,
    text_evidence_reliability,
)


ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "outputs" / "dev" / "boundary-profile-multimodal.json"
OUTPUT = ROOT / "outputs" / "dev" / "boundary-profile-scored.json"
CALIBRATION_PATH = ROOT / "configs" / "break_score_calibration.json"

EXPECTED_DEV_VIDEOS = {
    "bhojon_bilashi",
    "indubala_bhaater_hotel",
    "mandaar",
    "mohanagar",
    "money_honey",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_stem(name: str) -> str:
    return Path(name).stem


def is_ad_eligible(candidate: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if candidate.get("speech_crosses_boundary") is True:
        reasons.append("speech_crosses_boundary")

    speech_context = candidate.get("speech_context") or {}

    if speech_context.get("inside_detected_speech") is True:
        reasons.append("inside_detected_speech")

    return len(reasons) == 0, reasons


def acoustic_raw(candidate: dict) -> float | None:
    context = candidate.get("speech_context") or {}

    return acoustic_pause_margin(
        context.get("seconds_since_speech_end"),
        context.get("seconds_until_speech_start"),
    )


def visual_raw(candidate: dict) -> float | None:
    semantics = candidate.get("visual_semantics") or {}
    value = semantics.get("context_change")

    if value is None:
        return None

    value = float(value)

    if not np.isfinite(value):
        raise ValueError("Non-finite contextual visual change")

    return value


def valid_text(candidate: dict) -> bool:
    semantics = candidate.get("text_semantics")

    if not semantics:
        return False

    if semantics.get("available") is not True:
        return False

    change = semantics.get("semantic_change")
    left = semantics.get("left_characters")
    right = semantics.get("right_characters")

    if change is None or not left or not right:
        return False

    return (
        np.isfinite(float(change))
        and int(left) > 0
        and int(right) > 0
    )


def effective_text_length(candidate: dict) -> float:
    semantics = candidate["text_semantics"]

    return harmonic_mean(
        float(semantics["left_characters"]),
        float(semantics["right_characters"]),
    )


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return None

    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def rankdata(values: list[float]) -> np.ndarray:
    """
    Average ranks for ties, sufficient for Spearman correlation
    without requiring scipy.
    """
    array = np.asarray(values, dtype=np.float64)

    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)

    sorted_values = array[order]

    start = 0
    while start < len(array):
        end = start + 1

        while (
            end < len(array)
            and sorted_values[end] == sorted_values[start]
        ):
            end += 1

        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank

        start = end

    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None

    return pearson(
        rankdata(x).tolist(),
        rankdata(y).tolist(),
    )


def correlations(
    scores: list[float],
    component: list[float],
) -> dict[str, float | None]:
    return {
        "pearson": pearson(scores, component),
        "spearman": spearman(scores, component),
    }


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    source_hash_before = sha256(SOURCE)

    original = json.loads(SOURCE.read_text(encoding="utf-8"))
    scored = copy.deepcopy(original)

    videos = scored.get("videos")

    if not isinstance(videos, list):
        raise ValueError("Expected top-level videos list")

    actual_videos = {
        video_stem(video["video"])
        for video in videos
    }

    if actual_videos != EXPECTED_DEV_VIDEOS:
        raise ValueError(
            f"Unexpected development videos: {sorted(actual_videos)}"
        )

    if any("feluda" in name.lower() for name in actual_videos):
        raise ValueError("Feluda must not be used for Stage 4 calibration")

    eligible_candidates: list[dict] = []

    acoustic_reference: list[float] = []
    visual_reference: list[float] = []

    text_reference: list[float] = []
    text_lengths: list[float] = []

    total_candidates = 0
    speech_safe_candidates = 0
    rejected_crossing = 0
    rejected_inside = 0

    # ---------------------------------------------------------
    # Fit development-only reference distributions.
    # ---------------------------------------------------------

    for video in videos:
        for candidate in video["candidates"]:
            total_candidates += 1

            if candidate.get("speech_safe") is True:
                speech_safe_candidates += 1

            eligible, reasons = is_ad_eligible(candidate)

            if "speech_crosses_boundary" in reasons:
                rejected_crossing += 1

            if "inside_detected_speech" in reasons:
                rejected_inside += 1

            if not eligible:
                continue

            eligible_candidates.append(candidate)

            acoustic = acoustic_raw(candidate)
            visual = visual_raw(candidate)

            if acoustic is not None:
                acoustic_reference.append(acoustic)

            if visual is not None:
                visual_reference.append(visual)

            if valid_text(candidate):
                semantics = candidate["text_semantics"]

                text_reference.append(
                    float(semantics["semantic_change"])
                )

                text_lengths.append(
                    effective_text_length(candidate)
                )

    if not acoustic_reference:
        raise ValueError("No acoustic calibration values")

    if not visual_reference:
        raise ValueError("No visual calibration values")

    if not text_reference:
        raise ValueError("No text calibration values")

    acoustic_cdf = EmpiricalCDF.fit(acoustic_reference)
    visual_cdf = EmpiricalCDF.fit(visual_reference)
    text_cdf = EmpiricalCDF.fit(text_reference)

    reference_text_length = float(np.median(text_lengths))

    if reference_text_length <= 0:
        raise ValueError("Invalid reference text length")

    calibration = {
        "version": 1,
        "method": "development_empirical_cdf",
        "score_name": "EABS",
        "source_sha256": source_hash_before,
        "development_videos": sorted(actual_videos),
        "acoustic_reference_count": len(acoustic_reference),
        "visual_reference_count": len(visual_reference),
        "text_reference_count": len(text_reference),
        "reference_text_length": reference_text_length,
        "acoustic": acoustic_cdf.to_dict(),
        "visual": visual_cdf.to_dict(),
        "text": text_cdf.to_dict(),
    }

    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)

    CALIBRATION_PATH.write_text(
        json.dumps(
            calibration,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Score every candidate.
    # ---------------------------------------------------------

    eligible_count = 0
    scored_count = 0
    unscored_eligible = 0
    valid_text_count = 0
    missing_text_count = 0

    score_values: list[float] = []

    score_vs_acoustic: list[float] = []
    acoustic_components: list[float] = []

    score_vs_visual: list[float] = []
    visual_components: list[float] = []

    score_vs_text: list[float] = []
    text_components: list[float] = []

    diagnostics: list[dict] = []
    text_audit: list[dict] = []

    for video in videos:
        video_name = video["video"]

        for candidate in video["candidates"]:
            eligible, reasons = is_ad_eligible(candidate)

            break_score = {
                "ad_eligible": eligible,
                "rejection_reasons": reasons,
                "scorable": False,
                "acoustic": {
                    "raw_pause_margin": None,
                    "percentile": None,
                },
                "visual": {
                    "raw_context_change": None,
                    "percentile": None,
                },
                "text": {
                    "available": False,
                    "raw_semantic_change": None,
                    "effective_length": 0.0,
                    "reliability": 0.0,
                    "percentile": None,
                },
                "eabs": None,
            }

            candidate["break_score"] = break_score

            if not eligible:
                continue

            eligible_count += 1

            acoustic = acoustic_raw(candidate)
            visual = visual_raw(candidate)

            break_score["acoustic"]["raw_pause_margin"] = acoustic
            break_score["visual"]["raw_context_change"] = visual

            # Both modalities are required for EABS.
            if acoustic is None or visual is None:
                unscored_eligible += 1
                continue

            acoustic_percentile = acoustic_cdf.transform(acoustic)
            visual_percentile = visual_cdf.transform(visual)

            break_score["acoustic"]["percentile"] = acoustic_percentile
            break_score["visual"]["percentile"] = visual_percentile

            text_percentile = None
            reliability = 0.0
            effective_length = 0.0
            raw_text_change = None

            if valid_text(candidate):
                valid_text_count += 1

                semantics = candidate["text_semantics"]

                raw_text_change = float(
                    semantics["semantic_change"]
                )

                effective_length, reliability = (
                    text_evidence_reliability(
                        semantics["left_characters"],
                        semantics["right_characters"],
                        reference_text_length,
                    )
                )

                text_percentile = text_cdf.transform(
                    raw_text_change
                )

                break_score["text"] = {
                    "available": True,
                    "raw_semantic_change": raw_text_change,
                    "effective_length": effective_length,
                    "reliability": reliability,
                    "percentile": text_percentile,
                }

            else:
                missing_text_count += 1

            score = evidence_aware_break_score(
                acoustic_percentile,
                visual_percentile,
                text_percentile=text_percentile,
                text_reliability=reliability,
            )

            break_score["scorable"] = True
            break_score["eabs"] = score

            scored_count += 1
            score_values.append(score)

            score_vs_acoustic.append(score)
            acoustic_components.append(acoustic_percentile)

            score_vs_visual.append(score)
            visual_components.append(visual_percentile)

            if text_percentile is not None:
                score_vs_text.append(score)
                text_components.append(text_percentile)

            diagnostic = {
                "video": video_name,
                "timestamp": candidate["timestamp_seconds"],
                "centered_pause": candidate.get(
                    "centered_pause"
                ),
                "raw_pause_margin": acoustic,
                "acoustic_percentile": acoustic_percentile,
                "context_change": visual,
                "visual_percentile": visual_percentile,
                "text_change": raw_text_change,
                "text_reliability": reliability,
                "text_percentile": text_percentile,
                "eabs": score,
            }

            diagnostics.append(diagnostic)

            if raw_text_change is not None:
                text_audit.append(
                    {
                        **diagnostic,
                        "effective_text_length": effective_length,
                        "weighted_text_contribution": (
                            reliability * text_percentile
                        ),
                    }
                )

    # ---------------------------------------------------------
    # Validation.
    # ---------------------------------------------------------

    if not score_values:
        raise ValueError("No EABS scores produced")

    if not all(
        np.isfinite(value)
        and 0.0 <= value <= 100.0
        for value in score_values
    ):
        raise ValueError("Invalid EABS value")

    for video in videos:
        for candidate in video["candidates"]:
            score = candidate["break_score"]["eabs"]

            if (
                candidate["break_score"]["ad_eligible"] is False
                and score is not None
            ):
                raise ValueError(
                    "Rejected candidate received EABS"
                )

    # Remove the only new key and ensure all prior content matches.
    stripped = copy.deepcopy(scored)

    for video in stripped["videos"]:
        for candidate in video["candidates"]:
            candidate.pop("break_score", None)

    prior_fields_preserved = stripped == original

    if not prior_fields_preserved:
        raise ValueError("Prior JSON fields were modified")

    source_hash_after = sha256(SOURCE)

    if source_hash_after != source_hash_before:
        raise ValueError("Source multimodal JSON changed")

    # Validate frozen CDF reload/behavior.
    calibration_reloaded = json.loads(
        CALIBRATION_PATH.read_text(encoding="utf-8")
    )

    check_cdf = EmpiricalCDF.from_dict(
        calibration_reloaded["visual"]
    )

    cdf_validation = {
        "below_range": check_cdf.transform(
            check_cdf.values[0] - 1.0
        ),
        "median_range": check_cdf.transform(
            float(np.median(check_cdf.values))
        ),
        "above_range": check_cdf.transform(
            check_cdf.values[-1] + 1.0
        ),
    }

    if cdf_validation["below_range"] != 0.0:
        raise ValueError("CDF below-range validation failed")

    if cdf_validation["above_range"] != 1.0:
        raise ValueError("CDF above-range validation failed")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(
            scored,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    diagnostics_sorted = sorted(
        diagnostics,
        key=lambda item: item["eabs"],
        reverse=True,
    )

    # Stage-2 failure-mode audit:
    # examine highest raw text semantic changes and show how
    # reliability changes their influence.
    short_text_audit = sorted(
        text_audit,
        key=lambda item: item["text_change"],
        reverse=True,
    )[:15]

    report = {
        "candidate_counts": {
            "total_visual_boundaries": total_candidates,
            "speech_safe_candidates": speech_safe_candidates,
            "rejected_speech_crossing": rejected_crossing,
            "rejected_inside_detected_speech": rejected_inside,
            "ad_eligible": eligible_count,
            "scored": scored_count,
            "eligible_but_unscored": unscored_eligible,
            "scored_with_text": valid_text_count,
            "scored_without_text": missing_text_count,
        },
        "calibration": {
            "acoustic_reference_count": len(
                acoustic_reference
            ),
            "visual_reference_count": len(
                visual_reference
            ),
            "text_reference_count": len(
                text_reference
            ),
            "reference_text_length": reference_text_length,
            "acoustic_raw": percentile_summary(
                acoustic_reference
            ),
            "visual_raw": percentile_summary(
                visual_reference
            ),
            "text_raw": percentile_summary(
                text_reference
            ),
        },
        "eabs": percentile_summary(score_values),
        "correlations": {
            "eabs_acoustic": correlations(
                score_vs_acoustic,
                acoustic_components,
            ),
            "eabs_visual": correlations(
                score_vs_visual,
                visual_components,
            ),
            "eabs_text_valid_only": correlations(
                score_vs_text,
                text_components,
            ),
        },
        "highest_eabs": diagnostics_sorted[:15],
        "lowest_eabs": diagnostics_sorted[-15:],
        "high_semantic_change_text_audit": (
            short_text_audit
        ),
        "validation": {
            "source_sha256": source_hash_before,
            "source_unchanged": (
                source_hash_before == source_hash_after
            ),
            "prior_fields_preserved": (
                prior_fields_preserved
            ),
            "development_videos": sorted(
                actual_videos
            ),
            "feluda_processed": False,
            "cdf_synthetic_checks": cdf_validation,
        },
    }

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"\nSaved scored profile: {OUTPUT}")
    print(f"Saved frozen calibration: {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()