import math

from src.break_scoring.calibration import EmpiricalCDF, harmonic_mean
from src.break_scoring.evidence_score import (
    evidence_aware_break_score,
    text_evidence_reliability,
)

from src.break_scoring.evidence_score import acoustic_pause_margin


def test_acoustic_margin_uses_nearest_side():
    assert acoustic_pause_margin(2.0, 5.0) == 2.0


def test_acoustic_margin_handles_one_missing_side():
    assert acoustic_pause_margin(None, 14.17) == 14.17
    assert acoustic_pause_margin(3.5, None) == 3.5


def test_acoustic_margin_none_when_both_unavailable():
    assert acoustic_pause_margin(None, None) is None
    
def test_empirical_cdf_ordering():
    cdf = EmpiricalCDF.fit([1.0, 2.0, 3.0, 4.0])

    assert cdf.transform(0.0) == 0.0
    assert cdf.transform(1.0) == 0.25
    assert cdf.transform(2.5) == 0.50
    assert cdf.transform(4.0) == 1.0
    assert cdf.transform(100.0) == 1.0


def test_harmonic_mean():
    assert harmonic_mean(10, 10) == 10
    assert harmonic_mean(0, 10) == 0
    assert harmonic_mean(10, 20) < 15


def test_missing_text_does_not_penalize():
    without_text = evidence_aware_break_score(
        acoustic_percentile=0.8,
        visual_percentile=0.6,
    )

    assert without_text == 70.0


def test_reliable_text_contributes():
    score = evidence_aware_break_score(
        acoustic_percentile=0.8,
        visual_percentile=0.6,
        text_percentile=0.9,
        text_reliability=1.0,
    )

    expected = 100 * ((0.8 + 0.6 + 0.9) / 3)

    assert math.isclose(score, expected)


def test_short_text_is_downweighted():
    effective, reliability = text_evidence_reliability(
        left_characters=2,
        right_characters=100,
        reference_text_length=50,
    )

    assert effective < 4
    assert reliability < 0.1


def test_normal_text_reliability_caps_at_one():
    effective, reliability = text_evidence_reliability(
        left_characters=100,
        right_characters=100,
        reference_text_length=50,
    )

    assert effective == 100
    assert reliability == 1.0