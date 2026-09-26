from pathlib import Path

from src.break_scoring.calibration import EmpiricalCDF
from src.optimization.break_optimizer import EABS_NEUTRAL_BASELINE


def test_heldout_uses_frozen_eabs_neutral_baseline():
    assert EABS_NEUTRAL_BASELINE == 50.0


def test_empirical_cdf_transform_does_not_refit():
    frozen = EmpiricalCDF.from_dict(
        {
            "reference_values": [1.0, 2.0, 3.0],
        }
    )

    assert frozen.transform(2.5) == 2 / 3
    assert frozen.values == (1.0, 2.0, 3.0)


def test_heldout_runner_exists():
    assert Path("scripts/evaluate_feluda.py").is_file()



def test_no_safe_brand_becomes_no_fill_delivery():
    from scripts.evaluate_feluda import build_delivery_subset

    optimized = {
        "videos": [
            {
                "selected_breaks": [
                    {
                        "timestamp_seconds": 100.0,
                    },
                    {
                        "timestamp_seconds": 500.0,
                    },
                ],
                "selected_break_count": 2,
            }
        ],
        "summary": {
            "selected_breaks_total": 2,
        },
    }

    matches = {
        "breaks": [
            {
                "timestamp_seconds": 100.0,
                "selected_brand": None,
            },
            {
                "timestamp_seconds": 500.0,
                "selected_brand": {
                    "brand_id": "safe_brand",
                },
            },
        ],
        "summary": {
            "break_count": 2,
            "negative_context_violations": 0,
        },
    }

    delivery_optimized, delivery_matches = (
        build_delivery_subset(
            optimized,
            matches,
        )
    )

    assert (
        len(
            delivery_optimized[
                "videos"
            ][0][
                "selected_breaks"
            ]
        )
        == 1
    )

    assert (
        delivery_matches[
            "breaks"
        ][0][
            "selected_brand"
        ][
            "brand_id"
        ]
        == "safe_brand"
    )
