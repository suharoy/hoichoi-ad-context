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
