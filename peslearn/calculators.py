from typing import Sequence

from ase.calculators.calculator import Calculator
from mace.calculators.foundations_models import mace_mp


def get_calculator(
    models: str | Sequence[str],
    device: str = "cuda",
    dtype: str = "float32",
) -> Calculator:
    if models == "mace-mp":
        return _get_mace_mp(device, dtype)
    else:
        raise ValueError(f"Unknown calculator: {models}")


def _get_mace_mp(device, dtype) -> Calculator:
    return mace_mp(
        model="small",
        device=device,
        default_dtype=dtype,
        dispersion=True,
        damping="bj",
        dispersion_xc="pbe",
    )
