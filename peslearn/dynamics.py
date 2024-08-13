from typing import Sequence

import ase.units as units
from ase.atoms import Atoms
from ase.md.npt import NPT

import peslearn.calculators as calculator
import peslearn.structure as structure


def get_dynamics(
    config_file: str,
    models: str | Sequence[str],
    params: str,
):
    temperature_K, pressure_Gpa = _parse_params(params)
    atoms = structure.get_structure(config_file)
    if not structure.cell_is_upper_triangular(atoms):
        structure.rotate_to_upper_triangular_cell_(atoms)
    structure.set_velocities_(atoms, temperature_K)
    atoms.calc = calculator.get_calculator(models)

    dyn = NPT(
        atoms,
        _get_dt(atoms),
        temperature_K=temperature_K,
        ttime=_get_ttime(atoms, temperature_K),
        externalstress=pressure_Gpa * units.GPa if pressure_Gpa is not None else 0,
        pfactor=(
            _get_pfactor(atoms, temperature_K, pressure_Gpa)
            if pressure_Gpa is not None
            else None
        ),
    )

    return dyn


def _parse_params(params: str) -> tuple[float, float | None]:
    tempress = iter(params.split(","))
    temperature_K = float(next(tempress))
    try:
        pressure_Gpa = float(next(tempress))
    except StopIteration:
        pressure_Gpa = None
    return temperature_K, pressure_Gpa


def _get_dt(atoms: Atoms) -> float:
    # TODO: better way to determine timestep
    min_mass = min(atoms.get_masses())
    if min_mass < 1.5:  # Hydrogen
        x = 0.5
    elif min_mass < 11.0:  # up to Be
        x = 1.0
    elif min_mass < 40.0:  # up to Ar
        x = 2.5
    else:
        x = 5.0
    return x * units.fs


def _get_ttime(atoms: Atoms, temperature_K: float) -> float:
    # TODO: Implement this; for now, just return a constant
    return 100 * units.fs


def _get_pfactor(atoms: Atoms, temperature_K: float, pressure_Gpa: float) -> float:
    # TODO: Implement this; for now, just return a constant
    return 1000 * units.fs
