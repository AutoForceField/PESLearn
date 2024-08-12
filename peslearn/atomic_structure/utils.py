import numpy as np
from ase import Atoms
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

__all__ = [
    "set_velocities_",
    "rotate_to_upper_triangular_cell_",
    "cell_is_upper_triangular",
    "atoms_are_equal",
]


def set_velocities_(atoms: Atoms, temp_K: float, overwrite=False) -> None:
    """
    Set the velocities of the atoms to a Maxwell-Boltzmann
    distribution at the given temperature. If the velocities
    are already set, they are not overwritten unless the
    `overwrite` flag is set to True.
    """
    vel_ini = atoms.get_velocities()
    if np.allclose(vel_ini, 0) or overwrite:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temp_K)
        Stationary(atoms)
        ZeroRotation(atoms)


def rotate_to_upper_triangular_cell_(atoms: Atoms) -> None:
    """
    Apply a rotation such that the cell is transformed
    into a upper triangular 3x3 matrix. This constraint
    is required for some algorithms e.g. ase.md.NPT.
    """

    d = atoms.get_all_distances()
    v = atoms.get_volume()
    atoms.rotate(atoms.cell[2], v="z", rotate_cell=True)
    atoms.rotate([*atoms.cell[1][:2], 0], v="y", rotate_cell=True)
    dd = atoms.get_all_distances()
    vv = atoms.get_volume()
    assert np.allclose(d, dd)
    assert np.allclose(v, vv)
    assert np.allclose(atoms.cell.flat[[3, 6, 7]], 0, atol=1e-6), print(
        atoms.cell.flat[[3, 6, 7]]
    )
    atoms.cell.flat[[3, 6, 7]] = 0


def cell_is_upper_triangular(atoms: Atoms) -> bool:
    """
    Check if the cell is an upper triangular 3x3 matrix.
    """
    return all(atoms.cell.flat[[3, 6, 7]] == 0)


def atoms_are_equal(atoms1: Atoms, atoms2: Atoms) -> bool:
    if not (atoms1.get_global_number_of_atoms() == atoms2.get_global_number_of_atoms()):
        return False
    if not (all(atoms1.get_atomic_numbers() == atoms2.get_atomic_numbers())):
        return False
    if not (np.allclose(atoms1.get_cell(), atoms2.get_cell())):
        return False
    if not (np.allclose(atoms1.get_positions(), atoms2.get_positions())):
        return False
    if not (all(atoms1.get_pbc() == atoms2.get_pbc())):
        return False
    return True
