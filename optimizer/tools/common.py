"""
Stolen from FanCI:
Handy tools for lazy people.
"""

import os
import glob

# General tools

def find_datafile(file_name):
    """Find file from data directory.

    Arguments
    ---------
    file_name : str
        Name of the file

    Returns
    -------
    file_path : str
        Absolute path of the file.

    Raises
    ------
    IOError
        If file cannot be found.
        If more than one file is found.

    """
    try:
        rel_path, = glob.glob(os.path.join(os.path.dirname(__file__), '..', 'data', file_name))
    except ValueError:
        raise IOError('Having trouble finding the file, {0}.'.format(file_name))


# Density related tools

def onedm_from_orb(nbasis, dm, *orbs):
    """
    Slow and simple way to compute the HF density matrix from FCI 1DM
    
    Arguments
    ---------
    nbasis: int
        Total number of spatial basis functions.
    dm: np ndarray((nbasis, nbasis))
        1-DM from CI calculations
    orbs: Orbital
        Orbital or list of molecular orbitals objects from Horton.

    """
    if len(orb) == 2:
        orb1 = orbs[0]; orb2 = orbs[1]
    elif len(orb) > 2:
        raise ValueError("Only one or two sets of orbitals are accepted.")
    else:
        orb1 = orbs; orb2 = orb1

    # Create empty array
    dmbar = np.zeros((nbasis, nbasis))
    for k in range(nbasis):
        for l in range(nbasis):
            for mu in range(nbasis):
                for v in range(nbasis):
                    dmbar[mu, v] += dm[k, l]*orb1.coeffs[mu,k]*orb2.coeffs[v,l] 
    return dmbar


def compute_density_from_dm(obasis, gridpoints, npoints, dm, *orbs):
    """
    Compute density on a grid from the FCI 1DM
    
    Arguments
    ---------
    obasis: GOBasis
        Horton object with all the information and functions related with
        basis sets and integrals.
    npoints: int
        Size of the grid to be used.
    gridpoints: np ndarray((npoints,3))
        Coordinates of the grid points where to evaluate the density.
    """
    # Use all the orbitals
    iorbs = np.array(range(obasis.nbasis))
    # Compute the orbitals in each point of space.
    if len(orb) == 2:
        orb1 = obasis.compute_grid_orbitals_exp(orbs[0], gridpoints, iorbs)
        orb2 = obasis.compute_grid_orbitals_exp(orbs[1], gridpoints, iorbs)
    elif len(orb) > 2:
        raise ValueError("Only one or two sets of orbitals are accepted.")
    else:
        orb1 = obasis.compute_grid_orbitals(orbs[0], gridpoints, iorbs)
        orb2 = orb1
    # Create empty array to store the density
    density = np.zeros(npoints)
    for i in range(obasis.nbasis):
        for j in range(obasis.nbasis):
            density += dm[i,j]*orb1[:,i]*orb2[:,j]
    return density


def get_density_error(density1, density2):
    """The Root-Mean-Square error between two densities on a grid"""
    return np.sqrt(np.sum((np.power(density1 - density2,2))))
