# Optimizer of Long-Range model Hamiltonians
# Copyright (C) 2018 Cristina E. Gonzalez-Espinoza
# <gonzalce@mcmaster.ca>.
# 
# This file is part of LROptimizer.
# 
# LROptimizer is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
# 
# LROptimizer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>

"""Auxiliary functions for the interaction with CI programs"""

import os
import math
import numpy as np

import pyci
from wfns.ham.chemical import ChemicalHamiltonian
from wfns.wfn.ci.base import CIWavefunction
from wfns.wfn.ci.fci import FCI
from wfns.wfn.ci.cisd import CISD
from wfns.solver.ci import brute

__all__ = ['get_ci_info_from_ciflow', 'get_ci_sd_pyci', 'get_larger_ci_coeffs',
           'get_ci_sd_pyci', 'compute_ci_fanCI', 'compute_my_dm', 'compute_FCI']


def get_ci_info_from_ciflow(fn):
    """Read slater determinants and their coeffs from
    CIFlow outputfile.

    **Arguments**
    fn
        Name of the output file.

    **Returns**

    civec
        A list of integers that represent the occupations
        in each Slater determinant, ordered as 0bbetaalpha.

    coeffs
        A numpy array with the coefficients of each Slater determinant.
    """
    # Get info from file
    beginning = 'upbitstring'
    end = '#Significant determinants: '
    info = []
    with open(fn, 'r') as fcifile:
        for line in fcifile:
            if beginning in line:
                break
        for line in fcifile:
            if end not in line:
                info.append(line.split('\t'))
            else:
                break
    civec = []
    coeffs = []
    for sd in info:
        # get the civec
        vec = '0b%s%s' % (sd[1],sd[2])
        vec = int(vec,2)
        civec.append(vec)
        # get the coeffs
        cs = sd[3].split('\n')[0]
        cs = float(cs)
        coeffs.append(cs)
    return civec, coeffs


def get_ci_sd_pyci(nbasis, wfn):
    """
    Get Slater determinants and coefficients from PyCI

    Arguments
    ---------
    nbasis: int
        Total number of basis functions.
    wfn: CIWavefunction
        Wavefunction object from PyCI
    """
    sd_vec = []
    # Loop over civectors and make strings
    for i in wfn:
        # Convert to bitstring
        a = bin(i[0][0]).split('b')[1]
        b = bin(i[1][0]).split('b')[1]
        # Add zeros when missing
        amiss = nbasis - len(a)
        bmiss = nbasis - len(b)
        a = '0'*amiss + a
        b = '0'*bmiss + b
        vec = '0b%s%s' % (b, a)
        # Convert bitstring to integer
        vec = int(vec,2)
        sd_vec.append(vec)
    return sd_vec


def get_larger_ci_coeffs(coeffs, limit=1e-4):
    """
    Find and count the CI expansion coefficients with
    values above certain limit.

    Returns:
    ccount: int
        Number of coefficients greater than limit value.
    location: np darray (ccount,)
        Mask array with the location of the desired coefficients.
    """

    location = np.where(abs(coeffs)>limit)[0]
    ccount = len(location)
    return ccount, location


def get_dets_pyci(wfn):
    """
    Get alpha and beta occupations for all the
    determinants in wfn.

    Arguments:
    ----------
    wfn: PyCI WaveFunction object
        Wavefunction.
    """
    alphas = []; betas = []
    # Iterate over all determinants
    for idx, det in wfn.iter_dets():
        a, b = det
        alphas.append(list(a))
        betas.append(list(b))
    return alphas, betas


def compute_ci_fanCI(nelec, nbasis, one, two, core_energy, civec=None, full=True):
    """
    Compute CI energy
    
    Arguments
    ---------
    nelec: int
        Total number of electrons.
    nbasis: int
        Number of spatial basis functions.
    one: ndarray (nbasis, nbasis)
        One electron integrals in MO basis.
    two: ndarray (nbasis, nbasis, nbasis, nbasis)
        Two electron integrals in MO basis.
    civec: list of int
        List with integer of the (bit string) determinants.
        If provided a CI calculation with those determinants
        will be performed. If None, FCI is performed.
    full: bool
        Whether to do Full-CI or CISD. Default to True for FCI,
        change to False for CISD.

    Returns:
    CI Ground State Energy, Groud State CI coefficients
    """
    nspin = nbasis*2
    if civec is not None:
        ci = CIWavefunction(nelec, nspin, sd_vec=civec)
    else:
        if full:
            ci = FCI(nelec, nspin)
        else:
            ci = CISD(nelec, nspin)
        civec = ci.sd_vec
    ham = ChemicalHamiltonian(one, two, orbtype='restricted', energy_nuc_nuc=core_energy)

    # optimize
    energies, coeffs = brute(ci, ham)
    return energies[0] + core_energy, coeffs[0], civec


def compute_FCI(nbasis, core_energy, one_int, two_int, na, nb, ncore=0, state=0):
    """Compute FCI energy with PyCI code

    Arguments:
    ----------
    nbasis: int
        Number of basis functions
    core_energy: float
        Energy of nuclear repulsion
    one_int: np ndarray(nbasis, nbasis)
        One-electron integrals
    two_int: np ndarray((nbasis, nbasis, nbasis, nbasis))
        Two-electron integrals
    na/nb: int
        Number of alpha/beta electrons
    ncore: int
        Number of core ORBITALS, takes into account two electrons
    state: int
        State of preference. Default to zero for the ground state.

    Returns:
    cienergy: float
        FCI energy
    cicoeffs:
        FCI expansion coefficients
    civec:
        FCI vectors (Slater determinants occupations)
    """
    # Using PyCI to compute FCI energy
    two_int = two_int.reshape(two_int.shape[0]**2, two_int.shape[1]**2)
    ciham = pyci.Hamiltonian(core_energy, one_int, two_int)
    wfn = pyci.FullCIWavefunction(nbasis, na, nb)
    
    # Compute the energy
    solver = pyci.SparseSolver(wfn, ciham)
    solver()
    cienergy = solver.eigenvalues()[state]
    cicoeffs = solver.eigenvectors().flatten()

    # Get the Slater determinats
    civec = get_ci_sd_pyci(nbasis, wfn)
    return cienergy, cicoeffs, civec

def compute_my_dm(dm1, orb, nbasis):
    dmbar = np.zeros((nbasis, nbasis))
    for k in range(nbasis):
        for l in range(nbasis):
            for mu in range(nbasis):
                for v in range(nbasis):
                    dmbar[mu, v] += dm1[k, l]*orb.coeffs[mu,k]*orb.coeffs[v,l]
    return dmbar
