"""
Tests for the ErfGauOptimizer classes
"""

import numpy as np

from horton.io.iodata import IOData
from horton.gbasis.gobasis import get_gobasis
from optimizer.hf_lda import *
from optimizer.minimizer import *
from optimizer.integrals import *

def test_erf_lr():
    """
    Test the FCI energy of the long-range
    Hamiltonian, using the error function
    """
    mol = IOData(numbers=np.array([2]), coordinates=np.array([[0., 0., 0.]]))
    obasis = get_gobasis(mol.coordinates, mol.numbers, 'aug-cc-pvdz')
    one_approx = ['standard']
    two_approx = ['erfgau']
    c = 1.9*0.9
    a = (1.5*0.9)**2
    pars = [[0.9, c, a]]
    kin, natt, olp, er = prepare_integrals(mol, obasis)
    na = 1
    nb = 1
    er_ref = obasis.compute_electron_repulsion()
    hf_energy, core_energy, orbs = compute_hf(mol, obasis, na, nb, kin, natt, er, olp)
    modintegrals = IntegralsWrapper(mol, obasis, one_approx, two_approx, pars, orbs)
    # Transform integrals to NO basis
    (one_mo_ref,), (two_mo_ref,) = transform_integrals(kin + natt, er_ref, 'tensordot', orbs[0])
    refintegrals = [one_mo_ref, two_mo_ref]
    optimizer = FullErfGauOptimizer(obasis.nbasis, core_energy, modintegrals, refintegrals, na, nb, mu=0.9)
    optimizer.optype = 'standard'
    cpars = [c, a]
    energy = optimizer.compute_energy(cpars)

