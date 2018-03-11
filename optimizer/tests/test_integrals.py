"""Tests for the integral wrapper"""

import numpy as np
from nose.tools import assert_raises

from horton.gbasis.gobasis import get_gobasis
from horton.io.iodata import IOData
from horton.meanfield.orbitals import Orbitals
from horton.meanfield.indextransform import split_core_active
from optimizer.hf_lda import *
from optimizer.integrals import *

def test_standard_one():
    """
    Test the standard one-electron integrals.
    """
    function = compute_standard_one_integrals
    mol = [1,2]
    obasis = [2,3,1,3]
    # Check mol
    assert_raises(TypeError, function, mol, obasis)
    mol = IOData(numbers=np.array([1]), coordinates=np.array([[0.,0.,0]]))
    # Check obasis
    assert_raises(TypeError, function, mol, obasis)
    obasis = get_gobasis(mol.coordinates, mol.numbers, '3-21G')

    # Check the actual function
    kin = obasis.compute_kinetic()
    na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers)
    one_ref = kin + na
    one = function(mol, obasis)
    assert (one == one_ref).all()


def test_two_integrals():
    """
    Test function to compute modified two-electron integrals.
    """
    function = compute_two_integrals
    mol = IOData(numbers=np.array([1]), coordinates=np.array([[0.,0.,0]]))
    obasis = [0,1,2]
    approxs = [2]
    pars = [3]
    # Check obasis
    assert_raises(TypeError, function, obasis, approxs, pars)
    obasis = get_gobasis(mol.coordinates, mol.numbers, '3-21G')
    # Check approxs
    assert_raises(TypeError, function, obasis, approxs, pars)
    approxs = ['erf']
    # Check pars
    assert_raises(TypeError, function, obasis, approxs, pars)
    pars = [[0.1]]
    approxs = ['nada']
    # Check approxs options
    assert_raises(ValueError, function, obasis, approxs, pars)
    approxs = ['erf']
    # Check the actual function
    erf_ref = obasis.compute_erf_repulsion(0.1)
    erf = function(obasis, approxs, pars)
    assert (erf == erf_ref).all()
    gauss = obasis.compute_gauss_repulsion(1.0, 1.5)
    erfgau_ref = erf + gauss
    erfgau = function(obasis, ['erf', 'gauss'], [[0.1],[1.0,1.5]])
    assert (erfgau == erfgau_ref).all()


def test_two_exchange_doci():
    """
    Test the whole exchange for double excitations.
    """
    function = use_full_exchange_two_doci
    nbasis = 1.0
    two_full = [0,1]
    two_lr = [0,1]
    # Check obasis
    assert_raises(TypeError, function, nbasis, two_full, two_lr)
    nbasis = 3
    # Check two_full
    assert_raises(TypeError, function, nbasis, two_full, two_lr)
    two_full = np.array([[1, 2],[3, 4]])
    # Check two_lr
    assert_raises(TypeError, function, nbasis, two_full, two_lr)
    two_lr = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
    # Check for shapes of two_full and two_lr
    assert_raises(ValueError, function, nbasis, two_full, two_lr)
    # Check for nbasis and shape consistency
    two_lr = two_full.copy()
    assert_raises(ValueError, function, nbasis, two_full, two_lr)


def test_integrals_wrapper_init():
    """
    Test the integral wrapper.
    """
    function = IntegralsWrapper
    mol = [1,2]
    obasis = [2,3,1,3]
    one_approx = [0,1]
    two_approx = [0]
    pars = [0,3]
    orbs = 'orbs'
    # Check mol
    assert_raises(TypeError, function, mol, obasis, one_approx, two_approx, pars, orbs)
    mol = IOData(numbers=np.array([1]), coordinates=np.array([[0.,0.,0]]))
    # Check obasis
    assert_raises(TypeError, function, mol, obasis, one_approx, two_approx, pars, orbs)
    obasis = get_gobasis(mol.coordinates, mol.numbers, '3-21G')
    # Check one_approx
    assert_raises(TypeError, function, mol, obasis, one_approx, two_approx, pars, orbs)
    one_approx = ['man']
    # Check options for one_approx
    assert_raises(ValueError, function, mol, obasis, one_approx, two_approx, pars, orbs)
    one_approx = ['standard']
    del function
    function = IntegralsWrapper
    # Check two_approx
    assert_raises(TypeError, function, mol, obasis, one_approx, two_approx, pars, orbs)
    two_approx = ['nada']
    # Check for options two_approx
    assert_raises(ValueError, function, mol, obasis, one_approx, two_approx, pars, orbs)
    two_approx = ['erf', 'gauss']
    # Check for pars
    assert_raises(TypeError, function, mol, obasis, one_approx, two_approx, pars, orbs)
    pars = [[0.1],[1.0, 1.5]]
    # Check for orbs
    assert_raises(TypeError, function, mol, obasis, one_approx, two_approx, pars, orbs)
    orbs = [Orbitals(obasis.nbasis)]


def test_two_integrals_wrapper_ncore():
    """
    Test function to compute modified two-electron wrapper with frozen core.
    """
    function = IntegralsWrapper
    mol = IOData(numbers=np.array([4]), coordinates=np.array([[0.,0.,0]]))
    obasis = get_gobasis(mol.coordinates, mol.numbers, '3-21G')
    one_approx = ['standard']
    two_approx = ['erf']
    pars = [[0.1]]
    na = 2
    nb = 2
    kin, natt, olp, er = prepare_integrals(mol, obasis)
    hf_energy, core_energy, orbs = compute_hf(mol, obasis, na, nb, kin, natt, er, olp)
    ncore = 1.0
    # Check ncore
    assert_raises(TypeError, function, mol, obasis, one_approx, two_approx, pars, orbs, ncore)
    ncore = 1
    # Check core energy
    assert_raises(ValueError, function, mol, obasis, one_approx, two_approx, pars, orbs, ncore)
    del function
    # Check actual function result
    one_ref = kin + natt
    erf_ref = obasis.compute_erf_repulsion(0.1)
    nactive = obasis.nbasis - ncore
    ints = IntegralsWrapper(mol, obasis, one_approx, two_approx, pars, orbs,
                            ncore, core_energy)
    one_mo, two_mo, core_energy = split_core_active(one_ref, erf_ref, core_energy,
                                                    orbs[0], ncore=ncore, nactive=nactive)

    assert core_energy == ints.core_energy
    assert (ints.one == one_mo).all()
    assert (ints.two == two_mo).all()
