"""
Tests for the ErfGauOptimizer classes
"""

import numpy as np
from numpy.linalg import eigh

from horton.io.iodata import IOData
from horton.gbasis.gobasis import get_gobasis
from horton.meanfield.orbitals import Orbitals
from optimizer.hf_lda import *
from optimizer.minimizer import *
from optimizer.integrals import *
from lrtools.ciwrapper import compute_ci_fanCI
from wfns.ham.density import density_matrix


def get_naturalorbs(nbasis, dm1, orb_alpha):
    """
    Get the natural orbitals from the 1-DM
    """
    dm1 /= 2.0
    norb = Orbitals(nbasis)
    # Diagonalize and compute eigenvalues
    evals, evecs = eigh(dm1)
    # Reorder the matrix and the eigenvalues
    evecs1 = evecs.copy()
    evecs1[:] = evecs[:,::-1]
    norb.occupations[::-1] = evals
    # Get the natural orbitals
    norb.coeffs[:] = np.dot(orb_alpha.coeffs, evecs1)
    norb.energies[:] = 0.0
    return norb, evecs


def test_hf_orbitals():
    """
    Check if HF orbitals are orthormal
    """
    mol = IOData(numbers=np.array([2]), coordinates=np.array([[0., 0., 0.]]))
    obasis = get_gobasis(mol.coordinates, mol.numbers, 'aug-cc-pvdz')
    kin, natt, olp, er = prepare_integrals(mol, obasis)
    er_ref = obasis.compute_electron_repulsion()
    hf_energy, core_energy, orbs = compute_hf(mol, obasis, na, nb, kin, natt, er, olp)
    # Define a numerical integration grid needed for integration
    orblist = np.array(range(obasis.nbasis))
    for i in orblist:
        for j in orblist:
            overlap = 0.0
            for a in orblist:
                for b in orblist:
                    overlap += orbs[0].coeffs[a,i]*orbs[0].coeffs[b,j]*olp[a,b]
            if i == j:
                assert abs(overlap - 1.0) < 1e-5


def test_natural_orbitals():
    """
    Check if HF orbitals are orthormal
    """
    mol = IOData(numbers=np.array([4]), coordinates=np.array([[0., 0., 0.]]))
    obasis = get_gobasis(mol.coordinates, mol.numbers, 'cc-pvdz')
    one_approx = ['standard']
    two_approx = ['erfgau']
    c = 1.9*0.9
    a = (1.5*0.9)**2
    pars = [[0.9, c, a]]
    kin, natt, olp, er = prepare_integrals(mol, obasis)
    na = 2
    nb = 2
    nelec = na + nb
    er_ref = obasis.compute_electron_repulsion()
    hf_energy, core_energy, orbs = compute_hf(mol, obasis, na, nb, kin, natt, er, olp)
    # Define a numerical integration grid needed for integration
    grid = BeckeMolGrid(mol.coordinates, mol.numbers, mol.pseudo_numbers)

    modintegrals = IntegralsWrapper(mol, obasis, one_approx, two_approx, pars, orbs)
    # Transform integrals to NO basis
    (one_mo,), (two_mo,) = transform_integrals(kin + natt, er_ref, 'tensordot', orbs[0])
    refintegrals = [one_mo, two_mo]
    optimizer = FullErfGauOptimizer(obasis.nbasis, core_energy, modintegrals, refintegrals, na, nb, mu=0.9)
    optimizer.set_olp(olp)
    # Make occupations for determinants
    nbasis = obasis.nbasis
    civec0 = []
    a0 = list('0' * nbasis)
    a0[0] = '1'; a0[1] = '1'
    b0 = a0
    a0 = a0[::-1]; a0 = ''.join(a0)
    b0 = b0[::-1]; b0 = ''.join(b0)
    vec = '0b%s%s' % (b0, a0)
    # Convert bitstring to integer
    vec = int(vec,2)
    civec0.append(vec)
    civec1 = []
    civec1.append(vec)
    a1 = list('0' * nbasis)
    a1[0] = '1'; a1[2] = '1'
    b1 = a1
    a1 = a1[::-1]; a1 = ''.join(a1)
    b1 = b1[::-1]; b1 = ''.join(b1)
    vec1 = '0b%s%s' % (b1, a1)
    vec1 = int(vec1,2)
    civec1.append(vec1)

    cienergy0, cicoeffs0, civec = compute_ci_fanCI(nelec, nbasis, one_mo, two_mo,
                                                   core_energy, civec0)
    (dm1_0,), (dm2_0,) = density_matrix(cicoeffs0, civec, obasis.nbasis)
    norb0, umatrix = get_naturalorbs(nbasis, dm1_0, orbs[0])
    orblist = np.array(range(obasis.nbasis))
    print "norb occs ", norb0.occupations
    assert sum(norb0.occupations) == 2.0
    for i in orblist:
        for j in orblist:
            overlap = np.dot(norb0.coeffs[:,i], np.dot(olp, norb0.coeffs[:,j]))
            print "overlap ", overlap, i
            if i == j:
                assert abs(overlap - 1.0) < 1e-10

#test_natural_orbitals()

def test_erfgau_lr():
    """
    Test the FCI energy of the long-range
    Hamiltonian, using the error function
    """
    mol = IOData(numbers=np.array([2]), coordinates=np.array([[0., 0., 0.]]))
    obasis = get_gobasis(mol.coordinates, mol.numbers, 'cc-pvdz')
    one_approx = ['standard']
    two_approx = ['erfgau']
    mu = 1.0
    c = 1.9*mu
    a = (1.5*mu)**2
    pars = [[mu, c, a]]
    kin, natt, olp, er = prepare_integrals(mol, obasis)
    na = 1
    nb = 1
    er_ref = obasis.compute_electron_repulsion()
    hf_energy, core_energy, orbs = compute_hf(mol, obasis, na, nb, kin, natt, er, olp)
    modintegrals = IntegralsWrapper(mol, obasis, one_approx, two_approx, pars, orbs)
    # Transform integrals to MO basis
    (one_mo_ref,), (two_mo_ref,) = transform_integrals(kin + natt, er_ref, 'tensordot', orbs[0])
    refintegrals = [one_mo_ref, two_mo_ref]
    optimizer = FullErfGauOptimizer(obasis.nbasis, core_energy, modintegrals, refintegrals, na, nb, mu=mu)
    optimizer.set_olp(olp)
    optimizer.set_orb_alpha(orbs[0])
    optimizer.set_dm_alpha(orbs[0].to_dm())
    optimizer.optype = 'standard'
    cpars = [c, a]
    optimizer.naturals = True
    energy = optimizer.compute_energy(cpars)
    print "energy", energy
    print "mu energy", optimizer.mu_energy

#test_erfgau_lr()

def test_erfgau_truc():
    """
    Test the CI energy of the long-range
    Hamiltonian, using the error function
    """
    mol = IOData(numbers=np.array([2]), coordinates=np.array([[0., 0., 0.]]))
    obasis = get_gobasis(mol.coordinates, mol.numbers, 'cc-pvdz')
    one_approx = ['standard']
    two_approx = ['erfgau']
    mu = 1.0
    c = 1.9*mu
    a = (1.5*mu)**2
    pars = [[mu, c, a]]
    kin, natt, olp, er = prepare_integrals(mol, obasis)
    na = 1
    nb = 1
    er_ref = obasis.compute_electron_repulsion()
    hf_energy, core_energy, orbs = compute_hf(mol, obasis, na, nb, kin, natt, er, olp)
    modintegrals = IntegralsWrapper(mol, obasis, one_approx, two_approx, pars, orbs)
    # Transform integrals to MO basis
    (one_mo_ref,), (two_mo_ref,) = transform_integrals(kin + natt, er_ref, 'tensordot', orbs[0])
    refintegrals = [one_mo_ref, two_mo_ref]
    ndet = 20
    optimizer = TruncErfGauOptimizer(obasis.nbasis, core_energy, modintegrals, refintegrals, na, nb, mu=mu, ndet=ndet)
    optimizer.set_olp(olp)
    optimizer.set_orb_alpha(orbs[0])
    optimizer.set_dm_alpha(orbs[0].to_dm())
    optimizer.optype = 'standard'
    cpars = [c, a]
    energy = optimizer.compute_energy(cpars)
    print "energy", energy
    print "mu energy", optimizer.mu_energy
test_erfgau_truc()
