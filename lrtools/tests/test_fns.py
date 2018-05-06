from lrtools.ciwrapper import *
from optimizer.hf_lda import *
import pyci


def test_get_ci_sd_pyci():
    nbasis = 5
    na = 2
    nb = 1
    wfn = pyci.FullCIWavefunction(nbasis, na, nb)
    sd_vec = get_ci_sd_pyci(nbasis, wfn)
    assert sd_vec == [35, 67, 131, 259, 515, 37, 69,
                      133, 261, 517, 38, 70, 134, 262,
                      518, 41, 73, 137, 265, 521, 42, 74,
                      138, 266, 522, 44, 76, 140, 268, 524,
                      49, 81, 145, 273, 529, 50, 82, 146, 274,
                      530, 52, 84, 148, 276, 532, 56, 88, 152, 280, 536]
#test_get_ci_sd_pyci()


def test_FCI():
    mol = IOData(numbers=np.array([2]), coordinates=np.array([[0., 0., 0.]]))
    obasis = get_gobasis(mol.coordinates, mol.numbers, 'cc-pvdz')
    na = 1
    nb = 1
    kin, natt, olp, er = prepare_integrals(mol, obasis)
    hf_energy, core_energy, orbs = compute_hf(mol, obasis, na, nb, kin, natt, er, olp)
    (one_mo,), (two_mo,) = transform_integrals(kin + natt, er, 'tensordot', orbs[0])
    cienergy, cicoeffs, civec = compute_FCI(obasis.nbasis, core_energy, one_mo, two_mo,
                                            na, nb)
    wfn = pyci.FullCIWavefunction(obasis.nbasis, na, nb)
    two_mo = two_mo.reshape(two_mo.shape[0]**2, two_mo.shape[1]**2)
    ciham = pyci.Hamiltonian(core_energy, one_mo, two_mo)
    solver = pyci.SparseSolver(wfn, ciham)
    solver()
    cienergy0 = solver.eigenvalues()[0]
    cicoeffs0 = solver.eigenvectors().flatten()
    print cienergy
    assert abs(cienergy + 2.88759483109) < 1e-11
    assert cienergy0 == cienergy
    assert (abs(cicoeffs0 - cicoeffs) < 1e-10).all()
#test_FCI()
