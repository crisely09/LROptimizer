""" Module to manage integrals from Horton"""

import numpy as np
from scipy.linalg import eigh

from horton.io.iodata import IOData
from horton.gbasis.gobasis import GOBasis
from horton.meanfield.orbitals import Orbitals
from horton.meanfield.indextransform import transform_integrals


class IntegralsWrapper(object):
    """
    Wrapper to compute, update and transform one and two-electron integrals
    
    """

    def __init__(self, mol, obasis, one_approx, two_approx, pars, orbs):
        """
        one_approx: str
            Approximation for the one-electron Hamiltonian.
            Defaul set to 'standard' for the kinetic and nuclear attraction integrals
            Standard 
            Other options are:
            - 'sr-x' To add the short-range contribution of HF exact exchange 
        two_approx: list, str
            - 'sr-xdoci' To add the short-range contribution of HF exact exchange 
            only for double excitations
            requires density matrices
            - 'erf' to use the error function (Long Range) operator
              requires 1 parameter: mu, the range-separation parameter.
            - 'gauss' to use a Gaussian function as operator
              requires 2 parameters: c and alpha, the coefficient and
              exponent of the Gaussian function
            - 'erfgau' to use Erf + Gauss (Long Range) operator
              requires 3 parameters: mu, the range-separation parameter,
              c and alpha, the coefficient and exponent of the Gaussian function
        pars: list with shape (2,)
            Parameters for the integrals. [[parameters for one],[parameters for two]]
        orbs: Orbital
            HF/LDA orbital objects from HORTON
        """
        if not isinstance(mol, IOData):
            raise TypeError("mol must be a HORTON IOData object")
        if not isinstance(obasis, GOBasis):
             TypeError("obasis must be a HORTON GOBasis object")
        if not all([isinstance(approx, str) for approx in one_approx]):
            raise TypeError("one_approx should be a list of str for the approximations to\
                             be included in the one-electron integrals")
        if one_approx[0] not in ['standard', 'sr-x']:
            raise ValueError("The option given is not implemented, valid options are:\
                              'standard' and 'sr-x'.")
        if not all([isinstance(approx, str) for approx in two_approx]):
            print "approx", two_approx
            raise TypeError("two_approx should be a list of str for the approximations to\
                             be included in the two-electron integrals")
        if not all([approx in ['erf', 'erfgau', 'gauss', 'sr-xdoci'] for approx in two_approx]):
            raise ValueError("The approximation %d is not available, valid options are:\
                             'erf', 'gauss', 'erfgau', 'sr-xdoci'")
        if not all([isinstance(par, list) for par in pars]):
            raise TypeError("The parameters should be given in a list")
        if not all([isinstance(orb, Orbitals) for orb in orbs]):
            raise TypeError("Orbitals should be HORTON Orbital objects")
        if len(orbs) > 1:
            raise NotImplementedError("Only restricted cases implemented at the moment.")
        self.mol = mol
        self.obasis = obasis
        self.nbasis = obasis.nbasis
        self.one_approx = one_approx
        self.two_approx = two_approx
        self.orbs = orbs
        dms = []
        for orb in self.orbs:
            dms.append(orb.to_dm())
        self.dms = dms

        # Compute integrals
        one_ao = compute_standard_one_integrals(self.mol, self.obasis)
        self.one_ao_ref = one_ao
        self.two_ao_ref = self.obasis.compute_electron_repulsion()
        self.update(pars)

    def update(self, newpars):
        """
        Compute again the integrals for new parameters
        """
        self.pars = newpars
        one_ao = self.one_ao_ref.copy()
        two_ao = compute_two_integrals(self.obasis, self.two_approx, self.pars)
        if self.one_approx == 'sr-x':
            er_sr = self.two_ao_ref.copy()
            er_sr -= two_ao
            sr_pot = compute_sr_potential(self.nbasis, er_sr, dms, 'exchange')
            sr_pot += compute_sr_potential(self.nbasis, er_sr, dms, 'hartree')
            one_ao += sr_pot
        if 'sr-xdoci' in self.two_approx:
            use_full_exchange_twoe_doci(self.nbasis, self.two_ao_ref, two_ao)
        (one,), (two,) = transform_integrals(one_ao, two_ao, 'tensordot', self.orbs[0])
        self.one = one
        self.two = two

    def assign(self, one, two):
        """
        Assign values to the integrals
        """
        self.one[:] = one
        self.two[:] = two


def compute_standard_one_integrals(mol, obasis):
    """
    Compute the standard one and two electron integrals in the AO basis

    Arguments:
    ----------
    obasis: GOBases
        Horton object with all the information of the basis set
    orbs: list, Orbitals
        Horton object to work with orbitals' information (energies, occupations
        and coefficients)
    """
    if not isinstance(mol, IOData):
        raise TypeError("mol should be a IOData object from HORTON")
    if not isinstance(obasis, GOBasis):
        raise TypeError("obasis should be a GOBasis object from HORTON")
    kin = obasis.compute_kinetic()
    natt = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers)
    one = kin + natt
    return one


def compute_two_integrals(obasis, approxs, pars):
    """
    Compute special types of two-electron integrals

    obasis: GOBasis
        Horton object for basis functions
    approxs: list, str
        Approximation for the two-electron Hamiltonian
        Other options are:
        - 'erf' to use the error function (Long Range) operator
          requires 1 parameter: mu, the range-separation parameter.
        - 'gauss' to use a Gaussian function as operator
          requires 2 parameters: c and alpha, the coefficient and
          exponent of the Gaussian function
        - 'erfgau' to use Erf + Gauss (Long Range) operator
          requires 3 parameters: mu, the range-separation parameter,
          c and alpha, the coefficient and exponent of the Gaussian function
    pars: list
        List with parameters needed for the integrals
        for each term in approxs a list with parameters.
    """
    if not isinstance(obasis, GOBasis):
        raise TypeError("obasis should be a GOBasis object from HORTON")
    if not all([isinstance(approx, str) for approx in approxs]):
        raise TypeError("The approximations should be given as a string in a list")
    if not all([isinstance(par, list) for par in pars]):
        raise TypeError("The parameters should be given in a list")
    nbasis = obasis.nbasis
    two = np.zeros((nbasis, nbasis, nbasis, nbasis))
    for i,approx in enumerate(approxs):
        if approx == 'sr-xdoci':
            pass
        if approx == 'erf':
            print len(pars[i])
            assert len(pars[i]) == 1
            two += obasis.compute_erf_repulsion(pars[i][0])
        elif approx == 'gauss':
            assert len(pars[i]) == 2
            two += obasis.compute_gauss_repulsion(pars[i][0], pars[i][1])
        elif approx == 'erfgau':
            assert len(pars[i]) == 3
            two = obasis.compute_erf_repulsion(pars[i][0])
            gauss = obasis.compute_gauss_repulsion(pars[i][1], pars[i][2])
            two += gauss
        else:
            raise ValueError("The %s approximation for the two-electron integrals is not\
                              implemented in this version" % approx)
    return two


def compute_sr_potential(nbasis, er_sr, dms, whichpot):
    """
    Add Short-Range Hartree or Exchange Potential to one-electron integrals
    using Horton.

    Arguments:
    ----------
    nbasis: int
        Number of basis functions
    er_sr: np.ndarray((nbasis, nbasis, nbasis, nbasis))
        Short-Range two-electron integrals
    dms: list, np.ndarray((nbasis, nbasis))
        Density matrices
    whichpot: str
        Type of potential to be computed. Options are:
        'exchange' for the exchange potential
        'hartree' for the Hartree potential
    """
    if len(dms) == 1:
        # Restricted case
        if whichpot == 'hartree':
            ham = REffHam([RDirectTerm(er_sr, 'hartree')])
        else:
            ham = REffHam([RExchangeTerm(er_sr, 'x')])
        ham.reset(dms[0])
    elif len(dms) == 2:
        # Unrestricted case
        if whichpot == 'hartree':
            ham = UEffHam([UDirectTerm(er_sr, 'hartree')])
        else:
            ham = UEffHam([UExchangeTerm(er_sr, 'x')])
        ham.reset(dms)
    sr_potential = np.zeros((nbasis, nbasis))
    ham.compute_fock(sr_potential)
    return sr_potential


def use_full_exchange_two_doci(nbasis, two_full, two_lr):
    """
    Use the two electron integrals with full exchange HF
    for all elements when only two spatial orbitals
    are involved.

    Arguments:
    ----------
    nbasis: int
        Number of basis functions
    two_full: np.ndarray((nbasis, nbasis, nbasis, nbasis))
        Two-electron integrals of the full Coulomb interaction
    two_lr: np.ndarray((nbasis, nbasis, nbasis, nbasis))
        Two-electron integrals of the long-range interaction
    """
    if not isinstance(nbasis, int):
        raise TypeError("nbasis should be an integer")
    if not isinstance(two_full, np.ndarray):
        raise TypeError("two_full should be given as a np.ndarray")
    if not isinstance(two_lr, np.ndarray):
        raise TypeError("two_full should be given as a np.ndarray")
    if two_full.shape != two_full.shape:
        raise ValueError("The two arrays should have the same shape")
    if two_full.shape != ((nbasis, nbasis)):
        raise ValueError("The shape of the arrays doesn't match with the\
                          size of the basis nbasis")
    for i in range(nbasis):
        for j in range(nbasis):
            two_lr[i,j,i,j] = two_full[i,j,i,j]
            two_lr[i,j,j,i] = two_full[i,j,j,i]


def compute_energy_from_potential(potential, dms):
    """
    Compute energy of a given potential.
    (Expectation value of potential constructed as a Fock operator)

    Arguments:
    ----------
    potential: np.ndarray(nbasis, nbasis)
        Potential in Fock operator form
        **Attention** for the Hartree potential it should be divided
        by 2 before passing it here.
    dms: list, np.ndarray(nbasis, nbasis)
        Density matrices
    """
    if len(dms) == 1:
        energy = 2.0 * np.einsum('ab,ba', potential, dms[0])
    elif len(dms) ==2:
        energy = np.einsum('ab', potential, dms[0])
        energy += np.einsum('ab', potential, dms[1])
    else:
        raise ValueError("Only up to two density matrices are allowed (alpha and beta).")
    return energy
