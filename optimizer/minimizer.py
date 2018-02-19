"""Variational optimizer of long-range erfgau potential"""

import numpy as np

import pyci
# TODO replace this by explicit import
from horton import *
from ci.cispace import FullCISpace
from wfns.ham.density import density_matrix
from optimizer.tools.ciwrapper import compute_ci_fanCI, compute_FCI
from optimizer.tools.slsqp import *
from optimizer.tools.functions import *


__all__ = ['ErfGauOptimizer', 'FullErfGauOptimizer', 'TruncErfGauOptimizer']


class ErfGauOptimizer(object):
    """
    Class to optimize the variables of the Gaussian function
    of the erfgau model potential for range-separated methods.
    

    Implementation can be easily extended to Unrestricted wavefunctions

    Methods
    =======
    compute_energy
    optimize_energy
    
    """
    def __init__(self, core_energy, modintegrals, refintegrals, na, nb, ncore=0, mu=0.0):
        """
        Arguments:
        -----------
        core_energy: float
            Nuclear-repulsion energy
        modintegrals: IntegralsWrapper
            One- and two-electron intregrals of the model to be optimized
        refintegrals: IntegralsWrapper
            HF one- and two-electron integrals
        na/nb
            Number of alpha/beta electrons
        ncore
            Number of core orbitals, assumes are occupied by two electrons
        mu: float
            Range-separation parameter
        """
        self.core_energy = core_energy
        self.modintegrals = modintegrals
        self.refintegrals = refintegrals
        self.na = na
        self.nb = nb
        self.ncore = ncore
        self.mu = mu
        self.mu_energy = 0.0

    def set_mu(self, mu):
        """ Give some value for parameter mu"""
        self.mu = mu

    def get_naturalorbs(self, dm1):
        """
        Get the natural orbitals from the 1-DM
        """
        # Get natural orbitals
        norb = Orbitals(self.nbasis)
        noccs, ncoeffs = eigh(dm1)
        norb.coeffs[:] = ncoeffs[:,:self.nbasis]
        norb.occupations[:] = noccs
        norb.energies[:] = 0.0
        return norb

    def compute_energy(self, pars):
        """Function passed to Scipy to compute energy"""
        raise NotImplementedError("Called base class, something is wrong.")

    def optimize_energy(self, var_vec, optype='standard'):
        """Wrapper for SLQS optimizer from Scipy
        
        Arguments:
        -----------
        var_vector: np.ndarrya
            Variables to be optimized.
            -cinit, float
                Initial guess for c parameter.
            -alphainit, float
                Initial guess for alpha parameter.
        optype: str
            Type of optimization. Options are:
            'standard': minime the expectation value of physical Hamiltonian
            'diff' :  minimize the difference between the model Energy and the
                      expectation value of H (minimize first order perturbation)
        *Returns*
        result: full scipy output
            result[0] : optimal parameters
            result[1] : Value of f_min (E_min)
        """
        if len(args) > 1:
            raise ValueError("For this method only one parameter is required")
        if len(var_vec) != 2:
            raise ValueError("For this method two variables are required: c and alpha")
        if optype not in ['standard', 'diff']:
            raise ValueError("The type of minimization is incorrect, only\
                              'standard' and 'diff' options are valid")
        self.optype = optype
        cinit, alphainit = var_vec
        pars = np.array([cinit, alphainit])
        fn = self.compute_energy

        result = fmin_slsqp(fn, pars, full_output=True)
        fmin = result[0]
        emin = result[1]
        return result


class FullErfGauOptimizer(ErfGauOptimizer):
    """
    Variational optimizer for the erfgau potential from FCI expansion
    """
    def __init__(self, core_energy, modintegrals, refintegrals, na, nb, ncore=0, mu=0.0):
        """
        Arguments:
        -----------
        core_energy: float
            Nuclear-repulsion energy
        modintegrals: IntegralsWrapper
            One- and two-electron intregrals of the model to be optimized
        refintegrals: IntegralsWrapper
            HF one- and two-electron integrals
        na/nb: int
            Number of alpha/beta electrons
        ncore: int
            Number of core orbitals, assumes are occupied by two electrons
        mu: float
            Range-separation parameter
        """
        ErfGauOptimizer.__init__(core_energy, modintegrals, refintegrals, na, nb, ncore, mu)

    def compute_energy(self, pars):
        """Function for Scipy to compute the energy"""
        # Compute Long-Range integrals
        c, alpha = pars
        c *= self.mu
        alpha *= self.mu; alpha *= alpha
        # Update the integrals
        pars_ints = [[self.mu, c, alpha]]
        self.modintegrals.update(pars_ints)

        # Use PyCI to compute FCI energy
        cienergy, cicoeffs, civec = compute_FCI(self.nbasis, self.core_energy,
                                                self.modintegrals.one, self.modintegrals.two,
                                                self.na, self.nb)
        (dm1,), (dm2,) = density_matrix(coeffs, new_civec, self.nbasis)
        norb = self.get_naturalorbs(dm1)

        # Transform integrals to NO basis
        (one_no,), (two_no,) = transform_integrals(self.modintegrals.one,
                                                   self.modintegrals.two, 'tensordot',
                                                   norb_alpha)
        (one_no_full,), (two_no_full,) = transform_integrals(self.refintegrals.one,
                                                             self.refintegrals.two,
                                                             'tensordot', norb_alpha)

        self.mu_energy = np.einsum('ij,ij', one_no, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', two_no, dm2) + self.core_energy
        energy_exp = np.einsum('ij,ij', one_no_full, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', two_no_full, dm2)
        energy_exp += self.core_energy
        if self.optype == 'standard':
            return energy_exp
        else:
            return self.mu_energy - energy_exp


class TruncErfGauOptimizer(ErfGauOptimizer):
    """
    Variational optimizer for the erfgau potential from truncated CI expansion
    count_cicoeffs
    """
    def __init__(self, core_energy, modintegrals, refintegrals, na, nb, ncore=0, mu=0.0, ndet):
        """
        Arguments:
        -----------
        core_energy: float
            Nuclear-repulsion energy
        modintegrals: IntegralsWrapper
            One- and two-electron intregrals of the model to be optimized
        refintegrals: IntegralsWrapper
            HF one- and two-electron integrals
        na/nb
            Number of alpha/beta electrons
        ncore
            Number of core orbitals, assumes are occupied by two electrons
        mu: float
            Range-separation parameter
        ndet: int
            Number of determinants to be included in the CI expansion
        """
        ErfGauOptimizer.__init__(core_energy, modintegrals, refintegrals, na, nb, ncore, mu)


    def converge_naturalorbs(self, civec, cicoeffs, eps=1e-4):
        """
        Converge natural orbitals

        Arguments:
        ----------
        civec: np.ndarray((nbasis,))
            Slater determinants' occupations
        cicoeffs: np.ndarray((nbasis, nbasis))
            Coefficients of the CI expansion
        eps: float
            Threshold to determine convergence

        *Returns* the 1-DM and 2-DM
        """
        nelec = self.na + self.nb
        tmpcoeffs = np.zeros(cicoeffs.shape)
        # Loop until convergence
        while (abs(tmpcoeffs - cicoeffs)).all() < eps:
            tmpcoeffs[:] = cicoeffs.copy()
            # Use FanCI to compute CI
            cienergy, cicoeffs, civec = compute_ci_fanCI(nelec, self.nbasis, self.modintegrals.one,
                                                         self.modintegrals.two, core_energy, civec)
            
            (dm1,), (dm2,) = density_matrix(coeffs, new_civec, self.nbasis)
            norb = self.get_naturalorbs_low(dm1)
            
            # Transform integrals to NO basis
            (one_no,), (two_no,) = transform_integrals(self.modintegrals.one,
                                                       self.modintegrals.two, 'tensordot',
                                                       norb_alpha)
            (one_no_full,), (two_no_full,) = transform_integrals(self.refintegrals.one,
                                                                 self.refintegrals.two,
                                                                 'tensordot', norb_alpha)
            # Assign values to integrals
            self.refintegrals.assign(one_no_full, two_no_full)
            self.modintegrals.assign(one_no, two_no)
        return dm1, dm2


    def compute_energy(self, pars):
        """
        Compute FCI with the model Hamiltonian

        Arguments:
        ----------
        pars: np.ndarray
            Array with parameters for the model potential

        **Returns**
            cienergy: CI Energy for that model Hamiltonian
            cicoeffs: The FCI solution for the wavefunction
            civec: Slater determinants' occupations
        """
        # Compute Long-Range integrals
        c, alpha = pars
        c *= self.mu
        alpha *= self.mu; alpha *= alpha
        # Update the integrals
        pars_ints = [[self.mu, c, alpha]]
        self.modintegrals.update(pars_ints)

        # Use CISD as guess
        # Use FanCI to compute CISD energy
        nelec = self.na + self.nb
        cienergy, cicoeffs, civec = compute_ci_fanCI(nelec, self.nbasis, self.modintegrals.one,
                                                     self.modintegrals.two, core_energy,
                                                     civec=None, full=False)
        
        # Check the first lmax larger coefficients
        coeffcopy = np.absolute(cicoeffs).flatten()
        coeffcopy = np.sort(coeffcopy)[::-1]
        limit = coeffcopy[self.detmax]

        # Check coefficients larger than certain value
        ccount, loc = get_larger_ci_coeffs(result[0][1], limit)
        new_civec = np.array(civec)[loc]
        new_cicoeffs =cicoeffs[loc]

        # Converge natural orbitals for CI expansion
        dm1, dm2 = self.converge_naturalorbs(new_civec, new_cicoeffs)

        # Compute energy
        self.mu_energy = np.einsum('ij,ij', self.modintegrals.one, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', self.modintegrals.two, dm2) 
        self.mu_energy += self.core_energy
        energy_exp = np.einsum('ij,ij', self.refintegrals.one, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', self.modintegrals.two, dm2)
        energy_exp += self.core_energy
        if self.optype == 'standard':
            return energy_exp
        else:
            return self.mu_energy - energy_exp
        return cienergy
