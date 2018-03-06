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

"""Variational optimizer of long-range erfgau potential"""

import numpy as np
from numpy.linalg import eigh

import pyci
# TODO replace this by explicit import
from horton import *
from horton.meanfield.orbitals import Orbitals
from horton.meanfield.hamiltonian import REffHam
from horton.meanfield.observable import RExchangeTerm, RDirectTerm
from wfns.ham.density import density_matrix
from lrtools.ciwrapper import compute_ci_fanCI, compute_FCI, compute_my_dm
from lrtools.ciwrapper import get_larger_ci_coeffs
from lrtools.slsqp import *
from lrtools.common import *
from optimizer.integrals import compute_sr_potential, compute_energy_from_potential


__all__ = ['ErfGauOptimizer', 'FullErfOptimizer',
           'FullErfGauOptimizer', 'TruncErfGauOptimizer']


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
    def __init__(self, nbasis, core_energy, modintegrals, refintegrals, na, nb, ncore=0, mu=0.0):
        """
        Arguments:
        -----------
        nbasis: int
            Number of spatial orbitals
        core_energy: float
            Nuclear-repulsion energy
        modintegrals: IntegralsWrapper
            One- and two-electron intregrals of the model to be optimized
        refintegrals: list, np.ndarray
            HF one- and two-electron integrals
        na/nb
            Number of alpha/beta electrons
        ncore
            Number of core orbitals, assumes are occupied by two electrons
        mu: float
            Range-separation parameter
        """
        self.nbasis = nbasis
        self.core_energy = core_energy
        self.modintegrals = modintegrals
        self.refintegrals = refintegrals
        self.na = na
        self.nb = nb
        self.ncore = ncore
        self.mu = mu
        self.mu_energy = 0.0
        self.naturals = False

    def set_mu(self, mu):
        """ Give some value for parameter mu"""
        self.mu = mu

    def set_olp(self, olp):
        """Give a value for basis overlap"""
        self.olp = olp

    def set_dm_alpha(self, dm):
        self.dm_alpha = dm

    def set_orb_alpha(self, orb):
        self.orb_alpha = orb

    def get_naturalorbs(self, dm1):
        """
        Get the natural orbitals from the 1-DM
        """
        dm1 /= 2.0
        norb = Orbitals(self.nbasis)
        # Diagonalize and compute eigenvalues
        evals, evecs = eigh(dm1)
        # Reorder the matrix and the eigenvalues
        evecs1 = evecs.copy()
        evecs1[:] = evecs[:,::-1]
        norb.occupations[::-1] = evals
        # Get the natural orbitals
        norb.coeffs[:] = np.dot(self.orb_alpha.coeffs, evecs1)
        norb.energies[:] = 0.0
        return norb

    def compute_energy(self, pars):
        """Function passed to Scipy to compute energy"""
        raise NotImplementedError("Called base class, something is wrong.")

    def optimize_energy(self, pars, optype='standard'):
        """Wrapper for SLQS optimizer from Scipy
        
        Arguments:
        -----------
        pars: np.ndarrya
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
        if len(pars) != 2:
            raise ValueError("For this method two variables are required: c and alpha")
        if optype not in ['standard', 'diff']:
            raise ValueError("The type of minimization is incorrect, only\
                              'standard' and 'diff' options are valid")
        self.optype = optype
        fn = self.compute_energy

        result = fmin_slsqp(fn, pars, full_output=True)
        fmin = result[0]
        emin = result[1]
        return result


class FullErfOptimizer(ErfGauOptimizer):
    """
    Variational optimizer for the erfgau potential from FCI expansion
    """
    def __init__(self, nbasis, core_energy, modintegrals, refintegrals, na, nb,
                 ncore=0, mu=0.0):
        """
        Arguments:
        -----------
        nbasis: int
            Number of spatial orbitals
        core_energy: float
            Nuclear-repulsion energy
        modintegrals: IntegralsWrapper
            One- and two-electron intregrals of the model to be optimized
        refintegrals: list, np.ndarray
            HF one- and two-electron integrals
        na/nb: int
            Number of alpha/beta electrons
        ncore: int
            Number of core orbitals, assumes are occupied by two electrons
        mu: float
            Range-separation parameter
        """
        ErfGauOptimizer.__init__(self, nbasis, core_energy, modintegrals, refintegrals,
                                 na, nb, ncore, mu)

    def compute_energy(self, pars):
        """Function for Scipy to compute the energy"""
        # Compute Long-Range integrals
        mu = pars[0]
        self.set_mu(mu)
        # Update the integrals
        pars_ints = [[mu]]
        self.modintegrals.update(pars_ints)

        # Use PyCI to compute FCI energy
        cienergy, cicoeffs, civec = compute_FCI(self.nbasis, self.core_energy,
                                                self.modintegrals.one, self.modintegrals.two,
                                                self.na, self.nb)
        (dm1,), (dm2,) = density_matrix(cicoeffs, civec, self.nbasis)
        if self.naturals:
            norb = self.get_naturalorbs(dm1)
            
            # Transform integrals to NO basis
            (one_no,), (two_no,) = transform_integrals(self.modintegrals.one_ao_ref,
                                                       self.modintegrals.two_ao, 'tensordot',
                                                       norb)
            (one_no_full,), (two_no_full,) = transform_integrals(self.modintegrals.one_ao_ref,
                                                                 self.modintegrals.two_ao_ref,
                                                                 'tensordot', norb)
            # Use PyCI to compute FCI energy
            cienergy, cicoeffs, civec = compute_FCI(self.nbasis, self.core_energy,
                                                    one_no, two_no,
                                                    self.na, self.nb)
            (dm1,), (dm2,) = density_matrix(cicoeffs, civec, self.nbasis)
            self.mu_energy = np.einsum('ij,ij', one_no, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', two_no, dm2) + self.core_energy
            energy_exp = np.einsum('ij,ij', one_no_full, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', two_no_full, dm2)

        else:
            energy_exp = np.einsum('ij, ij', self.refintegrals[0], dm1)\
                + 0.5*np.einsum('ijkl, ijkl', self.refintegrals[1], dm2)
            self.mu_energy = np.einsum('ij,ij', self.modintegrals.one, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', self.modintegrals.two, dm2)\
                        + self.core_energy

        energy_exp += self.core_energy
        if self.optype == 'standard':
            return energy_exp
        else:
            return self.mu_energy - energy_exp

class FullErfGauOptimizer(ErfGauOptimizer):
    """
    Variational optimizer for the erfgau potential from FCI expansion
    """
    def __init__(self, nbasis, core_energy, modintegrals, refintegrals, na, nb,
                 ncore=0, mu=0.0):
        """
        Arguments:
        -----------
        nbasis: int
            Number of spatial orbitals
        core_energy: float
            Nuclear-repulsion energy
        modintegrals: IntegralsWrapper
            One- and two-electron intregrals of the model to be optimized
        refintegrals: list, np.ndarray
            HF one- and two-electron integrals
        na/nb: int
            Number of alpha/beta electrons
        ncore: int
            Number of core orbitals, assumes are occupied by two electrons
        mu: float
            Range-separation parameter
        """
        ErfGauOptimizer.__init__(self, nbasis, core_energy, modintegrals, refintegrals,
                                 na, nb, ncore, mu)

    def compute_energy(self, pars):
        """Function for Scipy to compute the energy"""
        # Compute Long-Range integrals
        c, alpha = pars
        c = c * self.mu
        alpha = (alpha * self.mu)**2
        # Update the integrals
        pars_ints = [[self.mu, c, alpha]]
        self.modintegrals.update(pars_ints)

        # Use PyCI to compute FCI energy
        cienergy, cicoeffs, civec = compute_FCI(self.nbasis, self.core_energy,
                                                self.modintegrals.one, self.modintegrals.two,
                                                self.na, self.nb)
        (dm1,), (dm2,) = density_matrix(cicoeffs, civec, self.nbasis)
        if self.naturals:
            norb = self.get_naturalorbs(dm1)
            # Transform integrals to NO basis
            (one_no,), (two_no,) = transform_integrals(self.modintegrals.one_ao_ref,
                                                       self.modintegrals.two_ao, 'tensordot',
                                                       norb)
            (one_no_full,), (two_no_full,) = transform_integrals(self.modintegrals.one_ao_ref,
                                                                 self.modintegrals.two_ao_ref,
                                                                 'tensordot', norb)
            # Use PyCI to compute FCI energy
            cienergy, cicoeffs, civec = compute_FCI(self.nbasis, self.core_energy,
                                                    one_no, two_no,
                                                    self.na, self.nb)
            (dm1,), (dm2,) = density_matrix(cicoeffs, civec, self.nbasis)
            energy_exp = np.einsum('ij,ij', one_no_full, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', two_no_full, dm2)
            self.mu_energy = np.einsum('ij,ij', one_no, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', two_no, dm2)\
                        + self.core_energy

        else:
            energy_exp = np.einsum('ij, ij', self.refintegrals[0], dm1)\
                + 0.5*np.einsum('ijkl, ijkl', self.refintegrals[1], dm2)
            self.mu_energy = np.einsum('ij,ij', self.modintegrals.one, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', self.modintegrals.two, dm2)\
                        + self.core_energy

        energy_exp += self.core_energy
        if self.modintegrals.one_approx[0] == 'sr-x':
            er_sr = self.modintegrals.two_ao_ref.copy()
            er_sr -= self.modintegrals.two_ao
            poth = compute_sr_potential(self.nbasis, er_sr, [self.dm_alpha],
                                        'hartree')
            potx = compute_sr_potential(self.nbasis, er_sr, [self.dm_alpha],
                                        'exchange')
            dm_final = compute_my_dm(dm1*0.5, self.orb_alpha, self.nbasis)
            vx_sr = compute_energy_from_potential(potx, [dm_final])
            ex_sr = compute_energy_from_potential(potx, [self.dm_alpha])
            vh_sr = 0.25*compute_energy_from_potential(poth, [dm_final])
            sr_energy = - vx_sr - vh_sr + ex_sr
            self.mu_energy += sr_energy

        if self.optype == 'standard':
            return energy_exp
        else:
            return self.mu_energy - energy_exp


class TruncErfGauOptimizer(ErfGauOptimizer):
    """
    Variational optimizer for the erfgau potential from truncated CI expansion
    count_cicoeffs
    """
    def __init__(self, nbasis, core_energy, modintegrals, refintegrals, na, nb,
                 ncore=0, mu=0.0, ndet=None):
        """
        Arguments:
        -----------
        nbasis: int
            Number of spatial orbitals
        core_energy: float
            Nuclear-repulsion energy
        modintegrals: IntegralsWrapper
            One- and two-electron intregrals of the model to be optimized
        refintegrals: list, np.ndarray
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
        ErfGauOptimizer.__init__(self, nbasis, core_energy, modintegrals, refintegrals,
                                 na, nb, ncore, mu)
        self.ndet = ndet


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
        while (abs(tmpcoeffs - cicoeffs)).all() > eps:
            tmpcoeffs[:] = cicoeffs.copy()
            # Use FanCI to compute CI
            cienergy, cicoeffs, civec = compute_ci_fanCI(nelec, self.nbasis, self.modintegrals.one,
                                                         self.modintegrals.two,
                                                         self.core_energy, civec)
            
            (dm1,), (dm2,) = density_matrix(cicoeffs, civec, self.nbasis)
            norb = self.get_naturalorbs(dm1)
            
            # Transform integrals to NO basis
            (one_no,), (two_no,) = transform_integrals(self.modintegrals.one_ao_ref,
                                                       self.modintegrals.two_ao, 'tensordot',
                                                       norb)
            (one_no_full,), (two_no_full,) = transform_integrals(self.modintegrals.one_ao_ref,
                                                                 self.modintegrals.two_ao_ref,
                                                                 'tensordot', norb)
            # Assign values to integrals
            self.refintegrals[0][:] = one_no_full
            self.refintegrals[1][:] = two_no_full
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
                                                     self.modintegrals.two, self.core_energy,
                                                     civec=None, full=False)
        
        # Check the first lmax larger coefficients
        coeffcopy = np.absolute(cicoeffs).flatten()
        coeffcopy = np.sort(coeffcopy)[::-1]
        limit = coeffcopy[self.ndet]

        # Check coefficients larger than certain value
        ccount, loc = get_larger_ci_coeffs(cicoeffs, limit)
        new_civec = np.array(civec)[loc]
        new_cicoeffs =cicoeffs[loc]

        # Converge natural orbitals for CI expansion
        dm1, dm2 = self.converge_naturalorbs(new_civec, new_cicoeffs)

        # Compute energy
        self.mu_energy = np.einsum('ij,ij', self.modintegrals.one, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', self.modintegrals.two, dm2) 
        self.mu_energy += self.core_energy
        energy_exp = np.einsum('ij,ij', self.refintegrals[0], dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', self.refintegrals[1], dm2)
        energy_exp += self.core_energy
        if self.optype == 'standard':
            return energy_exp
        else:
            return self.mu_energy - energy_exp
        return cienergy
