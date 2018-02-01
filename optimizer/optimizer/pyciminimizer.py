"""Variational optimizer of long-range erfgau potential"""

import numpy as np

import pyci
# TODO replace this by explicit import
from horton import *
from ci.cispace import FullCISpace
from wfns.ham.density import *
from optimizer.tools.slsqp import *
from optimizer.tools.functions import *


__all__ = ['VariationalOptimizer',]


class VariationalOptimizer(object):
    """
    Class to optimize the variables of the Gaussian function
    of the erfgau model potential for range-separated methods.
    

    Implementation can be easily extended to Unrestricted wavefunctions

    Methods
    =======
    compute_fci_energy
    solve_rhf
    solve_rlda
    compute_energy_lr
    compute_energy_withx
    optimize_energy
    
    """
    def __init__(self, mol, obasis, nelec, name, hamtype='lr', refmethod='LDA'):
        """
        Parameters:
        -----------
        mol: Molecule
            Information of the system.
        obasis: GOBasis
            Basis set object.
        nelec: integer
            Number of electrons in the system.
        name: string
            Name of molecule, to be used on files.
        hamtype: str
            The name of the approximation for the Hamiltonian. The posible
            options are:
            `lr` : Use LR potential and evaluate FCI.
            `withx`: Keep the exact-exchange fixed through the whole range.
        refmethod: str
            Name of the method from where we get the MO basis.
        """
        self.mol = mol
        self.obasis = obasis
        self.nbasis = obasis.nbasis
        self.nelec = nelec
        self.mu = 0.0
        self.mu_energy = 0.0
        self.name = name
        self.hamtype = hamtype
        # Compute integrals
        self.olp = obasis.compute_overlap()
        self.kin = obasis.compute_kinetic()
        self.na = obasis.compute_nuclear_attraction(mol.coordinates, mol.pseudo_numbers)
        self.er = obasis.compute_electron_repulsion()
        # Get and save orbitals and density from reference method
        if refmethod == 'HF':
            orb_alpha, dm_alpha = self.solve_rhf()
        elif refmethod == 'LDA':
            orb_alpha, dm_alpha = self.solve_rlda()
        else:
            raise NotImplementedError('Only HF and LDA are available now.')
        self.orb_alpha = orb_alpha
        self.dm_alpha = dm_alpha
        # Transform integrals
        one = self.kin.copy()
        one += self.na
        two = self.er
        (one_mo,), (two_mo,) = transform_integrals(one, two, 'tensordot', orb_alpha)
        self.one = one
        self.one_mo = one_mo
        self.two_mo = two_mo
        #self.check_fci_olsens()

    def get_fci_energy(self, ncore=0):
        """Little check for the energy"""
        # Using PyCI to compute FCI energy
        ciham = pyci.FullCIHam(self.core_energy, self.one_mo, self.two_mo)
        na = self.nelec/2
        nb = na
        wfn = pyci.FullCIWfn(ciham.nbasis, na, nb, ncore=0)

        # Compute the energy
        op = ciham.sparse_operator(wfn)
        result = pyci.cisolve(op)
        return result[0][0]


    def set_mu(self, mu):
        """ Give some value for parameter mu"""
        self.mu = mu


    def solve_rhf(self):
        '''Solve KS-HF equations'''
        # Create alpha orbitals
        orb_alpha = Orbitals(self.nbasis)
        
        # Initial guess
        guess_core_hamiltonian(self.olp, self.kin + self.na, orb_alpha)
        
        # Construct the restricted HF effective Hamiltonian
        external = {'nn': compute_nucnuc(self.mol.coordinates, self.mol.pseudo_numbers)}
        terms = [
            RTwoIndexTerm(self.kin, 'kin'),
            RDirectTerm(self.er, 'hartree'),
            RTwoIndexTerm(self.na, 'ne'),
            RExchangeTerm(self.er, 'x'),
        ]
        ham = REffHam(terms, external)
        
        # Decide how to occupy the orbitals (5 alpha electrons)
        occ_model = AufbauOccModel(self.nelec/2)
        
        # Converge WFN with Optimal damping algorithm (ODA) SCF
        # - Construct the initial density matrix (needed for ODA).
        occ_model.assign(orb_alpha)
        dm_alpha = orb_alpha.to_dm()
        # - SCF solver
        scf_solver = EDIIS2SCFSolver(1e-6)
        scf_solver(ham, self.olp, occ_model, dm_alpha)
        ham.reset(dm_alpha)
        self.mol.total_energy = ham.compute_energy()
        fock_alpha = np.zeros((self.nbasis, self.nbasis))
        ham.compute_fock(fock_alpha)
        orb_alpha.from_fock_and_dm(fock_alpha, dm_alpha, self.olp)
        self.core_energy = external['nn']
        self.hf_energy = ham.cache['energy']
        return orb_alpha, dm_alpha


    def solve_rlda(self):
        '''Solve KS-LDA equations'''
        
        # Create alpha orbitals
        orb_alpha = Orbitals(self.nbasis)
        
        # Initial guess
        guess_core_hamiltonian(self.olp, self.kin + self.na, orb_alpha)
        
        # Create grid for LDA parts
        self.grid = BeckeMolGrid(self.mol.coordinates, self.mol.numbers, self.mol.pseudo_numbers)
        # Construct the restricted HF effective Hamiltonian
        external = {'nn': compute_nucnuc(self.mol.coordinates, self.mol.pseudo_numbers)}
        terms = [
            RTwoIndexTerm(self.kin, 'kin'),
            RDirectTerm(self.er, 'hartree'),
            RTwoIndexTerm(self.na, 'ne'),
            RGridGroup(self.obasis, self.grid, [
            RLibXCLDA('x'),
            RShortRangeACorrelation(mu=0.0)
        ]),
        ]
        ham = REffHam(terms, external)
        
        # Decide how to occupy the orbitals (5 alpha electrons)
        occ_model = AufbauOccModel(self.nelec/2)
        
        # Converge WFN with Optimal damping algorithm (ODA) SCF
        # - Construct the initial density matrix (needed for ODA).
        occ_model.assign(orb_alpha)
        dm_alpha = orb_alpha.to_dm()
        # - SCF solver
        scf_solver = EDIIS2SCFSolver(1e-6)
        scf_solver(ham, self.olp, occ_model, dm_alpha)
        ham.reset(dm_alpha)
        self.mol.total_energy = ham.compute_energy()
        fock_alpha = np.zeros((self.nbasis, self.nbasis))
        ham.compute_fock(fock_alpha)
        self.hf_energy = ham.compute_energy()
        orb_alpha.from_fock_and_dm(fock_alpha, dm_alpha, self.olp)
        self.core_energy = external['nn']
        return orb_alpha, dm_alpha


    def compute_energy_lr(self, pars):
        """Function for Scipy to compute the energy"""
        # Compute Long-Range integrals
        c, alpha = pars
        c *= self.mu
        alpha *= self.mu
        alpha *= alpha
        erf = self.obasis.compute_erf_repulsion(self.mu)
        gauss = self.obasis.compute_gauss_repulsion(c, alpha)
        erf += gauss
        # Transform integrals
        (one_mo,), (two_mo,) = transform_integrals(self.one, erf, 'tensordot', self.orb_alpha)

        # Using PyCI to compute FCI energy
        ciham = pyci.FullCIHam(self.core_energy, self.one_mo, self.two_mo)
        na = self.nelec/2
        nb = na
        wfn = pyci.FullCIWfn(ciham.nbasis, na, nb)

        # Compute the energy
        op = ciham.sparse_operator(wfn)
        result = pyci.cisolve(op)
        energy = result[0][0]
        coeffs = result[0][1]
        cispace = FullCISpace(ciham.nbasis, na, nb, excitations=None)
        dm1, dm2 = pyci.fullci_density_matrices(wfn, result)
        self.mu_energy = np.einsum('ij,ij', one_mo, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', two_mo, dm2)
        energy_real = np.einsum('ij,ij', self.one_mo, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', self.two_mo, dm2)
        print("ENERGY(mu) = ", self.mu_energy)
        print("=============================== ENERGY = ", energy_real)
        return float(energy_real)


    def compare_energy_lr(self, pars):
        """Function for Scipy to compute the energy"""
        # Compute Long-Range integrals
        c, alpha = pars
        c *= self.mu
        alpha *= self.mu
        alpha *= alpha
        erf = self.obasis.compute_erf_repulsion(self.mu)
        gauss = self.obasis.compute_gauss_repulsion(c, alpha)
        erf += gauss
        # Transform integrals
        (one_mo,), (two_mo,) = transform_integrals(self.one, erf, 'tensordot', self.orb_alpha)

        # Using CIFlow to compute FCI and Olsens for DMs
        # Write files
        molout = IOData(core_energy=self.core_energy, nelec=2,
                        energy=self.hf_energy, obasis=self.obasis,
                        one_mo=one_mo, two_mo=two_mo, dm_alpha=self.dm_alpha)
        # useful for exchange with other codes
        datname = '%s_avdz' % self.name
        molout.to_file('%s_%2.2f.psi4.dat' % (datname, self.mu))
        dm1, dm2 = get_dm_from_fci(datname, one_mo, two_mo, self.core_energy,
                                    self.hf_energy, 2, self.mu)
        self.mu_energy = np.einsum('ij,ij', one_mo, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', two_mo, dm2) + self.core_energy
        energy_real = np.einsum('ij,ij', self.one_mo, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', self.two_mo, dm2)
        energy_real += self.core_energy
        print("ENERGY(mu) = ", self.mu_energy)
        print("=============================== ENERGY = ", energy_real)

        # Using PyCI to compute FCI energy
        ciham = pyci.FullCIHam(self.core_energy, one_mo, two_mo)
        na = self.nelec/2
        nb = na
        wfn = pyci.FullCIWfn(ciham.nbasis, na, nb)
        alphas, betas = get_dets_pyci(wfn)
        civec = get_ci_sd_pyci(self.nbasis, alphas, betas)

        # Compute the energy
        op = ciham.sparse_operator(wfn)
        result = pyci.cisolve(op)
        # The density matrices from pyci are no right!!
        #dm1_2, dm2_2 = pyci.fullci_density_matrices(wfn, result)
        (dm1_2,), (dm2_2,) = density_matrix(result[0][1], civec, self.nbasis)
        mu_energy2 = np.einsum('ij,ij', one_mo, dm1_2)\
                        + 0.5*np.einsum('ijkl, ijkl', two_mo, dm2_2)
        energy_real2 = np.einsum('ij,ij', self.one_mo, dm1_2)\
                        + 0.5*np.einsum('ijkl, ijkl', self.two_mo, dm2_2)
        print("ENERGY(mu) = ", mu_energy2)
        print("=============================== ENERGY 2= ", energy_real2)

        # Using method from previous version of olsens
        dm1_3, dm2_3 = get_dms_from_fanCI_simple(one_mo, two_mo, self.core_energy,
                                                 self.hf_energy, self.nelec, civec, result[0][1])
        mu_energy3 = np.einsum('ij,ij', one_mo, dm1_3)\
                        + 0.5*np.einsum('ijkl, ijkl', two_mo, dm2_3)
        energy_real3 = np.einsum('ij,ij', self.one_mo, dm1_3)\
                        + 0.5*np.einsum('ijkl, ijkl', self.two_mo, dm2_3)
        print("ENERGY(mu) = ", mu_energy3)
        print("=============================== ENERGY 2= ", energy_real3)


    def compute_energy_lr_muspace(self, pars):
        """Function for Scipy to compute the energy"""
        # Compute Long-Range integrals
        c, alpha = pars
        c *= self.mu
        alpha *= self.mu
        alpha *= alpha
        erf = self.obasis.compute_erf_repulsion(self.mu)
        gauss = self.obasis.compute_gauss_repulsion(c, alpha)
        erf += gauss
        # Transform integrals
        (one_mo,), (two_mo,) = transform_integrals(self.one, erf, 'tensordot', self.orb_alpha)
        # Using PyCI to compute FCI energy
        ciham = pyci.FullCIHam(self.core_energy, one_mo, two_mo)
        na = self.nelec/2
        nb = na
        wfn = pyci.FullCIWfn(ciham.nbasis, na, nb)
        alphas, betas = get_dets_pyci(wfn)
        civec = get_ci_sd_pyci(self.nbasis, alphas, betas)

        # Compute the energy
        op = ciham.sparse_operator(wfn)
        result = pyci.cisolve(op)
        # Check coefficients larger than certain value
        ccount, loc = get_larger_ci_coeffs(result[0][1], 1e-10)
        new_civec = np.array(civec)[loc]
        energy, coeffs = compute_ci_fanCI(self.nelec, self.nbasis, new_civec,
                                          one_mo, two_mo, self.core_energy)
        (dm1,), (dm2,) = density_matrix(coeffs, new_civec, self.nbasis)
        self.mu_energy = np.einsum('ij,ij', one_mo, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', two_mo, dm2) + self.core_energy
        energy_exp = np.einsum('ij,ij', self.one_mo, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', self.two_mo, dm2)
        energy_exp += self.core_energy
        print("ENERGY(mu) = ", self.mu_energy)
        print("=============================== ENERGY = ", energy_exp)
        return energy_exp


    def compute_energy_lrdoci(self, pars):
        """Function for Scipy to compute the energy"""
        # Compute Long-Range integrals
        c, alpha = pars
        c *= self.mu
        alpha *= self.mu
        alpha *= alpha
        erf = self.obasis.compute_erf_repulsion(self.mu)
        gauss = self.obasis.compute_gauss_repulsion(c, alpha)
        erf += gauss
        # Transform integrals
        (one_mo,), (two_mo,) = transform_integrals(self.one, erf, 'tensordot', self.orb_alpha)

        # Use whole interaction terms when only
        # two spatial orbitals
        for i in range(self.nbasis):
            for j in range(self.nbasis):
                two_mo[i,j,i,j] = self.two_mo[i,j,i,j]
                two_mo[i,j,j,i] = self.two_mo[i,j,j,i]

        # Using CIFlow to compute FCI and Olsens for DMs
        # Write files
        molout = IOData(core_energy=self.core_energy, nelec=self.nelec,
                        energy=self.hf_energy, obasis=self.obasis,
                        one_mo=one_mo, two_mo=two_mo, dm_alpha=self.dm_alpha)
        # useful for exchange with other codes
        datname = '%s_avdz' % self.name
        molout.to_file('%s_%2.2f.psi4.dat' % (datname, self.mu))
        dm1, dm2 = get_dm_from_fci(datname, one_mo, two_mo, self.core_energy,
                                    self.hf_energy, self.nelec, self.mu)
        self.mu_energy = np.einsum('ij,ij', one_mo, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', two_mo, dm2) + self.core_energy
        energy_real = np.einsum('ij,ij', self.one_mo, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', self.two_mo, dm2)
        energy_real += self.core_energy
        print("ENERGY(mu) = ", self.mu_energy)
        print("=============================== ENERGY = ", energy_real)
        return float(energy_real)


    def compute_energy_withx(self, pars):
        """Function for Scipy to compute the energy"""
        # Compute Long-Range integrals
        c, alpha = pars
        c *= self.mu
        alpha *= self.mu
        alpha *= alpha
        erf = self.obasis.compute_erf_repulsion(self.mu)
        gauss = self.obasis.compute_gauss_repulsion(c, alpha)
        erf += gauss
    def compute_energy_withx(self, pars):
        """Function for Scipy to compute the energy"""
        # Compute Long-Range integrals
        c, alpha = pars
        c *= self.mu
        alpha *= self.mu
        alpha *= alpha
        erf = self.obasis.compute_erf_repulsion(self.mu)
        gauss = self.obasis.compute_gauss_repulsion(c, alpha)
        erf += gauss

        er_sr = self.er.copy()
        er_sr -= erf
        # Compute the Short-Range Coulomb integrals
        # as Fock operator
        # Exchange potential
        hamsrx = REffHam([
                 RExchangeTerm(er_sr, 'x')])
        hamsrx.reset(self.dm_alpha)
        srx_potential = np.zeros((self.nbasis, self.nbasis))
        hamsrx.compute_fock(srx_potential)
        # Hartree potential
        # Replace all the K-sr for the whole interaction
        hamsrh = REffHam([
                 RDirectTerm(er_sr, 'hartree')])
        sr_potential = np.zeros((self.nbasis, self.nbasis))

        print("*******************Converging the density*************************")
        run = True
        maxcycle = 180
        count = 0
        converged = False
        dm_final = self.dm_alpha.copy()
        density1 = self.obasis.compute_grid_density_dm(self.dm_alpha, self.grid.points)
        while run:
            hamsrh.reset(dm_final)
            hamsrh.compute_fock(sr_potential)
            sr_potential = sr_potential + srx_potential
            one = self.one + sr_potential
            # Transform integrals
            (one_mo,), (two_mo,) = transform_integrals(one, erf, 'tensordot', self.orb_alpha)
            
            # Using PyCI to compute FCI energy
            ciham = pyci.FullCIHam(self.core_energy, self.one_mo, self.two_mo)
            na = self.nelec/2
            nb = na
            wfn = pyci.FullCIWfn(ciham.nbasis, na, nb)
            
            # Compute the energy
            op = ciham.sparse_operator(wfn)
            result = pyci.cisolve(op)
            energy = result[0][0]
            coeffs = result[0][1]
            cispace = FullCISpace(ciham.nbasis, na, nb, excitations=None)
            dm1, dm2 = pyci.fullci_density_matrices(wfn, result)
            dm_final = dm_mine(dm1, self.orb_alpha, self.nbasis)
            dm_final *= 0.5
            npoints = len(self.grid.points)
            density2 = compute_density_dmfci(self.grid.points, npoints, 0.5*dm1,
                                             self.orb_alpha, self.obasis)
            error =  get_density_error(density1, density2)
            print "Error in densities ", error
            density1 = density2 
            count += 1
            if error < 5e-4:
                run = False
                converged = True
            elif count == maxcycle:
                break
        if converged:
            # Compute final energy
            # SR-Coulomb (Hartree) energy
            hamsrh.reset(dm_final)
            hamsrh.compute_fock(sr_potential)
            sr_coulomb =  hamsrh.compute_energy()
            ex_sr =  hamsrx.compute_energy()
            sr_vx = 2.0 * np.einsum('ab,ba', srx_potential, dm_final)
            sr_vh = np.einsum('ab,ba', sr_potential, dm_final)
            sr_energy = - sr_vh - sr_vx + ex_sr
            lr_energy = np.einsum('ij,ij', one_mo, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', two_mo, dm2)
            self.mu_energy = lr_energy + sr_energy
            
            energy_real = np.einsum('ij,ij', self.one_mo, dm1)\
                          + 0.5*np.einsum('ijkl, ijkl', self.two_mo, dm2)
            print("ENERGY(mu) = ", self.mu_energy)
            print("=============================== ENERGY = ", energy_real)
            return float(energy_real)
        else:
            self.mu_energy = 0.0
            return 0.0


    def compute_energy_withxc(self, pars):
        """Function for Scipy to compute the energy"""
        # Compute Long-Range integrals
        c, alpha = pars
        c *= self.mu
        alpha *= self.mu
        alpha *= alpha
        erf = self.obasis.compute_erf_repulsion(self.mu)
        gauss = self.obasis.compute_gauss_repulsion(c, alpha)
        erf += gauss

        er_sr = self.er.copy()
        er_sr -= erf
        # Exchange potential
        hamsrx = REffHam([
                 RExchangeTerm(er_sr, 'x')])
        srx_potential = np.zeros((self.nbasis, self.nbasis))
        hamsrx.reset(self.dm_alpha)
        hamsrx.compute_fock(srx_potential)
        # Hartree potential
        hamsrh = REffHam([
                 RDirectTerm(er_sr, 'hartree')])
        sr_potential = np.zeros((self.nbasis, self.nbasis))
        # Correlation potential
        hammod = REffHam([
            #RGridGroup(self.obasis, self.grid, [RLibXCLDA('c_vwn'),])
            RGridGroup(self.obasis, self.grid, [RShortRangeACorrelation(mu=self.mu),])
        ])
        src_potential = np.zeros((self.nbasis, self.nbasis))

        # Converge the density
        print("*******************Converging the density*************************")
        run = True
        maxcycle = 180
        count = 0
        converged = False
        dm_final = self.dm_alpha.copy()
        density1 = self.obasis.compute_grid_density_dm(self.dm_alpha, self.grid.points)
        while run:
            hamsrh.reset(dm_final)
            hamsrh.compute_fock(sr_potential)
            hammod.reset(dm_final)
            hammod.compute_fock(src_potential)
            sr_potential = sr_potential + srx_potential + src_potential
            one = self.one + sr_potential
            # Transform integrals
            (one_mo,), (two_mo,) = transform_integrals(one, erf, 'tensordot', self.orb_alpha)
            
            # Using PyCI to compute FCI energy
            ciham = pyci.FullCIHam(self.core_energy, self.one_mo, self.two_mo)
            na = self.nelec/2
            nb = na
            wfn = pyci.FullCIWfn(ciham.nbasis, na, nb)
            
            # Compute the energy
            op = ciham.sparse_operator(wfn)
            result = pyci.cisolve(op)
            energy = result[0][0]
            coeffs = result[0][1]
            cispace = FullCISpace(ciham.nbasis, na, nb, excitations=None)
            dm1, dm2 = pyci.fullci_density_matrices(wfn, result)
            dm_final = dm_mine(dm1, self.orb_alpha, self.nbasis)
            dm_final *= 0.5
            npoints = len(self.grid.points)
            density2 = compute_density_dmfci(self.grid.points, npoints, 0.5*dm1,
                                             self.orb_alpha, self.obasis)
            error =  get_density_error(density1, density2)
            print "Error in densities ", error
            density1 = density2 
            count += 1
            if error < 5e-4:
                run = False
                converged = True
            elif count == maxcycle:
                break
        if converged:
            # Correlation
            sr_vc = 2.0 * np.einsum('ab,ab', src_potential, dm_final)
            ec_sr = hammod.compute_energy()
            # Exchange
            ex_sr =  hamsrx.compute_energy()
            sr_vx = 2.0 * np.einsum('ab,ba', srx_potential, dm_final)
            # Hartree
            sr_vh = np.einsum('ab,ba', sr_potential, dm_final)
            
            # SR Energy
            # Take off the potential from LDA density, integrated over
            # FCI^mu wavefunction
            sr_energy = - sr_vh - sr_vx - sr_vc + ex_sr + ec_sr
            # LR Energy
            lr_energy = np.einsum('ij,ij', one_mo, dm1)\
                        + 0.5*np.einsum('ijkl, ijkl', two_mo, dm2)
            self.mu_energy = lr_energy + sr_energy
            
            energy_real = np.einsum('ij,ij', self.one_mo, dm1)\
                          + 0.5*np.einsum('ijkl, ijkl', self.two_mo, dm2)
            print("ENERGY(mu) = ", self.mu_energy)
            print("=============================== ENERGY = ", energy_real)
            return float(energy_real)
        else:
            self.mu_energy = 0.0
            return 0.0


    def optimize_energy(self, mu, cinit=1.0, alphainit=1.0):
        """Wrapper for SLQS optimizer from Scipy
        
        Parameters:
        -----------
        mu, float
            Range-separation parameter.
        cinit, float
            Initial guess for c parameter.
        alphainit, float
            Iinitial guess for alpha parameter.
        """
        self.set_mu(mu)
        pars = np.array([cinit, alphainit])
        if self.hamtype == 'lr':
            fn = self.compute_energy_lr
        elif self.hamtype == 'withx':
            fn = self.compute_energy_withx
        elif self.hamtype == 'withxc':
            fn = self.compute_energy_withxc
        elif self.hamtype == 'lrmu':
            fn = self.compute_energy_lr_muspace

        result = fmin_slsqp(fn, pars, full_output=True)
        fmin = result[0]
        emin = result[1]
        return result

