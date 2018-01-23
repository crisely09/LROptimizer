"""Auxiliary functions for Range-Separation"""

import os
from horton import *
from horton.test.common import tmpdir
import numpy as np
from geminals.ci.fci import FCI
from geminals.wrapper.horton import gaussian_fchk
import math
from nose.plugins.attrib import attr
import pyci

__all__ = ['get_ci_info', 'compute_density_dmfci', 'dm_mine',
           'get_dm_from_fci', 'get_density_error']


def get_ci_info(fn):
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
    # get the civec
    civec = []
    coeffs = []
    for sd in info:
        # get the civec
        vec = '0b%s%s' % (sd[1],sd[2])
        #print "vec ", vec
        vec = int(vec,2)
        civec.append(vec)
        # get the coeffs
        cs = sd[3].split('\n')[0]
        cs = float(cs)
        coeffs.append(cs)
    return civec, coeffs

def get_ci_info_pyci(cispace, na, nb, nbasis):
    """Transform the occupations to bitstrings
    and then to integers to be used in olsens to get the DMS

    Arguments:
    ----------

    cispace
        Instance of FullCIWfn from PyCI. Contains all the information
        to index, address a vector's bitstring and occupations.

    na
        Integer. Number of alpha electrons (used for excitations)

    nb
        Integer. Number of beta electrons (used for excitations)

    nbasis
        Integer. Number of basis functions (used for excitations)

    **Returns**

    civec
        A list of integers that represent the occupations
        in each Slater determinant, ordered as 0bbetaalpha.
    """

    civec = []
    # Loop over civectors and make strings
    for i in range(nbasis):
        a = list('0' * nbasis)
        b = list('0' * nbasis)
        for j in range(nbasis):
            if j in cispace[i][0]:
                a[j] = '1'
            if j in cispace[i][1]:
                b[j] = '1'
        a = a[::-1]; a = ''.join(a)
        b = b[::-1]; b = ''.join(b)
        vec = '0b%s%s' % (a, b)
        vec = int(vec,2)
        civec.append(vec)
    return civec


def distance(point_grid1, point_grid2):
    """Compute the distance between two points"""
    distance = 0
    for i in range(3):
        distance += math.pow((point_grid1[i] - point_grid2[i]), 2)
    distance = math.pow(distance, 0.5)
    return distance

def compute_density_dmfci(gridpoints, npoints, dm, exp, obasis):
    """Compute density on a grid from the FCI 1DM (only one spin)"""
    density = np.zeros(npoints)
    iorbs = np.array(range(obasis.nbasis))
    orbitals = obasis.compute_grid_orbitals_exp(exp, gridpoints, iorbs)
    for i in range(obasis.nbasis):
        for j in range(obasis.nbasis):
            density += dm[i,j]*orbitals[:,i]*orbitals[:,j]
    return density

def dm_mine(dm, exp, nbasis):
    dmbar = np.zeros((nbasis, nbasis))
    for k in range(nbasis):
        for l in range(nbasis):
            for mu in range(nbasis):
                for v in range(nbasis):
                    dmbar[mu, v] += dm[k, l]*exp.coeffs[mu,k]*exp.coeffs[v,l] 
    return dmbar

def compute_sr_ldapot(density, exp_alpha, obasis, grid, mu, output):
    """Compute SR-LDA (alpha) potential matrix elements in AO basis from the density.
    
    **Arguments**

    density
        Density on a grid.
    exp_alpha
        DenseExpansion object.
    obasis
        GOBasis object.
    lf
        LinAlgFactory object.
    grid
        BeckeMolGrid.
    mu
        The range-separation parameter.
    output
        TwoIndex object for Short-Range LDA potential.

    """
    sr_lda_alpha = np.zeros((obasis.nbasis, obasis.nbasis))
    sr_lda_beta = np.zeros((obasis.nbasis, obasis.nbasis))
    rho_both = np.zeros((grid.size, 2))
    rho_both[:,0] = density
    rho_both[:,1] = density
    exc = np.zeros(grid.size)
    vxc_both = np.zeros((grid.size, 2))
    lsdsr_polarized_wrapper(mu, rho_both, exc, vxc_both)

    for i in xrange(obasis.nbasis):
        orb_alpha_i = obasis.compute_grid_orbitals_bfns(exp_alpha, grid.points, np.array([i]))
        for j in xrange(obasis.nbasis):
            orb_alpha_j = obasis.compute_grid_orbitals_bfns(exp_alpha, grid.points, np.array([j]))
            sr_lda_alpha[i, j] += grid.integrate(orb_alpha_i, orb_alpha_j, vxc_both[:,0])

    output.assign(sr_lda_alpha)
    return output

def compute_sr_ldaenergy(density, exp_alpha, obasis, grid, mu):
    """Compute SR-LDA (alpha) energy from the density.
    
    **Arguments**

    density
        Density on a grid.
    exp_alpha
        DenseExpansion object.
    obasis
        GOBasis object.
    lf
        LinAlgFactory object.
    grid
        BeckeMolGrid.
    mu
        The range-separation parameter.
    output
        TwoIndex object for Short-Range LDA potential.

    """
    sr_lda_exc = np.zeros((obasis.nbasis, obasis.nbasis))
    sr_lda_vxc = np.zeros((obasis.nbasis, obasis.nbasis))
    rho_both = np.zeros((grid.size, 2))
    rho_both[:,0] = density
    rho_both[:,1] = density
    exc = np.zeros(grid.size)
    vxc_both = np.zeros((grid.size, 2))
    lsdsr_polarized_wrapper(mu, rho_both, exc, vxc_both)

    energy_exc = 2.0*grid.integrate(density, exc)
    energy_vxc = 2.0*grid.integrate(density, vxc_both[:,0])
    return energy_exc, energy_vxc

def compute_sr_coulombpot(dmbar, operator, output):
    """Compute the Short-Range Coulomb matrix elements in AO basis
    
    **Arguments**

    dmbar
        The full density matrix (alpha+beta).

    operator
        The two-electron Coulomb integrals in AO basis.
    """
    operator.contract_two_to_two('abcd,bd->ac', dmbar, output)
    return output

def get_dm_from_fci_pyci(cispace, coeffs, nbasis, h1, h2, na, nb, hf_energy, nuc):
    """ Do FCI calculation, get energy and coefficients from PyCI.
    Then construct the DM using method from olsens.

    **Arguments**
    mol
        Molecule object with all the information of the system.
    """
    import gc
    civec = get_ci_info_pyci(cispace, na, nb, nbasis)

    nelec = na + nb
    sd_coeffs_ground = coeffs
    fci = FCI(nelec=nelec, H=h1, G=h2, nuc_nuc=nuc, spin=0, civec=civec)
    fci.sd_coeffs  = sd_coeffs_ground
    dm1, dm2 = fci.get_density_matrix(val_threshold=0, notation='physicist')
    del fci
    gc.collect()
    return dm1, dm2


def get_dm_from_fci(datname, h1, h2, nuc, hf_energy, nelec, mu):
    """Compute FCI and get the One-electron Density Matrix

    **Arguments**

    datname
        A string with the name/info about the system.
    mol
        Molecule object with all the information of the system.
    """

    import os
    import shutil
    import subprocess
    import re
    import sys

    name = 'flow'
    tofile = open(name, 'w')
    inname = '%s_%2.2f.psi4.dat' % (datname, mu)
    tofile.write(inname+"\n")
    tofile.write("fci\n")
    tofile.write("none\n")
    tofile.write("end\n")
    tofile.close()
    try:
        doci_en =""
        process = subprocess.Popen(["./ciflow.x" ] , stdin =open(name , 'r'), stdout = subprocess.PIPE)
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            doci_en += line
    except subprocess.CalledProcessError as e:
        print("Problem")

    fname = '%s_%2.2f.psi4outputfci.dat' % (datname, mu)
    civec, coeffs = get_ci_info(fname)
    sd_coeffs_ground = np.array(coeffs)
    fci = FCI(nelec=nelec, H=h1, G=h2, nuc_nuc=nuc, spin=0, civec=civec)
    fci.sd_coeffs  = sd_coeffs_ground
    dm1, dm2 = fci.get_density_matrix(val_threshold=0, notation='physicist')
    return dm1, dm2
    

def get_density_error(density1, density2):
    """The Root-Mean-Square error between two densities on a grid"""
    return np.sqrt(np.sum((np.power(density1 - density2,2))))

def check_density_graph(obasis, dmfci, dm, exp):
    """Check densities in a graph"""

    # Make a line
    line = np.array([[0.,0.,0.+i] for i in np.arange(0., 4., 0.01)])*angstrom
    ds =  np.array([i for i in np.arange(0., 4., 0.01)])*angstrom

    # Compute densities in the line
    rho1 = obasis.compute_grid_density_dm(dm, line)
    rho2 = compute_density_dmfci(line, len(line), dmfci, exp, obasis)
    

    import matplotlib.pyplot as plt

    plt.clf()
    plt.plot(ds, rho1)
    plt.plot(ds, rho2)
    plt.xlim(0.,4)
    plt.ylim(0,2)
    plt.savefig('rho_line.png')
 #  ###
    plt.clf()
    plt.semilogy(ds, rho1)
    plt.semilogy(ds, rho2)
    plt.xlim(0.,4)
    plt.ylim(10E-7,10)
    plt.savefig('rho_line_log.png')


def compute_sr_xldapot(density, exp_alpha, obasis, grid, mu, c, alpha, output):
    """Compute SR-LDA modified (alpha) potential matrix elements in AO basis from the density.
    
    **Arguments**

    density
        Density on a grid.
    exp_alpha
        DenseExpansion object.
    obasis
        GOBasis object.
    lf
        LinAlgFactory object.
    grid
        BeckeMolGrid.
    mu
        The range-separation parameter.
    c
        The coefficient of the Gaussian.
    alpha
        The exponent of the Gaussian.
    output
        TwoIndex object for Short-Range LDA potential.

    """
    sr_lda_alpha = np.zeros((obasis.nbasis, obasis.nbasis))
    sr_vxpot = compute_dirac_potential(density, grid)
    sr_vxpot -= (2.0 ** (1.0/3.0)) * modified_exchange_potential(density, mu, c, alpha)

    for i in xrange(obasis.nbasis):
        orb_alpha_i = obasis.compute_grid_orbitals_bfns(exp_alpha, grid.points, np.array([i]))
        for j in xrange(obasis.nbasis):
            orb_alpha_j = obasis.compute_grid_orbitals_bfns(exp_alpha, grid.points, np.array([j]))
            sr_lda_alpha[i, j] += grid.integrate(orb_alpha_i, orb_alpha_j, sr_vxpot)

    output.assign(sr_lda_alpha)
    return output


def modified_exchange_energy(rho, mu, c, alpha, output):
    '''Compute exchange energy per particle of the asymptotic potential
        V(r) = erf(mu r)/r + c exp(-alpha^2 r^2)
    '''
    import math as m
    from scipy.special import erf as erf

    output = np.zeros(rho.size)
    for i in range(len(rho)):
        if rho[i] < 1e-10 or mu < 1e-10:
            output[i] = 0.
        else:
            rho_inv = 1.0 / rho[i]
            var1 = np.power(3.0 * rho[i] * m.pi**2.0, 1.0/3.0)
            var2 = var1 * var1
            mu_sqr = mu * mu
            alpha_sqr = alpha * alpha
            # terms that depend on rho^-1/3
            ex1 = (alpha * c * 3.0 ** (2.0 / 3.0))/(2.0 * m.pi ** (7.0 / 6.0))
            ex1 -= (alpha * c * m.exp(-var2 / (alpha_sqr)))/(2.0 * (3.0 ** (1.0 / 3.0)) * m.pi ** (7.0 / 6.0))
            ex1 += (mu_sqr * 3.0 ** (2.0 / 3.0))/(2 * m.pi ** (5.0 / 3.0))
            ex1 -= (mu_sqr * m.exp(-var2/(mu_sqr)))/((m.pi ** (5.0 / 3.0)) * 3.0 ** (1.0/3.0))
            ex1 *= rho[i] ** (-1.0 / 3.0)
            # terms that depend on rho^-1
            ex2 = -(alpha * alpha_sqr * c) / (3 * m.pi ** (5.0 / 2.0))
            ex2 += (alpha * alpha_sqr * c * m.exp(-var2/alpha_sqr))/(3 * m.pi ** (5.0 / 2.0))
            ex2 -= (mu_sqr * mu_sqr)/ (6.0 * m.pi ** 3.0)
            ex2 += (m.exp(- var2 / mu_sqr) * mu_sqr * mu_sqr) / (6 * m.pi ** 3.0)
            ex2 *= rho_inv
            # terms rho independent
            #ex1 *= rho[i] ** (-1.0 / 3.0)
            ex3 = - 0.5 * c * erf(var1/alpha) - (mu * erf(var1/mu)) / (m.pi ** (1.0 / 2.0))
            output[i] = ex1 + ex2 + ex3
    return output


def modified_exchange_potential(rho, mu, c, alpha):
    '''Compute exchange potential of the asymptotic potential
        V(r) = erf(mu r)/r + c exp(-alpha^2 r^2)
    '''
    import math as m
    from scipy.special import erf as erf

    output = np.zeros(rho.size)
    for i in range(len(rho)):
        if rho[i] < 1e-10 or mu < 1e-10 :
            output[i] += 0.
        else:
            rho_inv = 1.0 / rho[i]
            var1 = np.power(3.0 * rho[i] * m.pi**2.0, 1.0/3.0)
            var2 = var1 * var1
            mu_sqr = mu * mu
            alpha_sqr = alpha * alpha
            # terms that depend on rho^-4/3
            pot1 = (alpha * c)/(m.pi ** (7.0 / 6.0))
            pot1 -= (alpha * c *m.exp(-var2 / (alpha_sqr)))/(m.pi ** (7.0 / 6.0))
            pot1 += (mu_sqr)/(m.pi ** (5.0 / 3.0))
            pot1 -= (mu_sqr * m.exp(-var2/mu_sqr))/(m.pi ** (5.0 / 3.0))
            pot1 *= (3 * rho[i]) ** (-1.0/3.0)
            # terms rho independent
            pot2 = - 0.5 * c * erf(var1/alpha) - (mu * erf(var1/mu)) / (m.pi ** (1.0 / 2.0))
            output[i] += pot1 + pot2
    return output

def compute_dirac_potential(rho, grid):
    coeff = 3.0 / 4.0 * (3.0 / np.pi) ** (1.0 / 3.0)
    derived_coeff = -coeff * (4.0 / 3.0) * 2 ** (1.0 / 3.0)
    pot = np.zeros(grid.size)
    pot[:] = derived_coeff * (rho) ** (1.0 / 3.0)
    return pot

