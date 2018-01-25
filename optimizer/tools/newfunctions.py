"""Auxiliary functions for Range-Separation"""

import os
from horton import *
from horton.test.common import tmpdir
import numpy as np
import math
from nose.plugins.attrib import attr
import pyci
from wfns.wfn.ci.base import CIWavefunction
from wfns.ham.chemical import ChemicalHamiltonian
from wfns.solver.ci import brute
#from geminals.ci.fci import FCI
#from geminals.wrapper.horton import gaussian_fchk

__all__ = ['get_ci_info', 'get_ci_sd_pyci', 'get_dets_pyci',
           'get_dms_from_fanCI_simple', 'get_larger_ci_coeffs',
           'compute_ci_fanCI', 'compute_fci_fanCI',
           'compute_density_dmfci', 'dm_mine',
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


def get_larger_ci_coeffs(coeffs, limit=1e-6):
    """
    Find and count the CI expansion coefficients with
    values above certain limit.

    Returns:
    ccount: int
        Number of coefficients greater than limit value.
    location: np darray (ccount,)
        Mask array with the location of the desired coefficients.
    """

    location = np.where(abs(coeffs)>limit)
    ccount = len(location)
    return ccount, location


def get_ci_sd_pyci(nbasis, alphas, betas):
    """
    Get Slater determinants and coefficients from PyCI
    
    In PyCI Determinants are store as list of integers of occupied
    orbitals, alphas and betas separately.

    FanCI uses bit strings converted to integers.

    Returns
    -------
    sd_vec
        List with Slater Determinants.
    """
    sd_vec = []
    # Loop over civectors and make strings
    for i in range(len(alphas)):
        a = list('0' * nbasis)
        b = list('0' * nbasis)
        for j in range(len(alphas)):
            if j in alphas[i]:
                a[j] = '1'
            if j in betas[i]:
                b[j] = '1'
        a = a[::-1]; a = ''.join(a)
        b = b[::-1]; b = ''.join(b)
        vec = '0b%s%s' % (b, a)
        vec = int(vec,2)
        sd_vec.append(vec)
    return sd_vec


def get_dets_pyci(wfn):
    alphas = []
    betas = []
    for idx, det in wfn.iter_dets():
        a, b = det
        alphas.append(list(a))
        betas.append(list(b))
    return alphas, betas


def compute_fci_fanCI(nelec, nbasis, one, two, nuc_nuc):
    """
    Compute FCI energy
    
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

    Returns:
    FCI Ground State Energy, Groud State CI coefficients
    """
    nspin = nbasis*2
    fci = FCI(nelec, nspin)
    ham = ChemicalHamiltonian(one_int, two_int, orbtype='restricted', energy_nuc_nuc=nuc_nuc)

    # optimize
    energies, coeffs = brute(fci, ham)
    return energies[0] + nuc_nuc, coeffs[:,0]


def compute_ci_fanCI(nelec, nbasis, sd_vec, one, two, nuc_nuc):
    """
    Compute CI energy
    
    Arguments
    ---------
    nelec: int
        Total number of electrons.
    nbasis: int
        Number of spatial basis functions.
    sd_vec: list of int
        List with integer of the (bit string) determinants.
    one: ndarray (nbasis, nbasis)
        One electron integrals in MO basis.
    two: ndarray (nbasis, nbasis, nbasis, nbasis)
        Two electron integrals in MO basis.

    Returns:
    CI Ground State Energy, Groud State CI coefficients
    """
    nspin = nbasis*2
    ci = CIWavefunction(nelec, nspin, sd_vec=sd_vec)
    ham = ChemicalHamiltonian(one, two, orbtype='restricted', energy_nuc_nuc=nuc_nuc)

    # optimize
    energies, coeffs = brute(ci, ham)
    return energies[0] + nuc_nuc, coeffs[:,0]


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


def get_dms_from_fanCI_simple(h1, h2, nuc, hf_energy, nelec, civec, coeffs):
    sd_coeffs_ground = np.array(coeffs)
    fci = FCI(nelec=nelec, H=h1, G=h2, nuc_nuc=nuc, spin=0, civec=civec)
    fci.sd_coeffs  = sd_coeffs_ground
    dm1, dm2 = fci.get_density_matrix(val_threshold=0, notation='physicist')
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
