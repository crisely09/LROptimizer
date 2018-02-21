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

"""Built-in short-range energy terms"""


import numpy as np

from horton.meanfield.gridgroup import GridObservable, DF_LEVEL_LDA
from horton.grid.molgrid import BeckeMolGrid
from horton.grid.poisson import solve_poisson_becke
from horton.utils import doc_inherit
from optimizer.tools.common import find_datafile


__all__ = ['RCustomGridObservable', 'RShortRangeAExchange', 
            'UShortRangeAExchange', 'ModifiedCorrelation', 
            'RShortRangeACorrelation', 'UShortRangeACorrelation', 
           ]


class RCustomGridObservable(GridObservable):
    '''Class used to load potentials in numpy arrays and convert them to
        Hamiltonian terms
    '''
    df_level = DF_LEVEL_LDA

    def __init__(self, pot, label):
        '''
           **Arguments:**

           pot
                A numpy array whit the potential to be used

           label
                A unique label for this contribution
        '''

        self.pot = pot
        GridObservable.__init__(self, label)

    @doc_inherit(GridObservable)
    def compute_energy(self, cache, grid):
        #print list(cache.iterkeys())
        rho = cache['rho_full']
        return grid.integrate(self.pot, rho)

    @doc_inherit(GridObservable)
    def add_pot(self, cache, grid, dpot_alpha):
        dpot_alpha += self.pot


class ModifiedExchange(GridObservable):
    """Common code for the Dirac Exchange Functional implementations."""

    df_level = DF_LEVEL_LDA

    def __init__(self, label='x_erfgauss', mu=0.0, c=1.0, alpha=1.0):
        r"""Initialize a ModifiedExchange instance.

        Parameters
        ----------
        label : str
            A label for this observable.

        mu: float
            The reange-separation parameter

        c: float
            The coefficient of the gaussian function of the modified
            potential.

        alpha: float
            The exponent of  the gaussian function fo the modified
            potential.
        """
        self.mu = mu
        self.c = c
        self.alpha = alpha
        self._coeff = 2.0 ** (1.0/3.0)
        GridObservable.__init__(self, label)

    def _update_pot(self, cache, grid, select):
        """Recompute an Exchange potential if invalid.

        Parameters
        ----------
        cache : Cache
            Storage for the potentials.
        grid : IntGrid
            A numerical integration grid.
        select : str
            'alpha' or 'beta'
        """
        rho = cache['all_%s' % select][:, 0]
        pot, new = cache.load('pot_x_dirac_%s' % select, alloc=grid.size)
        if new:
            pot = self._coeff * modified_exchange_potential(rho, self.mu, self.c, self.alpha, pot)
        return pot

    def _update_ex(self, cache, grid, select):
        """Recompute an Exchange energy per particle if invalid.

        Parameters
        ----------
        cache : Cache
            Storage for the potentials.
        grid : IntGrid
            A numerical integration grid.
        select : str
            'alpha' or 'beta'
        """
        rho = cache['all_%s' % select][:, 0]
        ex, new = cache.load('ex_x_dirac_%s' % select, alloc=grid.size)
        if new:
            ex[:] = self._coeff * modified_exchange_energy(rho, self.mu, self.c, self.alpha, ex)
        return ex


class RShortRangeAExchange(ModifiedExchange):
    """The Dirac Exchange Functional for restricted wavefunctions."""

    def compute_dirac_energy(self, cache, grid, select):
        pot = self.compute_dirac_potential(cache, grid, select)
        rho = cache['all_%s' % select][:, 0]
        ex =  (3.0 / 2.0) * grid.integrate(pot, rho)
        return ex


    def compute_dirac_potential(self, cache, grid, select):
        coeff = 3.0 / 4.0 * (3.0 / np.pi) ** (1.0 / 3.0)
        derived_coeff = -coeff * (4.0 / 3.0) * 2 ** (1.0 / 3.0)
        rho = cache['all_%s' % select][:, 0]
        pot = np.zeros(grid.size)
        pot[:] = derived_coeff * (rho) ** (1.0 / 3.0)
        return pot

    @doc_inherit(GridObservable)
    def compute_energy(self, cache, grid):
        ex = self._update_ex(cache, grid, 'alpha')
        ex_dirac = self.compute_dirac_energy(cache, grid, 'alpha')
        rho = cache['all_alpha'][:, 0]
        return ex_dirac - (2.0) * grid.integrate(ex, rho)

    @doc_inherit(GridObservable)
    def add_pot(self, cache, grid, pots_alpha):
        dirac_pot = self.compute_dirac_potential(cache, grid, 'alpha')
        pots_alpha[:, 0] += dirac_pot - self._update_pot(cache, grid, 'alpha')


class UShortRangeAExchange(ModifiedExchange):
    """The Dirac Exchange Functional for unrestricted wavefunctions."""

    def compute_dirac_energy(self, cache, grid, select):
        pot = self.compute_dirac_potential(cache, grid, select)
        rho = cache['all_%s' % select][:, 0]
        ex =  grid.integrate(pot, rho)
        return ex


    def compute_dirac_potential(self, cache, grid, select):
        coeff = 3.0 / 4.0 * (3.0 / np.pi) ** (1.0 / 3.0)
        derived_coeff = -coeff * (4.0 / 3.0) * 2 ** (1.0 / 3.0)
        rho = cache['all_%s' % select][:, 0]
        pot = np.zeros(grid.size)
        pot[:] = derived_coeff * (rho) ** (1.0 / 3.0)
        return pot

    @doc_inherit(GridObservable)
    def compute_energy(self, cache, grid):
        ex_alpha = self._update_ex(cache, grid, 'alpha')
        ex_beta = self._update_ex(cache, grid, 'beta')
        ex_alpha_dirac = self.compute_dirac_energy(cache, grid, 'alpha')
        ex_beta_dirac = self.compute_dirac_energy(cache, grid, 'beta')
        total_dirac = (3.0 / 4.0) * (ex_alpha_dirac + ex_beta_dirac)
        rho_alpha = cache['all_alpha'][:, 0]
        rho_beta = cache['all_beta'][:, 0]
        return total_dirac - (grid.integrate(ex_alpha, rho_alpha) +
                              grid.integrate(ex_beta, rho_beta))

    @doc_inherit(GridObservable)
    def add_pot(self, cache, grid, pots_alpha, pots_beta):
        alpha_dirac_pot = self.compute_dirac_potential(cache, grid, 'alpha')
        beta_dirac_pot = self.compute_dirac_potential(cache, grid, 'beta')
        pots_alpha[:, 0] += alpha_dirac_pot - self._update_pot(cache, grid, 'alpha')
        pots_beta[:, 0] += beta_dirac_pot - self._update_pot(cache, grid, 'beta')


class ModifiedCorrelation(GridObservable):
    """Common code for the Modified (Asymptotic) SR-Correlation Functional implementations."""

    df_level = DF_LEVEL_LDA

    def __init__(self, label='c_erfgauss', mu=0.0):
        r"""Initialize a ModifiedCorrelation instance.

        Parameters
        ----------
        label : str
            A label for this observable.

        mu: float
            The reange-separation parameter

        """
        self.mu = mu
        self.interpolant = None
        GridObservable.__init__(self, label)

    def compute_rsgrid(self, rho):
        rs = np.copy(rho)
        rs = np.power((3.0/(4.0*np.pi*rs)), 1.0/3.0)
        return rs

    def _update_pot(self, cache, grid, select):
        """Recompute the correlation potential if invalid.

        Parameters
        ----------
        cache : Cache
            Storage for the potentials.
        grid : IntGrid
            A numerical integration grid.
        select : str
            'alpha' or 'beta'
        """
        rho = cache['all_%s' % select][:, 0]
        pot, new = cache.load('pot_c_%s' % select, alloc=grid.size)
        if new:
            ec = self._update_ec(cache, grid, select)
            which = np.where(rho > 1e-7)
            rs = self.compute_rsgrid(rho[which])
            pot[which] = ec[which] - ec[which]*self.interpolant(rs, 1)/np.power((36*np.pi*rho[which]), 1.0/3.0)
        return pot

    def _update_ec(self, cache, grid, select):
        """Recompute the correlation energy per particle if invalid.

        Parameters
        ----------
        cache : Cache
            Storage for the potentials.
        grid : IntGrid
            A numerical integration grid.
        select : str
            'alpha' or 'beta'
        """
        rho = cache['all_%s' % select][:, 0]
        ec, new = cache.load('ec_c_%s' % select, alloc=grid.size)
        if new:
            if not self.interpolant:
                self.interpolant = compute_interpolant(self.mu)
            which = np.where(rho > 5e-7)
            rs = self.compute_rsgrid(rho[which])
            ectmp = self.interpolant(rs)
            weird = np.where(ectmp > 0.0)
            if weird[0]:
                log("============= WARNING!! Rs TOO LARGE ==============")
            ec[which] = compute_exp(ectmp)
        return ec


class RShortRangeACorrelation(ModifiedCorrelation):
    """The Modified (Asymptotic) Correlation Functional for restricted wavefunctions."""

    @doc_inherit(GridObservable)
    def compute_energy(self, cache, grid):
        ec = self._update_ec(cache, grid, 'alpha')
        rho = cache['all_alpha'][:, 0]
        return (2.0) * grid.integrate(ec, rho)

    @doc_inherit(GridObservable)
    def add_pot(self, cache, grid, pots_alpha):
        pots_alpha[:, 0] += self._update_pot(cache, grid, 'alpha')


class UShortRangeACorrelation(ModifiedCorrelation):
    """The Modified (Asymptotic) Correlation Functional for unrestricted wavefunctions."""

    @doc_inherit(GridObservable)
    def compute_energy(self, cache, grid):
        ec_alpha = self._update_ec(cache, grid, 'alpha')
        ec_beta = self._update_ec(cache, grid, 'beta')
        rho_alpha = cache['all_alpha'][:, 0]
        rho_beta = cache['all_beta'][:, 0]
        return (grid.integrate(ec_alpha, rho_alpha) +
                              grid.integrate(ec_beta, rho_beta))

    @doc_inherit(GridObservable)
    def add_pot(self, cache, grid, pots_alpha, pots_beta):
        pots_alpha[:, 0] += self._update_pot(cache, grid, 'alpha')
        pots_beta[:, 0] += self._update_pot(cache, grid, 'beta')


def check_energy_array(earray, emin=1e-8):
    new = np.copy(earray)
    for i, e in enumerate(earray):
        if 0.0 + e > -1e-9:
            new[i] = - emin
        elif e > 0:
            new[i] = - e
    return new

def compute_log(y):
    fx = np.copy(y)
    fx = np.log(-fx)
    return fx

def compute_exp(fx):
    e = np.copy(fx)
    fx = - np.exp(e)
    return fx

def compute_interpolant(mu):
    #load and contruct arrays for interpolation
    from scipy.interpolate import CubicSpline
    if abs(mu - 0.0) < 1e-10:
        coulombfn = find_datafile('test/coulomb_grids.npy')
        egrid = np.load(coulombfn)
        x = egrid[:,0]
        y = egrid[:,1]
    else:
        srcfn = find_datafile('test/sr_grids.npy')
        egrid = np.load(srcfn)
        mus = np.where(abs(egrid[:,1] - mu) < 1e-10)
        x = np.squeeze(egrid[mus,0])
        y = np.squeeze(egrid[mus,2])
    y = check_energy_array(y)
    fx = compute_log(y)
    interpolant = CubicSpline(x, fx, axis=0, bc_type="natural")
    return interpolant

def modified_exchange_energy(rho, mu, c, alpha, output):
    '''Compute exchange energy per particle of the asymptotic potential
        V(r) = erf(mu r)/r + c exp(-alpha^2 r^2)
    '''
    import math as m
    from scipy.special import erf as erf

    for i in range(len(rho)):
        if rho[i] < 1e-15:
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

def modified_exchange_potential(rho, mu, c, alpha, output):
    '''Compute exchange potential of the asymptotic potential
        V(r) = erf(mu r)/r + c exp(-alpha^2 r^2)
    '''
    import math as m
    from scipy.special import erf as erf

    for i in range(len(rho)):
        if rho[i] < 1e-15:
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


