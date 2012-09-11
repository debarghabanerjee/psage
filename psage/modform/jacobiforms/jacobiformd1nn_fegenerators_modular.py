r"""
We provide methods to create Fourier expansions of (weak) Jacobi forms `\mathrm{mod} p`.

TODO: Unify this code with the implementation for expansions of ZZ.
"""

#===============================================================================
# 
# Copyright (C) 2010 Martin Raum
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
#===============================================================================

from sage.misc.all import prod
from sage.rings.all import PowerSeriesRing, GF
from sage.rings.all import binomial, factorial
from sage.structure.sage_object import SageObject
import operator

#===============================================================================
# JacobiFormD1NNModularFactory_class
#===============================================================================

class JacobiFormD1NNModularFactory_class (JacobiFormD1NNFactory_class) :
    
    def __init__(self, precision, p) :
        JacobiFormD1NNFactory_class.__init__(self, precision)
        
        self.__power_series_ring_modular = PowerSeriesRing(GF(p), 'q')
        self.__p = int(p)

    def power_series_ring_modular(self) :
        return self.__power_series_ring_modular
    
    def _set_theta_factors(self, theta_factors) :
        r"""
        Set the cache for theta factors. See _theta_factors.
        
        INPUT:
        
        - ``theta_factors`` -- A list of power series over `GF(p)`. 
        """
        if theta_factos not in self.power_series_ring_modular() :
            theta_factors = self.power_series_ring()(theta_factors)
        
        self.__theta_factors = theta_factors
    
    def _theta_factors(self) :
        r"""
        The vector `W^\# (\theta_0, .., \theta_{2m - 1})^{\mathrm{T}} \pmod{p}` as a list.
        The `q`-expansion is shifted by `-(m + 1)(2*m + 1) / 24`, which will be compensated
        for by the eta factor.
        """
        try :
            return self.__theta_factors
        except AttributeError :
            self.__theta_factors = self.power_series_ring_modular()(
                                     JacobiFormD1NNFactory_class._theta_factors(self, self.__p) )
        
            return self.__theta_factors

    def _set_eta_factor(self, eta_factor) :
        r"""
        Set the cache for theta factors. See _theta_factors.
        
        INPUT:
        
        - ``eta_factor`` -- A power series over `GF(p)`.
        """
        if not eta_factor in self.power_series_ring_modular() :
            eta_factor = self.power_series_ring_modular()(eta_factor)
        
        self.__eta_factor = eta_factor
    
    def _eta_factor(self) :
        r"""
        The inverse determinant of `W`, which in the considered cases is always a negative
        power of the eta function. See the thetis of Nils Skoruppa.
        """
        try :
            return self.__eta_factor
        except AttributeError :
            self.__eta_factor = self.power_series_ring_modular()( 
                                  JacobiFormD1NNFactory_class._eta_factors(self) )
        
            return self.__eta_factor 
 
    def by_taylor_expansion(self, fs, k) :
        r"""
        We combine the theta decomposition and the heat operator as in the
        thesis of Nils Skoruppa. This yields a bijections of the space of weak
        Jacobi forms of weight `k` and index `m` with the product of spaces
        of elliptic modular forms `M_k \times S_{k+2} \times .. \times S_{k+2m}`.

        INPUT:
        
        - ``fs`` -- A list of functions that given an integer `p` return the
                    q-expansion of a modular form with rational coefficients
                    up to precision `p`.  These modular forms correspond to
                    the components of the above product.
        
        - `k` -- An integer. The weight of the weak Jacobi form to be computed.
        
        NOTE:

            In order to make ``phi_divs`` integral we introduce an extra factor
            `2^{\mathrm{index}} * \mathrm{factorial}(k + 2 * \mathrm{index} - 1)`.
        """
        ## we introduce an abbreviations
        p = self.__p
        PS = self.power_series_ring()
            
        if not len(fs) == self.__precision.jacobi_index() + 1 :
            raise ValueError( "fs (which has length {0}) must be a list of {1} Fourier expansions" \
                              .format(len(fs), self.__precision.jacobi_index() + 1) )

        qexp_prec = self._qexp_precision()
        if qexp_prec is None : # there are no forms below the precision
            return dict()
        
        f_divs = dict()
        for (i, f) in enumerate(fs) :
            f_divs[(i, 0)] = PS(f(qexp_prec), qexp_prec)
                
        for i in xrange(self.__precision.jacobi_index() + 1) :
            for j in xrange(1, self.__precision.jacobi_index() - i + 1) :
                f_divs[(i,j)] = f_divs[(i, j - 1)].derivative().shift(1)
            
        phi_divs = list()
        for i in xrange(self.__precision.jacobi_index() + 1) :
            ## This is the formula in Skoruppas thesis. He uses d/ d tau instead of d / dz which yields
            ## a factor 4 m
            phi_divs.append( sum( f_divs[(j, i - j)] * (4 * self.__precision.jacobi_index())**i
                                  * binomial(i,j) * ( 2**self.index() // 2**i)
                                  * prod(2*(i - l) + 1 for l in xrange(1, i))
                                  * (factorial(k + 2*self.index() - 1) // factorial(i + k + j - 1))
                                  * factorial(2*self.__precision.jacobi_index() + k - 1)
                                  for j in xrange(i + 1) ) )
            
        phi_coeffs = dict()
        for r in xrange(self.index() + 1) :
            series = sum( map(operator.mul, self._theta_factors()[r], phi_divs) )
            series = self._eta_factor() * series

            for n in xrange(qexp_prec) :
                phi_coeffs[(n, r)] = int(series[n].lift()) % p

        return phi_coeffs
