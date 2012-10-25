r"""
We provide methods to create Fourier expansions of (weak) Jacobi forms.
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

from psage.modform.fourier_expansion_framework.monoidpowerseries.monoidpowerseries_lazyelement import \
                                        EquivariantMonoidPowerSeries_lazy
from psage.modform.jacobiforms.jacobiformd1nn_fourierexpansion import JacobiFormD1NNFourierExpansionModule,  \
                                        JacobiFormD1NNFilter, JacobiFormD1NNIndices, JacobiFormD1WeightCharacter
from sage.combinat.partition import number_of_partitions
from sage.libs.flint.fmpz_poly import Fmpz_poly  
from sage.matrix.constructor import matrix
from sage.misc.cachefunc import cached_function
from sage.misc.functional import isqrt
from sage.misc.misc import prod
from sage.modular.modform.constructor import ModularForms
from sage.modular.modform.element import ModularFormElement
from sage.modules.free_module_element import vector
from sage.rings import big_oh
from sage.rings.all import GF
from sage.rings.arith import binomial, factorial
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ
from sage.structure.sage_object import SageObject
import operator

#===============================================================================
# jacobi_forms_by_taylor_expansion
#===============================================================================

def jacobi_form_by_taylor_expansion(i, precision, weight) :
    r"""
    Lazy Fourier expansion of the i-th Jacobi form in a certain basis;
    see _jacobi_forms_by_taylor_expansion_coordinates.
    
    INPUT:
    
    - `i` -- A non-negative integer.

    - ``precision`` -- A filter for the Fourier indices of Jacobi forms.
    
    - ``weight`` -- An integer.
    
    OUTPUT:
    
    - A lazy element of the Fourier expansion module for Jacobi forms.
    
    TESTS:

    See also ``JacobiFormD1NNFactory_class._test__jacobi_corrected_taylor_expansions`` and
    ``JacobiFormD1NNFactory_class._test__jacobi_torsion_point``
    
    ::
    
        sage: from psage.modform.jacobiforms.jacobiformd1nn_fegenerators import *
        sage: JacobiFormD1NNFactory_class._test__jacobi_taylor_coefficients( jacobi_form_by_taylor_expansion(1, JacobiFormD1NNFilter(20, 1), 10), 10 )
        [O(q^20), q - 24*q^2 + 252*q^3 - 1472*q^4 + 4830*q^5 - 6048*q^6 - 16744*q^7 + 84480*q^8 - 113643*q^9 - 115920*q^10 + 534612*q^11 - 370944*q^12 - 577738*q^13 + 401856*q^14 + 1217160*q^15 + 987136*q^16 - 6905934*q^17 + 2727432*q^18 + 10661420*q^19 + O(q^20)]
        sage: JacobiFormD1NNFactory_class._test__jacobi_taylor_coefficients( jacobi_form_by_taylor_expansion(1, JacobiFormD1NNFilter(20, 2), 10), 10 )
        [O(q^20), q - 24*q^2 + 252*q^3 - 1472*q^4 + 4830*q^5 - 6048*q^6 - 16744*q^7 + 84480*q^8 - 113643*q^9 - 115920*q^10 + 534612*q^11 - 370944*q^12 - 577738*q^13 + 401856*q^14 + 1217160*q^15 + 987136*q^16 - 6905934*q^17 + 2727432*q^18 + 10661420*q^19 + O(q^20), q - 48*q^2 + 756*q^3 - 5888*q^4 + 24150*q^5 - 36288*q^6 - 117208*q^7 + 675840*q^8 - 1022787*q^9 - 1159200*q^10 + 5880732*q^11 - 4451328*q^12 - 7510594*q^13 + 5625984*q^14 + 18257400*q^15 + 15794176*q^16 - 117400878*q^17 + 49093776*q^18 + 202566980*q^19 + O(q^20)]
        sage: JacobiFormD1NNFactory_class._test__jacobi_taylor_coefficients( jacobi_form_by_taylor_expansion(0, JacobiFormD1NNFilter(20, 3), 9), 9 )
        [O(q^20), q - 24*q^2 + 252*q^3 - 1472*q^4 + 4830*q^5 - 6048*q^6 - 16744*q^7 + 84480*q^8 - 113643*q^9 - 115920*q^10 + 534612*q^11 - 370944*q^12 - 577738*q^13 + 401856*q^14 + 1217160*q^15 + 987136*q^16 - 6905934*q^17 + 2727432*q^18 + 10661420*q^19 + O(q^20)]
    """
    jacobi_index = precision.jacobi_index()
    
    return EquivariantMonoidPowerSeries_lazy( JacobiFormD1NNFourierExpansionModule(ZZ, jacobi_index),
                                              precision,
                                              DelayedFactory_JacobiFormD1NN_by_taylor_expansion( i, precision.index(), weight, jacobi_index ).getcoeff,
                                              [JacobiFormD1WeightCharacter(weight)] )

#===============================================================================
# DelayedFactory_JacobiFormD1NN_by_taylor_expansion
#===============================================================================

class DelayedFactory_JacobiFormD1NN_by_taylor_expansion :
    r"""
    Delayed computation of the Fourier coefficients of Jacobi forms.
    
    ALGORITHM:
    
    We first lift an echelon basis of elliptic modular forms to weak Jacobi forms;
    See _all_weak_jacobi_forms_by_taylor_expansion. We then use
    _jacobi_forms_by_taylor_expansion_coordinates to find linear combinations that
    correspond to Jacobi forms. 
    """
    def __init__(self, i, precision, weight, index) :
        r"""
        INPUT:
        
        - ``precision`` -- A non-negative integer that corresponds to a precision of
                           the q-expansion.

        - ``weight`` -- An integer.
    
        - ``index`` -- A non-negative integer.    
        """
        self.__i = i
        self.__precision = precision
        self.__weight = weight
        self.__index = index
        
        self.__series = None
        self.__ch = JacobiFormD1WeightCharacter(weight)
    
    def getcoeff(self, key, **kwds) :
        (ch, k) = key
        if ch != self.__ch :
            return ZZ.zero()
    
        if self.__series is None :
            self.__series = \
              sum( map( operator.mul,
                       _jacobi_forms_by_taylor_expansion_coordinates(self.__precision, self.__weight, self.__index)[self.__i],
                       _all_weak_jacobi_forms_by_taylor_expansion(self.__precision, self.__weight, self.__index) ) )

        try :
            return self.__series[key]
        except KeyError :
            return 0

#===============================================================================
# _jacobi_forms_by_taylor_expansion_coords
#===============================================================================

_jacobi_forms_by_taylor_expansion_coordinates_cache = dict()

def _jacobi_forms_by_taylor_expansion_coordinates(precision, weight, index) :
    r"""
    The coefficients of Jacobi forms with respect to a basis of weak
    Jacobi forms that is returned by _all_weak_jacobi_forms_by_taylor_expansion.
    
    INPUT:
    
    - ``precision`` -- A non-negative integer that corresponds to a precision of
                       the q-expansion.

    - ``weight`` -- An integer.
    
    - ``index`` -- A non-negative integer.
    
    TESTS::
    
        sage: from psage.modform.jacobiforms.jacobiformd1nn_fegenerators import _jacobi_forms_by_taylor_expansion_coordinates
        sage: _jacobi_forms_by_taylor_expansion_coordinates(10, 10, 1)
        [
        (1, 0, 0),
        (0, 0, 1)
        ]
        sage: _jacobi_forms_by_taylor_expansion_coordinates(10, 12, 1)
        [
        (1, 0, 0),
        (0, 1, 0)
        ]
    """
    global _jacobi_forms_by_taylor_expansion_coordinates_cache
    
    key = (index, weight)
    try :
        return _jacobi_forms_by_taylor_expansion_coordinates_cache[key]
    except KeyError :
        if precision < (index - 1) // 4 + 1 :
            precision = (index - 1) // 4 + 1
        
        
        weak_forms = _all_weak_jacobi_forms_by_taylor_expansion(precision, weight, index)
        weak_index_matrix = matrix(ZZ, [ [ f[(n,r)] for (n, r) in JacobiFormD1NNFilter(index + 1, index, weak_forms = True).iter_indefinite_forms()
                                                    if 4 * index * n - r**2 != 0 ]
                                         for f in weak_forms] )
        
        _jacobi_forms_by_taylor_expansion_coordinates_cache[key] = \
          weak_index_matrix.left_kernel().echelonized_basis()
          
        return _jacobi_forms_by_taylor_expansion_coordinates_cache[key]

#===============================================================================
# _all_weak_jacobi_forms_by_taylor_expansion
#===============================================================================

@cached_function
def _all_weak_jacobi_forms_by_taylor_expansion(precision, weight, index) :
    """
    INPUT:
    
    - ``precision`` -- A non-negative integer that corresponds to a precision of
                       the q-expansion.

    - ``weight`` -- An integer.
    
    - ``index`` -- A non-negative integer.    

    TESTS:
    
    We compute the Fourier expansion of a Jacobi form of weight `4` and index `2`.  This
    is denoted by ``d``.  Moreover, we span the space of all Jacobi forms of weight `8` and
    index `2`.  Multiplying the weight `4` by the Eisenstein series of weight `4` must
    yield an element of the weight `8` space.  Note that the multiplication is done using
    a polynomial ring, since no native multiplication for Jacobi forms is implemented.
    
    ::
    
        sage: from psage.modform.jacobiforms.jacobiformd1nn_fourierexpansion import *
        sage: from psage.modform.jacobiforms.jacobiformd1nn_fegenerators import _all_weak_jacobi_forms_by_taylor_expansion
        sage: from psage.modform.fourier_expansion_framework import *
        sage: q_precision = 20
        sage: index_filter = JacobiFormD1NNFilter(q_precision, 2)
        sage: jacobi_indices = JacobiFormD1NNIndices(2)
        sage: P.<q> = PolynomialRing(LaurentPolynomialRing(QQ, 'zeta')); zeta = P.base_ring().gen(0)
        sage: f = _all_weak_jacobi_forms_by_taylor_expansion(q_precision, 4, 2)[0]
        sage: f_poly = sum(f[k] * q**k[0] * zeta**k[1] for k in JacobiFormD1NNFilter(q_precision, 2, reduced = False))
        sage: jacobi_wt8 = ExpansionModule(_all_weak_jacobi_forms_by_taylor_expansion(q_precision, 8, 2))
        sage: E4_poly = ModularForms(1, 4).gen(0).qexp(q_precision).polynomial()
        sage: h_poly = E4_poly * f_poly
        sage: h = EquivariantMonoidPowerSeries( f.parent(), {JacobiFormD1WeightCharacter(8) : dict( ((n, r), c) for (n,lpoly) in h_poly.dict().iteritems() for ((r,), c) in lpoly.dict().iteritems() if (n, r) == jacobi_indices.reduce((n,r))[0] and n < q_precision )}, index_filter )
        sage: jacobi_wt8.coordinates(h, in_base_ring = False)
        [7/66, 0, 0, 4480]
        sage: hh = h - jacobi_wt8.0.fourier_expansion() * 7 / 66 - jacobi_wt8.3.fourier_expansion() * 4480
        sage: all( c == 0 for c in hh.coefficients().values() )
        True
    
    We test the Taylor coefficients of the weak Jacobi coefficients.
    See JacobiFormD1NNFactory_class._test__by_taylor_expansion for a similar test.
    
    ::
        
        sage: from psage.modform.jacobiforms.jacobiformd1nn_fegenerators import _test__all_weak_jacobi_forms_by_taylor_expansion
        sage: [ _test__all_weak_jacobi_forms_by_taylor_expansion(40, weight, jacobi_index) for (weight, jacobi_index) in [(10, 1), (12, 2), (9, 3)] ]
        [None, None, None]
        sage: [ _test__all_weak_jacobi_forms_by_taylor_expansion(40, weight, jacobi_index) for (weight, jacobi_index) in [(7, 5), (10, 10) ]          # long time
        [None, None]
    """
    factory = JacobiFormD1NNFactory(precision, index)
    
    return [ _weak_jacbi_form_by_taylor_expansion( precision, fs, weight, factory )
             for fs in _all_weak_taylor_coefficients(weight, index) ]

def _test__all_weak_jacobi_forms_by_taylor_expansion(q_precision, weight, jacobi_index) :
    r"""
    INPUT:
    
    - ``q_precision`` -- A non-negative integer that corresponds to a precision of
                         the q-expansion.
    
    - ``weight`` -- An integer.
    
    - ``index`` -- A non-negative integer.
    """
    jacobi_forms = _all_weak_jacobi_forms_by_taylor_expansion(q_precision, weight, jacobi_index)
    preimages = _all_weak_taylor_coefficients(weight, jacobi_index)
    assert all( proj == f
                for (phi, fs) in zip(jacobi_forms, preimages)
                for (proj, f) in zip(
                     JacobiFormD1NNFactory_class._test__jacobi_taylor_coefficients(phi, weight),
                     JacobiFormD1NNFactory_class._test__jacobi_predicted_taylor_coefficients(fs, q_precision) ) )
    
#===============================================================================
# _theta_decomposition_indices
#===============================================================================

@cached_function
def _all_weak_taylor_coefficients(weight, index) :
    r"""
    A product basis of the echelon bases of 
    
    - `M_k, M_{k + 2}, ..., M_{k + 2 m}` etc. if ``weight`` is even,
    
    - `M_{k + 1}, ..., M_{k + 2 m - 3}` if ``weight`` is odd.
    
    INPUT:
    
    - ``weight`` -- An integer.
    
    - ``index`` -- A non-negative integer.
    
    TESTS::
    
        sage: from psage.modform.jacobiforms.jacobiformd1nn_fegenerators import _all_weak_taylor_coefficients
        sage: _all_weak_taylor_coefficients(12, 1)
        [[<bound method ModularFormElement.qexp of 1 + 196560*q^2 + 16773120*q^3 + 398034000*q^4 + 4629381120*q^5 + O(q^6)>, <function <lambda> at ...>], [<bound method ModularFormElement.qexp of q - 24*q^2 + 252*q^3 - 1472*q^4 + 4830*q^5 + O(q^6)>, <function <lambda> at ...>], [<function <lambda> at ...>, <bound method ModularFormElement.qexp of 1 - 24*q - 196632*q^2 - 38263776*q^3 - 1610809368*q^4 - 29296875024*q^5 + O(q^6)>]]
    """
    R = PowerSeriesRing(ZZ, 'q'); q = R.gen()
    
    if weight % 2 == 0 :
        nmb_modular_forms = index + 1
        start_weight = weight
    else :
        nmb_modular_forms = index - 1
        start_weight = weight + 1
        
    modular_forms = list()
    for (i,k) in enumerate(range(start_weight, start_weight + 2 * nmb_modular_forms, 2)) :
        modular_forms += [ [lambda p: big_oh.O(q**p) for _ in range(i)] + [b.qexp] + [lambda p: big_oh.O(q**p) for _ in range(nmb_modular_forms - 1 - i)]
                           for b in ModularForms(1, k).echelon_basis() ]
        
    return modular_forms 

#===============================================================================
# _weak_jacbi_form_by_taylor_expansion
#===============================================================================

def _weak_jacbi_form_by_taylor_expansion(precision, fs, weight, factory = None) :
    r"""
    The lazy Fourier expansion of a Jacobi form of index ``len(fs) - 1``, whose first
    corrected Taylor coefficients are the ``fs``.
    
    INPUT:
    
    - ``precision`` -- A filter for the Fourier expansion of Jacobi forms of scalar index.
    
    - ``fs`` - A list of functions of one argument `p` that return a power series of
               precision `p`.  The coefficients must be integral.
    
    - ``weight`` -- An integer.
    
    - ``factory`` -- Either ``None``, or a factory class for Jacobi forms of degree 1 with
                     scalar index.
    
    OUTPUT:
    
    - A lazy Fourier expansion of a Jacobi form.
    
    TESTS::
    
        sage: from sage.rings import big_oh
        sage: from psage.modform.jacobiforms.jacobiformd1nn_fegenerators import _weak_jacbi_form_by_taylor_expansion
        sage: R = PowerSeriesRing(ZZ, 'q'); q = R.gen(0)
        sage: _weak_jacbi_form_by_taylor_expansion(10, 3 * [lambda p: big_oh.O(q**p)], 10).precision().jacobi_index()
        2
        sage: _weak_jacbi_form_by_taylor_expansion(10, 2 * [lambda p: big_oh.O(q**p)], 9).precision().jacobi_index()
        3
    """
    if factory is None :
        factory = JacobiFormD1NNFactory(precision, len(fs) - 1)
    
    if weight is None :
        raise ValueError( "Either one element of fs must be a modular form or " + \
                          "the weight must be passed." )
    
    expansion_ring = JacobiFormD1NNFourierExpansionModule(ZZ, len(fs) - 1 if weight % 2 == 0 else len(fs) + 1, True)
    return EquivariantMonoidPowerSeries_lazy( expansion_ring, expansion_ring.monoid().filter(precision),
                                              DelayedFactory_JacobiFormD1NN_taylor_expansion_weak( weight, fs, factory ).getcoeff,
                                              [JacobiFormD1WeightCharacter(weight)] )

#===============================================================================
# DelayedFactory_JacobiFormD1NN_taylor_expansion_weak
#===============================================================================

class DelayedFactory_JacobiFormD1NN_taylor_expansion_weak :
    r"""
    Delayed computation of the Fourier coefficients of weak Jacobi forms.
    """
    
    def __init__(self, weight, fs, factory) :
        r"""
        INPUT:
        
        - ``weight`` -- An integer.
        
        - ``fs`` -- A list of functions of one argument `p` that return a power series of
                    precision `p`.  The coefficients must be integral.
    
        - ``factory`` -- Either ``None``, or a factory class for Jacobi forms of degree 1 with
                         scalar index.
        """
        self.__weight = weight
        self.__fs = fs
        self.__factory = factory
        
        self.__series = None
        self.__ch = JacobiFormD1WeightCharacter(weight)
    
    def getcoeff(self, key, **kwds) :
        (ch, k) = key
        if ch != self.__ch :
            return ZZ.zero()
        
        if self.__series is None :
            self.__series = \
             self.__factory.by_taylor_expansion( self.__fs, self.__weight )
        
        try :
            return self.__series[k]
        except KeyError :
            return ZZ.zero()

#===============================================================================
# JacobiFormD1NNFactory
#===============================================================================

_jacobi_form_d1nn_factory_cache = dict()

def JacobiFormD1NNFactory(precision, m = None) :
    r"""
    A factory for Jacobi form of degree 1 and scalar index `m`.
    
    INPUT:
    
    - ``precision`` -- A filter for Fourier indices of Jacobi forms or
                       an integer.  In the latter case, `m` must not be
                       ``None``.
    
    - `m` -- A non-negative integer or ``None``, if ``precision`` is a filter.
    """
    
    if not isinstance(precision, JacobiFormD1NNFilter) :
        if m is None :
            raise ValueError("if precision is not filter the index m must be passed.")
        precision = JacobiFormD1NNFilter(precision, m)
    elif m is not None :
        assert precision.jacobi_index() == m
    
    global _jacobi_form_d1nn_factory_cache
        
    try :
        return _jacobi_form_d1nn_factory_cache[precision]
    except KeyError :
        tmp = JacobiFormD1NNFactory_class(precision)
        _jacobi_form_d1nn_factory_cache[precision] = tmp
        
        return tmp

#===============================================================================
# JacobiFormD1NNFactory_class
#===============================================================================

class JacobiFormD1NNFactory_class (SageObject) :
    r"""
    A factory for Jacobi form of degree 1 and scalar index.
    """
    
    def __init__(self, precision) :
        r"""
        INPUT:
        
        - ``precision`` -- A filter for Fourier indices of Jacobi forms. 
        """
        self.__precision = precision
        
        self.__power_series_ring_ZZ = PowerSeriesRing(ZZ, 'q')
        self.__power_series_ring = PowerSeriesRing(QQ, 'q')

    def jacobi_index(self) :
        r"""
        The index of the Jacobi forms that are computed by this factory. 
        """
        return self.__precision.jacobi_index()

    def power_series_ring(self) :
        r"""
        An auxiliary power series ring that is cached in the factory.
        """
        return self.__power_series_ring
    
    def integral_power_series_ring(self) :
        r"""
        An auxiliary power series ring over `\ZZ` that is cached in the factory.
        """
        return self.__power_series_ring_ZZ

    def _qexp_precision(self) :
        r"""
        The precision of Fourier expansions that are computed.
        """
        return self.__precision.index()

    def _set_wronskian_adjoint(self, wronskian_adjoint, weight_parity = 0) :
        r"""
        Set the cache for the adjoint of the wronskian. See _wronskian_adjoint.
        
        INPUT:
        
        - ``wronskian_adjoint`` -- A list of lists of power series over `\ZZ`.
        
        - ``weight_parity`` -- An integer (default: `0`).
        """
        wronskian_adjoint = [ [ e if e in self.integral_power_series_ring() else self.integral_power_series_ring()(e)
                                for e in row ]
                              for row in wronskian_adjoint ]
        
        if weight_parity % 2 == 0 :
            self.__wronskian_adjoint_even = wronskian_adjoint
        else :
            self.__wronskian_adjoint_odd = wronskian_adjoint
    
    def _wronskian_adjoint(self, weight_parity = 0, p = None) :
        r"""
        The matrix `W^\# \pmod{p}`, mentioned on page 142 of Nils Skoruppa's thesis.
        This matrix is represented by a list of lists of q-expansions.
        
        The q-expansion is shifted by `-(m + 1) (2*m + 1) / 24` in the case of even weights, and it is
        shifted by `-(m - 1) (2*m - 3) / 24` otherwise. This is compensated by the missing q-powers
        returned by _wronskian_invdeterminant.
        
        INPUT:
        
        - `p` -- A prime or ``None``.
        
        - ``weight_parity`` -- An integer (default: `0`).
        """
        try :
            if weight_parity % 2 == 0 :
                wronskian_adjoint = self.__wronskian_adjoint_even
            else :
                wronskian_adjoint = self.__wronskian_adjoint_ood

            if p is None :
                return wronskian_adjoint
            else :
                P = PowerSeriesRing(GF(p), 'q')
                
                return [map(P, row) for row in wronskian_adjoint] 

        except AttributeError :
            qexp_prec = self._qexp_precision()
            
            if p is None :
                PS = self.integral_power_series_ring()
            else :
                PS = PowerSeriesRing(GF(p), 'q')
            m = self.jacobi_index()
            
            twom = 2 * m
            frmsq = twom ** 2
            
            thetas = dict( ((i, j), dict())
                           for i in xrange(m + 1) for j in xrange(m + 1) )

            ## We want to calculate \hat \theta_{j,l} = sum_r (2 m r + j)**2l q**(m r**2 + j r)
            ## in the case of even weight, and \hat \theta_{j,l} = sum_r (2 m r + j)**(2l + 1) q**(m r**2 + j r),
            ## otherwise. 
            for r in xrange(isqrt((qexp_prec - 1 + m)//m) + 2) :
                for j in (xrange(m + 1) if weight_parity % 2 == 0 else range(1, m)) :
                    fact_p = (twom*r + j)**2
                    fact_m = (twom*r - j)**2
                    if weight_parity % 2 == 0 :
                        coeff_p = 2
                        coeff_m = 2
                    else :
                        coeff_p = 2 * (twom*r + j)
                        coeff_m = - 2 * (twom*r - j)
                    
                    for l in (xrange(m + 1) if weight_parity % 2 == 0 else range(1, m)) :
                        thetas[(j,l)][m*r**2 + r*j] = coeff_p
                        thetas[(j,l)][m*r**2 - r*j] = coeff_m
                        coeff_p = coeff_p * fact_p
                        coeff_m = coeff_m * fact_m
            if weight_parity % 2 == 0 :
                thetas[(0,0)][0] = 1
                                    
            thetas = dict( ( k, PS(th).add_bigoh(qexp_prec) )
                           for (k,th) in thetas.iteritems() )
            
            W = matrix( PS, m + 1 if weight_parity % 2 == 0 else (m - 1),
                        [ thetas[(j, l)]
                          for j in (xrange(m + 1) if weight_parity % 2 == 0 else range(1, m))
                          for l in (xrange(m + 1) if weight_parity % 2 == 0 else range(1, m)) ] )
            
            
            ## Since the adjoint of matrices with entries in a general ring
            ## is extremely slow for matrices of small size, we hard code the
            ## the cases `m = 2` and `m = 3`.  The expressions are obtained by
            ## computing the adjoint of a matrix with entries `w_{i,j}` in a
            ## polynomial algebra.
            if W.nrows() == 1 :
                Wadj = matrix(PS, [[ 1 ]])
            elif W.nrows() == 2 :
                Wadj = matrix(PS, [ [ W[1,1], -W[0,1]],
                                    [-W[1,0],  W[0,0]] ])
            
            elif W.nrows() == 3 and qexp_prec > 10**5 :
                adj00 =   W[1,1] * W[2,2] - W[2,1] * W[1,2]
                adj01 = - W[1,0] * W[2,2] + W[2,0] * W[1,2]
                adj02 =   W[1,0] * W[2,1] - W[2,0] * W[1,1]
                adj10 = - W[0,1] * W[2,2] + W[2,1] * W[0,2]
                adj11 =   W[0,0] * W[2,2] - W[2,0] * W[0,2]
                adj12 = - W[0,0] * W[2,1] + W[2,0] * W[0,1]
                adj20 =   W[0,1] * W[1,2] - W[1,1] * W[0,2]
                adj21 = - W[0,0] * W[1,2] + W[1,0] * W[0,2]
                adj22 =   W[0,0] * W[1,1] - W[1,0] * W[0,1]

                Wadj = matrix(PS, [ [adj00, adj01, adj02],
                                    [adj10, adj11, adj12],
                                    [adj20, adj21, adj22] ])
                  
            elif W.nrows() == 4 and qexp_prec > 10**5 :
                adj00 = -W[0,2]*W[1,1]*W[2,0] + W[0,1]*W[1,2]*W[2,0] + W[0,2]*W[1,0]*W[2,1] - W[0,0]*W[1,2]*W[2,1] - W[0,1]*W[1,0]*W[2,2] + W[0,0]*W[1,1]*W[2,2]
                adj01 = -W[0,3]*W[1,1]*W[2,0] + W[0,1]*W[1,3]*W[2,0] + W[0,3]*W[1,0]*W[2,1] - W[0,0]*W[1,3]*W[2,1] - W[0,1]*W[1,0]*W[2,3] + W[0,0]*W[1,1]*W[2,3]
                adj02 = -W[0,3]*W[1,2]*W[2,0] + W[0,2]*W[1,3]*W[2,0] + W[0,3]*W[1,0]*W[2,2] - W[0,0]*W[1,3]*W[2,2] - W[0,2]*W[1,0]*W[2,3] + W[0,0]*W[1,2]*W[2,3]
                adj03 = -W[0,3]*W[1,2]*W[2,1] + W[0,2]*W[1,3]*W[2,1] + W[0,3]*W[1,1]*W[2,2] - W[0,1]*W[1,3]*W[2,2] - W[0,2]*W[1,1]*W[2,3] + W[0,1]*W[1,2]*W[2,3]

                adj10 = -W[0,2]*W[1,1]*W[3,0] + W[0,1]*W[1,2]*W[3,0] + W[0,2]*W[1,0]*W[3,1] - W[0,0]*W[1,2]*W[3,1] - W[0,1]*W[1,0]*W[3,2] + W[0,0]*W[1,1]*W[3,2]
                adj11 = -W[0,3]*W[1,1]*W[3,0] + W[0,1]*W[1,3]*W[3,0] + W[0,3]*W[1,0]*W[3,1] - W[0,0]*W[1,3]*W[3,1] - W[0,1]*W[1,0]*W[3,3] + W[0,0]*W[1,1]*W[3,3]
                adj12 = -W[0,3]*W[1,2]*W[3,0] + W[0,2]*W[1,3]*W[3,0] + W[0,3]*W[1,0]*W[3,2] - W[0,0]*W[1,3]*W[3,2] - W[0,2]*W[1,0]*W[3,3] + W[0,0]*W[1,2]*W[3,3]
                adj13 = -W[0,3]*W[1,2]*W[3,1] + W[0,2]*W[1,3]*W[3,1] + W[0,3]*W[1,1]*W[3,2] - W[0,1]*W[1,3]*W[3,2] - W[0,2]*W[1,1]*W[3,3] + W[0,1]*W[1,2]*W[3,3]

                adj20 = -W[0,2]*W[2,1]*W[3,0] + W[0,1]*W[2,2]*W[3,0] + W[0,2]*W[2,0]*W[3,1] - W[0,0]*W[2,2]*W[3,1] - W[0,1]*W[2,0]*W[3,2] + W[0,0]*W[2,1]*W[3,2]
                adj21 = -W[0,3]*W[2,1]*W[3,0] + W[0,1]*W[2,3]*W[3,0] + W[0,3]*W[2,0]*W[3,1] - W[0,0]*W[2,3]*W[3,1] - W[0,1]*W[2,0]*W[3,3] + W[0,0]*W[2,1]*W[3,3]
                adj22 = -W[0,3]*W[2,2]*W[3,0] + W[0,2]*W[2,3]*W[3,0] + W[0,3]*W[2,0]*W[3,2] - W[0,0]*W[2,3]*W[3,2] - W[0,2]*W[2,0]*W[3,3] + W[0,0]*W[2,2]*W[3,3]
                adj23 = -W[0,3]*W[2,2]*W[3,1] + W[0,2]*W[2,3]*W[3,1] + W[0,3]*W[2,1]*W[3,2] - W[0,1]*W[2,3]*W[3,2] - W[0,2]*W[2,1]*W[3,3] + W[0,1]*W[2,2]*W[3,3]

                adj30 = -W[1,2]*W[2,1]*W[3,0] + W[1,1]*W[2,2]*W[3,0] + W[1,2]*W[2,0]*W[3,1] - W[1,0]*W[2,2]*W[3,1] - W[1,1]*W[2,0]*W[3,2] + W[1,0]*W[2,1]*W[3,2]
                adj31 = -W[1,3]*W[2,1]*W[3,0] + W[1,1]*W[2,3]*W[3,0] + W[1,3]*W[2,0]*W[3,1] - W[1,0]*W[2,3]*W[3,1] - W[1,1]*W[2,0]*W[3,3] + W[1,0]*W[2,1]*W[3,3]
                adj32 = -W[1,3]*W[2,2]*W[3,0] + W[1,2]*W[2,3]*W[3,0] + W[1,3]*W[2,0]*W[3,2] - W[1,0]*W[2,3]*W[3,2] - W[1,2]*W[2,0]*W[3,3] + W[1,0]*W[2,2]*W[3,3]
                adj33 = -W[1,3]*W[2,2]*W[3,1] + W[1,2]*W[2,3]*W[3,1] + W[1,3]*W[2,1]*W[3,2] - W[1,1]*W[2,3]*W[3,2] - W[1,2]*W[2,1]*W[3,3] + W[1,1]*W[2,2]*W[3,3]

                Wadj = matrix(PS, [ [adj00, adj01, adj02, adj03],
                                    [adj10, adj11, adj12, adj13],
                                    [adj20, adj21, adj22, adj23],
                                    [adj30, adj31, adj32, adj33] ])
            else :
                Wadj = W.adjoint()
            
            if weight_parity % 2 == 0 :
                wronskian_adjoint = [ [ Wadj[i,r] for i in xrange(m + 1) ]
                                      for r in xrange(m + 1) ]
            else :
                wronskian_adjoint = [ [ Wadj[i,r] for i in xrange(m - 1) ]
                                      for r in xrange(m - 1) ]
            
            if p is None :
                if weight_parity % 2 == 0 :
                    self.__wronskian_adjoint_even = wronskian_adjoint
                else :
                    self.__wronskian_adjoint_odd = wronskian_adjoint
                
            return wronskian_adjoint
    
    def _set_wronskian_invdeterminant(self, wronskian_invdeterminant, weight_parity = 0) :
        r"""
        Set the cache for the inverse determinant of the Wronskian. See _wronskian_adjoint.
        
        INPUT:
        
        - ``wronskian_invdeterminant`` -- A power series over `\ZZ`.
        
        - ``weight_parity`` -- An integer (default: `0`).
        """
        if not wronskian_invdeterminant in self.integral_power_series_ring() :
            wronskian_invdeterminant = self.integral_power_series_ring()(wronskian_invdeterminant)
        
        if weight_parity % 2 == 0 :
            self.__wronskian_invdeterminant_even = wronskian_invdeterminant
        else :
            self.__wronskian_invdeterminant_odd = wronskian_invdeterminant
    
    def _wronskian_invdeterminant(self, weight_parity = 0) :
        r"""
        The inverse determinant of `W`, which in the considered cases is always a negative
        power of the eta function. See the thesis of Nils Skoruppa.
        
        INPUT:
        
        - ``weight_parity`` -- An integer (default: `0`).
        """
        try :
            if weight_parity % 2 == 0 :
                wronskian_invdeterminant = self._wronskian_invdeterminant_even
            else :
                wronskian_invdeterminant = self._wronskian_invdeterminant_odd
        except AttributeError :
            m = self.jacobi_index()
            if weight_parity % 2 == 0 :
                pw = (m + 1) * (2 * m + 1)
            else :
                pw = (m - 1) * (2 * m - 1)
            qexp_prec = self._qexp_precision()
            
            wronskian_invdeterminant = self.integral_power_series_ring() \
                 ( [ number_of_partitions(n) for n in xrange(qexp_prec) ] ) \
                 .add_bigoh(qexp_prec) ** pw
                 
            if weight_parity % 2 == 0 :
                self._wronskian_invdeterminant_even = wronskian_invdeterminant
            else :
                self._wronskian_invdeterminant_odd = wronskian_invdeterminant

        return wronskian_invdeterminant
 
    def by_taylor_expansion(self, fs, k, is_integral=False) :
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
        
        - ``is_integral`` -- A boolean. If ``True``, the ``fs`` have integral
                             coefficients.
        
        TESTS::
            
            sage: from psage.modform.jacobiforms.jacobiformd1nn_fegenerators import *                      
            sage: from psage.modform.jacobiforms.jacobiformd1nn_fourierexpansion import *
            sage: prec = JacobiFormD1NNFilter(10, 3)
            sage: factory = JacobiFormD1NNFactory(prec)
            sage: R.<q> = ZZ[[]]
            sage: expansion = factory.by_taylor_expansion([lambda p: 0 + O(q^p), lambda p: CuspForms(1, 12).gen(0).qexp(p)], 9, True)
            sage: exp_gcd = gcd(expansion.values())
            sage: sorted([ (12 * n - r**2, c/exp_gcd) for ((n, r), c) in expansion.iteritems()])
            [(-4, 0), (-1, 0), (8, 1), (11, -2), (20, -14), (23, 32), (32, 72), (35, -210), (44, -112), (47, 672), (56, -378), (59, -728), (68, 1736), (71, -1856), (80, -1008), (83, 6902), (92, -6400), (95, -5792), (104, 10738), (107, -6564)]
        """
        if is_integral :
            PS = self.integral_power_series_ring()
        else :
            PS = self.power_series_ring()
            
        if (k % 2 == 0 and not len(fs) == self.jacobi_index() + 1) \
          or (k % 2 != 0 and not len(fs) == self.jacobi_index() - 1) :
            raise ValueError( "fs (which has length {0}) must be a list of {1} Fourier expansions" \
                              .format(len(fs), self.jacobi_index() + 1 if k % 2 == 0 else self.jacobi_index() - 1) )
        
        qexp_prec = self._qexp_precision()
        if qexp_prec is None : # there are no Fourier indices below the precision
            return dict()
        
        f_divs = dict()
        for (i, f) in enumerate(fs) :
            f_divs[(i, 0)] = PS(f(qexp_prec), qexp_prec + 10)

        ## a special implementation of the case m = 1, which is important when computing Siegel modular forms.
        ## TODO: Fix _by_taylor_expansion_m1
        if False and self.jacobi_index() == 1 and k % 2 == 0 :
            return self._by_taylor_expansion_m1(f_divs, k, is_integral)
        
        m = self.jacobi_index()
        
        for i in (xrange(m + 1) if k % 2 == 0 else xrange(m - 1)) :
            for j in xrange(1, m - i + 1) :
                f_divs[(i,j)] = f_divs[(i, j - 1)].derivative().shift(1)
            
        phi_divs = list()
        for i in (xrange(m + 1) if k % 2 == 0 else xrange(m - 1)) :
            if k % 2 == 0 :
                ## This is (13) on page 131 of Skoruppa (change of variables n -> i, r -> j).
                ## The additional factor (2m + k - 1)! is a renormalization to make coefficients integral.
                ## The additional factor (4m)^i stems from the fact that we have used d / d\tau instead of
                ## d^2 / dz^2 when treating the theta series.  Since these are annihilated by the heat operator
                ## the additional factor compensates for this. 
                phi_divs.append( sum( f_divs[(j, i - j)]
                                      * ( (4 * self.jacobi_index())**i
                                          * binomial(i,j) / 2**i # 2**(self.__precision.jacobi_index() - i + 1)
                                          * prod(2*l + 1 for l in xrange(i))
                                          / factorial(i + k + j - 1)
                                          * factorial(2*self.jacobi_index() + k - 1) ) 
                                      for j in range(i + 1) ) )
            else :
                phi_divs.append( sum( f_divs[(j, i - j)]
                                      * ( (4 * self.jacobi_index())**i
                                          * binomial(i,j) / 2**(i + 1) # 2**(self.__precision.jacobi_index() - i + 1)
                                          * prod(2*l + 1 for l in xrange(i + 1))
                                          / factorial(i + k + j)
                                          * factorial(2*self.jacobi_index() + k - 1) ) 
                                      for j in range(i + 1) ) )
                
        phi_coeffs = dict()
        for r in (xrange(m + 1) if k % 2 == 0 else xrange(1, m)) :
            if k % 2 == 0 :
                series = sum( map(operator.mul, self._wronskian_adjoint(k)[r], phi_divs) )
            else :
                series = sum( map(operator.mul, self._wronskian_adjoint(k)[r - 1], phi_divs) )
            series = self._wronskian_invdeterminant(k) * series

            for n in xrange(qexp_prec) :
                phi_coeffs[(n, r)] = series[n]

        return phi_coeffs

    def _by_taylor_expansion_m1(self, f_divs, k, is_integral=False) :
        r"""
        This provides faster implementation of by_taylor_expansion in the case
        of Jacobi index `1` (and even weight). It avoids the computation of the
        Wronskian by providing an explicit formula.
        """
        raise RuntimeError( "This code is known to have a bug. Use, for example,JacobiFormD1NNFactory_class._test__by_taylor_expansion(200, 10, 1) to check.")
        
        if is_integral :
            PS = self.integral_power_series_ring()
        else :
            PS = self.power_series_ring()
            
        qexp_prec = self._qexp_precision()
        
        
        fderiv = f_divs[(0,0)].derivative().shift(1)
        f = f_divs[(0,0)] * Integer(k/2)
        gfderiv = f_divs[(1,0)] - fderiv
        
        ab_prec = isqrt(qexp_prec + 1)
        a1dict = dict(); a0dict = dict()
        b1dict = dict(); b0dict = dict()
    
        for t in xrange(1, ab_prec + 1) :
            tmp = t**2
            a1dict[tmp] = -8*tmp
            b1dict[tmp] = -2
        
            tmp += t
            a0dict[tmp] = 8*tmp + 2
            b0dict[tmp] = 2
        b1dict[0] = -1
        a0dict[0] = 2; b0dict[0] = 2 
        
        a1 = PS(a1dict); b1 = PS(b1dict)
        a0 = PS(a0dict); b0 = PS(b0dict)

        Ifg0 = (self._wronskian_invdeterminant() * (f*a0 + gfderiv*b0)).list()
        Ifg1 = (self._wronskian_invdeterminant() * (f*a1 + gfderiv*b1)).list()

        if len(Ifg0) < qexp_prec :
            Ifg0 += [0]*(qexp_prec - len(Ifg0))
        if len(Ifg1) < qexp_prec :
            Ifg1 += [0]*(qexp_prec - len(Ifg1))

        Cphi = dict([(0,0)])
        for i in xrange(qexp_prec) :
            Cphi[-4*i] = Ifg0[i]
            Cphi[1-4*i] = Ifg1[i]

        del Ifg0[:], Ifg1[:]

        phi_coeffs = dict()
        for r in xrange(2) :
            for n in xrange(qexp_prec) :
                k = 4 * n - r**2
                if k >= 0 :
                    phi_coeffs[(n, r)] = Cphi[-k]
                               
        return phi_coeffs

    @staticmethod
    def _test__by_taylor_expansion(q_precision, weight, jacobi_index) :
        r"""
        Run tests that validate by_taylor_expansions for various indices and weights.
        
        TESTS::
            
            sage: from psage.modform.jacobiforms.jacobiformd1nn_fegenerators import *
            sage: JacobiFormD1NNFactory_class._test__by_taylor_expansion(100, 10, 2)
            sage: JacobiFormD1NNFactory_class._test__by_taylor_expansion(20, 11, 2)
            sage: JacobiFormD1NNFactory_class._test__by_taylor_expansion(50, 9, 3)      # long time 
            sage: JacobiFormD1NNFactory_class._test__by_taylor_expansion(50, 10, 10)    # long time
            sage: JacobiFormD1NNFactory_class._test__by_taylor_expansion(30, 7, 15)     # long time  
        """
        from sage.rings import big_oh
        
        prec = JacobiFormD1NNFilter(q_precision, jacobi_index)
        factory = JacobiFormD1NNFactory(prec)
        R = PowerSeriesRing(ZZ, 'q'); q = R.gen(0)
                
        if weight % 2 == 0 :
            nmb_modular_forms = jacobi_index + 1
            start_weight = weight
        else :
            nmb_modular_forms = jacobi_index - 1
            start_weight = weight + 1
            
        modular_forms = list()
        for (i,k) in enumerate(range(start_weight, start_weight + 2 * nmb_modular_forms, 2)) :
            modular_forms += [ [lambda p: big_oh.O(q**p) for _ in range(i)] + [b.qexp] + [lambda p: big_oh.O(q**p) for _ in range(nmb_modular_forms - 1 - i)]
                               for b in ModularForms(1, k).echelon_basis() ] 

        for (fs_index, fs) in enumerate(modular_forms) :
            expansion = factory.by_taylor_expansion(fs, weight, True)
            taylor_coefficients = JacobiFormD1NNFactory_class._test__jacobi_taylor_coefficients(expansion, weight, prec)
            predicted_taylor_coefficients = JacobiFormD1NNFactory_class._test__jacobi_predicted_taylor_coefficients(fs, q_precision)
            
            for (i, (proj, f)) in enumerate(zip(taylor_coefficients, predicted_taylor_coefficients)) :
                if f != proj :
                    raise AssertionError( "{0}-th Taylor coefficient of the {1}-th Jacobi form is not correct. Expansions are\n  {2}\nand\n {3}".format(i, fs_index, proj, f) )
    
    @staticmethod
    def _test__jacobi_taylor_coefficients(expansion, weight, prec = None) :
        r"""
        Compute the renormalized Taylor coefficients of
        a Jacobi form.
        
        INPUT:
        
        - ``expansion`` -- A Fourier expansion or a dictionary with corresponding keys.
        
        - ``weight`` -- An integer.
        
        - ``prec`` -- A filter for Fourier expansions, of if ``expansion`` is a Fourier expansion
                      possibly ``None``.
        
        OUTPUT:
        
        - A list of power series in `q`.
        """
        from sage.rings.arith import gcd
        
        if prec is None :
            prec = expansion.precision()
        jacobi_index = prec.jacobi_index()
        q_precision = prec.index()
        R = PowerSeriesRing(ZZ, 'q'); q = R.gen(0)
                
        weak_prec = JacobiFormD1NNFilter(prec, reduced = False, weak_forms = True)
        indices = JacobiFormD1NNIndices(jacobi_index)

        projs = list()
        for pw in (range(0, 2 * jacobi_index + 1, 2) if weight % 2 == 0 else range(1, 2 * jacobi_index - 1, 2)) :
            proj = dict( (n, 0) for n in range(q_precision) )
            for (n, r) in weak_prec :
                ((nred, rred), sign) = indices.reduce((n,r))
                try :
                    proj[n] +=  (sign * r)**pw * expansion[(nred, rred)]
                except (KeyError, ValueError) :
                    pass
            
            projs.append(proj)

        gcd_projs = [gcd(proj.values()) for proj in projs]
        gcd_projs = [g if g != 0 else 1 for g in gcd_projs]
        projs = [sorted(proj.iteritems()) for proj in projs]
        projs = [ R([c for (_, c) in proj]).add_bigoh(q_precision) / gcd_proj
                  for (proj, gcd_proj) in zip(projs, gcd_projs) ]
        
        return projs

    @staticmethod
    def _test__jacobi_predicted_taylor_coefficients(fs, q_precision) :
        r"""
        Given a list of power series, which are the corrected Taylor coefficients
        of a Jacobi form, return the renormalized uncorrected ones, assuming that
        all but one `f` vanish.
        
        INPUT:
        
        - ``fs`` -- A list of power series.
        
        - ``q_precision`` -- An integer.
        
        OUPUT:
        
        - A list of power series.
        
        TESTS:
        
        See jacobi_form_by_taylor_expansion.
        """
        from sage.rings.arith import gcd
        
        R = PowerSeriesRing(ZZ, 'q'); q = R.gen(0)
        
        diff = lambda f: f.derivative().shift(1)
        normalize = lambda f: f / gcd(f.list()) if f != 0 else f
        diffnorm = lambda f,l: normalize(reduce(lambda a, g: g(a), l*[diff], f))

        taylor_coefficients = list()
        allf = R(0)
        for f in fs :
            allf = f(q_precision) + diffnorm(allf, 1)            
            taylor_coefficients.append(allf)

        return taylor_coefficients

    @staticmethod
    def _test__jacobi_corrected_taylor_expansions(nu, phi, weight) :
        r"""
        Return the ``2 nu``-th corrected Taylor coefficient.
        INPUT:
        
        - ``nu`` -- An integer.  
        
        - ``phi`` -- A Fourier expansion of a Jacobi form.
        
        - ``weight`` -- An integer.
        
        OUTPUT:
        
        - A power series in `q`.
        
        ..TODO:
        
        Implement this for all Taylor coefficients.
        
        TESTS::
        
            sage: from psage.modform.jacobiforms.jacobiformd1nn_fegenerators import *
            sage: from psage.modform.jacobiforms.jacobiformd1nn_types import *
            sage: nu_bound = 10
            sage: precision = 100
            sage: weight = 10; index = 7
            sage: phis = [jacobi_form_by_taylor_expansion(i, JacobiFormD1NNFilter(precision, index), weight) for i in range(JacobiFormD1NN_Gamma(weight, index)._rank(QQ))]
            sage: fss = [ [JacobiFormD1NNFactory_class._test__jacobi_corrected_taylor_expansions(nu, phi, weight) for phi in phis] for nu in range(nu_bound) ]
            sage: fss_vec = [ [vector(f.padded_list(precision)) for f in fs] for fs in fss ]
            sage: mf_spans = [ span([vector(b.qexp(precision).padded_list(precision)) for b in ModularForms(1, weight + 2 * nu).basis()]) for nu in range(nu_bound) ] 
            sage: all(f_vec in mf_span for (fs_vec, mf_span) in zip(fss_vec, mf_spans) for f_vec in fs_vec)
            True
        """
        ## We use EZ85, p.29 (3), the factorial in one of the factors is missing
        factors = [ (-1)**mu * factorial(2 * nu) * factorial(weight + 2 * nu - mu - 2) / ZZ(factorial(mu) * factorial(2 * nu - 2 * mu) * factorial(weight + nu - 2))
                    for mu in range(nu + 1) ]
        gegenbauer = lambda n, r: sum( f * r**(2 * nu - 2 * mu) * n**mu 
                                       for (mu,f) in enumerate(factors) )
        ch = JacobiFormD1WeightCharacter(weight)
        jacobi_index = phi.precision().jacobi_index()
        
        coeffs = dict( (n, QQ(0)) for n in range(phi.precision().index()) )
        for (n, r) in phi.precision().monoid_filter() :
            coeffs[n] += gegenbauer(jacobi_index * n, r) * phi[(ch, (n,r))]
        
        return PowerSeriesRing(QQ, 'q')(coeffs)

    @staticmethod
    def _test__jacobi_torsion_point(phi, weight, torsion_point) :
        r"""
        Given a list of power series, which are the corrected Taylor coefficients
        of a Jacobi form, return the specialization to ``torsion_point``.
        
        INPUT:
        
        - ``phi`` -- A Fourier expansion of a Jacobi form.
        
        - ``weight`` -- An integer.
        
        - ``torsion_point`` -- A rational.
        
        OUPUT:
        
        - A power series.
        
        TESTS:
                
        See jacobi_form_by_taylor_expansion.
        
            sage: from psage.modform.jacobiforms.jacobiformd1nn_fegenerators import *
            sage: from psage.modform.jacobiforms.jacobiformd1nn_types import *
            sage: precision = 50
            sage: weight = 10; index = 7
            sage: phis = [jacobi_form_by_taylor_expansion(i, JacobiFormD1NNFilter(precision, index), weight) for i in range(JacobiFormD1NN_Gamma(weight, index)._rank(QQ))]
            sage: fs = [JacobiFormD1NNFactory_class._test__jacobi_torsion_point(phi, weight, 2/3) for phi in phis]
            sage: fs_vec = [vector(f.padded_list(precision)) for f in fs]
            sage: mf_span = span([vector(b.qexp(precision).padded_list(precision)) for b in ModularForms(GammaH(9, [4]), weight).basis()])
            sage: all(f_vec in mf_span for f_vec in fs_vec)
            True
        
        FIXME: The case of torsion points of order 5, which should lead to forms for Gamma1(25) fails even in the simplest case.
        """
        from sage.rings.all import CyclotomicField
        
        K = CyclotomicField(QQ(torsion_point).denominator()); zeta = K.gen()
        R = PowerSeriesRing(K, 'q'); q = R.gen(0)

        ch = JacobiFormD1WeightCharacter(weight)
    
        coeffs = dict( (n, QQ(0)) for n in range(phi.precision().index()) )
        for (n, r) in phi.precision().monoid_filter() :
            coeffs[n] += zeta**r * phi[(ch, (n,r))]
        
        return PowerSeriesRing(K, 'q')(coeffs) 
    
