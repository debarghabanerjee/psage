r"""
The Hecke action on Jacobi forms of degree 1 and scalar index.

AUTHORS:

- Martin Raum (2012 - 12 - 03) Initial version.
"""

#===============================================================================
# 
# Copyright (C) 2012 Martin Raum
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

from sage.misc.cachefunc import cached_method
from sage.misc.latex import latex
from sage.rings.arith import fundamental_discriminant, inverse_mod, kronecker_symbol
from sage.rings.integer import Integer
from sage.structure.sage_object import SageObject

_jacobiformd1nn_heckeoperator_cache = dict()

#===============================================================================
# SiegelModularFormG2FourierExpansionHeckeAction
#===============================================================================

def JacobiFormD1NNFourierExpansionHeckeAction( weight, jacobi_index, l ) :
    r"""
        INPUT:

        - ``weight`` -- An integer.
        
        - ``jacobi_index`` -- A positive integer.
        
        - `l` -- A positive integer.
    """
    global _jacobiformd1nn_heckeoperator_cache
    
    key = (weight, jacobi_index, l)
    try :
        return _jacobiformd1nn_heckeoperator_cache[key]
    except KeyError :
        A = JacobiFormD1NNFourierExpansionHeckeAction_class(weight, jacobi_index, l)
        _jacobiformd1nn_heckeoperator_cache[key] = A
        
        return A

#===============================================================================
# JacobiFormD1NNFourierExpansionHeckeAction_class
#===============================================================================

class JacobiFormD1NNFourierExpansionHeckeAction_class ( SageObject ) :
    r"""
    Implements the Hecke operator `T_l` acting on Jacobi forms of degree `1`, weight `k`, and index `m`.
    
    TESTS:

        sage: from psage.modform.jacobiforms import *
        sage: jfs = JacobiFormsD1NN(QQ, JacobiFormD1NNGamma(20, 3), 200)
        sage: hos = [jfs.graded_submodule(None).hecke_homomorphism(l) for l in range(2, 10) if gcd(3, l) == 1]
    """

    def __init__(self, weight, jacobi_index, l) :
        r"""
        INPUT:

        - ``weight`` -- An integer.
        
        - ``jacobi_index`` -- A positive integer.
        
        - `l` -- A positive integer.
        """
        if not Integer(jacobi_index).gcd(l) == 1 :
            raise ValueError( "Jacobi index and Hecke index must be coprime" )

        self.__weight = weight
        self.__jacobi_index = jacobi_index
        self.__l = l

        self.__lsq_divisors = Integer(l**2).divisors()

    def eval(self, expansion, weight = None) :
        precision = expansion.precision()
        if precision.is_infinite() :
            precision = expansion._bounding_precision()
        else :
            precision = precision._hecke_operator(self.__l)
        characters = expansion.non_zero_components()
        
        hecke_expansion = dict()
        for ch in characters :
            hecke_expansion[ch] = dict( (k, self.hecke_coeff(expansion, ch, k)) for k in precision )
        
        result = expansion.parent()._element_constructor_(hecke_expansion)
        result._set_precision(expansion.precision()._hecke_operator(self.__l))
        
        return result
        
    def hecke_coeff(self, expansion, ch, (n, r)) :
        r"""
        Computes the coefficient indexed by `(n, r)` of `T_l (\phi)`.
        """
        l = self.__l
        coeff = Integer(0)

        for a in self.__lsq_divisors :
            if ( l**2 * (r**2 - 4 * self.__jacobi_index * n) ) % a**2 != 0 :
                continue
            if ( (l**2 * (r**2 - 4 * self.__jacobi_index * n)) // a**2 ) % 4 not in [0, 1] :
                continue

            if a % 2 == 0 :
                # a / 2 r' = l / 2 r (m) defines r' uniquely
                rprime = ( r * (l // 2) * inverse_mod(a // 2, self.__jacobi_index) ) % (2 * self.__jacobi_index)
            else :
                rprime = ( r * l * inverse_mod(a, 2 * self.__jacobi_index) ) % (2 * self.__jacobi_index)

            nprime = rprime**2 - (l**2 * (r**2 - 4 * self.__jacobi_index * n)) // a**2
            if nprime % (4 * self.__jacobi_index) != 0 :
                assert a % 2 == 0
                rprime = rprime + self.__jacobi_index
                nprime = rprime**2 - (l**2 * (r**2 - 4 * self.__jacobi_index * n)) // a**2
            nprime = nprime // (4 * self.__jacobi_index)

            try :
                coeff = coeff + self.epsilon(r**2 - 4 * self.__jacobi_index * n, a) \
                                * a**(self.__weight - 2) \
                                * expansion[( ch, (nprime, rprime) )]
            except KeyError:
                raise ValueError( "Coefficient {0} is not defined".format( (nprime, rprime) ))
        
        return coeff

    def epsilon(self, D, n) :
        r"""
        The character `\epsilon` defined on page 50 of Eicher, Zagier - The theory of Jacobi forms.
        """
        if D == 0 :
            r = Integer(n).isqrt()
            if r**2 == n :
                return r
            else :
                return 0
        
        D0 = fundamental_discriminant(D)
        fsq = D // D0

        gsq = fsq.gcd(n)
        g = gsq.isqrt()

        if gsq != g**2 :
            return 0

        n0 = n // gsq

        f = fsq.isqrt()
        return kronecker_symbol(D0, n0) * g

    def _repr_(self) :
        return "{0}-th Hecke operator for Jacobi forms of weight {1} and index {2}".format(self.__l, self.__weight, self.__jacobi_index)

    def _latex_(self) :
        return r"\text{{${0}$-th Hecke operator for Jacobi forms of weight ${1}$ and index ${2}$}}".format(latex(self.__l), latex(self.__weight), latex(self.__jacobi_index))




