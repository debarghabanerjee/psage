"""
Functions for calculating Borcherds product.

REFERENCES:
    -- [GKR] Gehre, Kreuzer, Raum - Hermitian Borcherds products (???)

AUTHOR :
    -- Dominic Gehre, Martin Raum (2010 - 08 - 23) Initial version
    
EXAMPLES:

In order to compute a Borcherds product (up to norm 1 elements in `\CC`) use
the following commands.

::

    sage: from hermitianmodularforms.hermitianmodularformd2_borcherdsproducts import *
    sage: from hermitianmodularforms.hermitianmodularformd2_fourierexpansion import *
    sage: vv = {(0,0):{-3:1,0:90,3:100116},(1,0):{-1:0,2:16038,5:2125035},(-1,0):{-1:0,2:16038,5:2125035}}
    sage: index_filter = HermitianModularFormD2Filter_diagonal(6,-3)
    sage: f = borcherds_product__by_logarithm(vv, index_filter)
    
The ``index_filter`` determines the precision up to which the Borcherds products will be
computed.  In this case, the filter corresponds to all quadratic forms `[a, b, c]`
over ``\QQ(\sqrt{-3})`` satisfying `a, c < 6`.  The Borcherds product comes from
a vector valued modular form (see Borcherds' orginial paper
"Automorphic forms with singularities on Grassmannians").  The lattice that Borcherds
refers to is `L = U \oplus U \oplus (-1) L_0`, where `U` is the unimodular hyperbolic lattice
of rank `2`, and `L_0` is the norm lattice of the ring of integers of `\QQ(\sqrt{D})` (`D = -3`).
See hermitianmodularformd2_fegenerators.py for the format of `vv`, which represents
the vector valued modular forms.   
"""

#===============================================================================
# 
# Copyright (C) 2010 Dominic Gehre, Martin Raum
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

from psage.modform.fourier_expansion_framework.monoidpowerseries.monoidpowerseries_element import EquivariantMonoidPowerSeries
from hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import \
                            HermitianModularFormD2Filter_diagonal_borcherds, HermitianModularFormD2Indices_diagonal_borcherds
from hermitianmodularforms.hermitianmodularformd2_fegenerators import HermitianModularFormD2Factory_class
from hermitianmodularforms.hermitianmodularformd2_fourierexpansion import \
                            HermitianModularFormD2FourierExpansionRing, HermitianModularFormD2Filter_diagonal,\
                            HermitianModularFormD2FourierExpansionTrivialCharacter, \
                            HermitianModularFormD2FourierExpansionTransposeCharacter
from hermitianmodularforms.hermitianmodularformd2_fourierexpansion_cython import *
from sage.functions.other import ceil
from sage.matrix.all import matrix
from sage.misc.all import prod
from sage.misc.functional import isqrt
from sage.quadratic_forms.all import QuadraticForm 
from sage.rings.all import Integer, Rational
from sage.rings.all import QQ
from sage.rings.all import ZZ, LaurentPolynomialRing
from sage.rings.all import multinomial, sigma, factorial
import operator

#===============================================================================
# definite_positive_diagonal_part__diagonal_precision
#===============================================================================

def definite_positive_diagonal_part__diagonal_precision(coefficients, a_precision, c_precision, D) :
    """
    The part `\mathfrak{A}` in [GKR] and its powers. It subsumes all indices `(a,b,c)` that are definite and 
    satisfy `a, c > 0`.
    
    INPUT:
        - ``coefficients`` -- A dictionary tuple -> (dictionary Integer -> ring element);
                              The coefficients separated by components of a vector valued elliptic
                              modular form.
        - ``a_precision``  -- A positive integer; An upper bound for `a` in the result's indices.
        - ``c_precision``  -- A positive integer; An upper bound for `c` in the result's indices.
        - `D`              -- A negative integer; A discriminant of an imaginary quadratic field.
    
    NOTE:
        Since ``coefficients`` must come from a modular form, it is automatically symmetric under conjugation
        of indices. We assume this when performing the calculations.
    
    OUTPUT:
        A list of instances of :class:~`fourier_expansion_framework.monoidpowerseries.monoidpowerseries_element.EquivariantMonoidPowerSeries`.
    
    TESTS::
        sage: definite_positive_diagonal_part__diagonal_precision({}, 2, 4, -3)[1].coefficients(True)
        {1: {}}
        sage: coefficients = {(1, 0): {}, (0, 0): {0: 4, 1: 1, 2: 8, 3: 8, 4: -3, 5: -7, 6: -7, 7: -7, 8: -10, 9: 10}, (-1, 0): {}}
        sage: definite_positive_diagonal_part__diagonal_precision(coefficients, 2, 4, -3)[0].coefficients()
        {(0, 0, 0, 0): 1}
        sage: definite_positive_diagonal_part__diagonal_precision(coefficients, 2, 4, -3)[1].coefficients()
        {(1, 0, 0, 1): -8, (1, 0, 0, 2): 7, (1, 0, 0, 3): -10}
        sage: coefficients = {(1, 0): {0: 2, 1: -6, 2: 6, 3: -9, 4: -10, 5: 2, 6: -2, 7: -9, 8: 4, 9: 6}, (0, 0): {0: 7, 1: -9, 2: 1, 3: 0, 4: 5, 5: 0, 6: -10, 7: 7, 8: 7, 9: -3}, (-1, 0): {0: 2, 1: -6, 2: 6, 3: -9, 4: -10, 5: 2, 6: -2, 7: -9, 8: 4, 9: 6}}
        sage: definite_positive_diagonal_part__diagonal_precision(coefficients, 4, 4, -3)[1].coefficients()
        {(2, 3, 2, 2): 3, (1, 1, 1, 2): -2, (2, 2, 2, 2): -7, (2, 0, 0, 2): 0, (1, 1, 1, 1): -6, (1, 0, 0, 1): 0, (1, 0, 0, 2): 10, (1, 1, 1, 3): -4, (3, 3, 3, 3): -2, (3, 0, 0, 3): 0, (1, 0, 0, 3): 3}
        sage: definite_positive_diagonal_part__diagonal_precision(coefficients, 4, 4, -3)[2].coefficients()
        {(2, 3, 2, 2): 72, (2, 3, 2, 3): 192, (2, 2, 2, 2): 36, (2, 1, 1, 2): 72, (3, 1, 1, 3): -4, (2, 0, 0, 2): 216, (3, 3, 2, 3): 328, (3, 2, 2, 3): -112, (2, 0, 0, 3): 144, (3, 4, 3, 3): 120, (3, 3, 3, 3): 180, (3, 0, 0, 3): 680, (2, 2, 2, 3): 24}
    """
    reduce_b = HermitianModularFormD2Factory_class(HermitianModularFormD2Filter_diagonal(0, D = D))._reduce_vector_valued_index
        
    exponents = dict()
    
    index_filter = HermitianModularFormD2Filter_diagonal(max(a_precision, c_precision), D = D)
    for ((a, b1, b2, c),_,disc) in index_filter.iter_positive_forms_for_character_with_content_and_discriminant() :
        for n in range(1, min((a_precision - 1) // a, (c_precision - 1) // c) + 1) :
            try :
                exponents[(n*a, n*b1, n*b2, n*c)] += -Integer(coefficients[reduce_b((b1, b2))][disc]) / n
            except KeyError :
                try :
                    exponents[(n*a, n*b1, n*b2, n*c)] = -Integer(coefficients[reduce_b((b1, b2))][disc]) / n
                except KeyError :
                    pass

            
    sym_character = HermitianModularFormD2FourierExpansionTrivialCharacter(D, ZZ)

    fering = HermitianModularFormD2FourierExpansionRing(ZZ, D)
    
    monoidpowerseries = [fering(1), EquivariantMonoidPowerSeries(fering, {sym_character: exponents}, index_filter)]
    for _ in range(2, min(a_precision, c_precision)) :
        monoidpowerseries.append(monoidpowerseries[1] * monoidpowerseries[-1])
    
    return monoidpowerseries

#===============================================================================
# indefinite_positive_diagonal_part__diagonal_precision
#===============================================================================

def indefinite_positive_diagonal_part__diagonal_precision(coefficients, a_precision, c_precision, discriminant_bound, D) :
    """
    The part `\mathfrak{B}` in [GKR] and its powers. It subsumes all indices `(a,b,c)` that are indefinite or semidefinite
    and that satisfy `c > 0,\,a > 0`.
    
    INPUT:
        - ``coefficients``       -- A dictionary tuple -> (dictionary Integer -> ring element);
                                    The coefficients separated by components of a vector valued elliptic
                                    modular form.
        - ``a_precision``        -- A positive integer; An upper bound for `a` in the result's indices.
        - ``c_precision``        -- A positive integer; An upper bound for `c` in the result's indices.
        - ``discriminant_bound`` -- A negative integer; A lower bound for the coefficients that do
                                    not vanish.
        - `D`                    -- A negative integer; A discriminant of an imaginary quadratic field.
    
    OUTPUT:
        A list of Laurent polynomial.
    
    TESTS::
        sage: indefinite_positive_diagonal_part__diagonal_precision({}, 3, 7, -10, -3)
        [1, 0, 0]
        sage: coefficients = {(1, 0): {0: -1, 1: 8, 2: -4, 3: 4, 4: -7, 5: -3, 6: 7, 7: 1, 8: -9, 9: 4, -2: 10, -1: -5}, (0, 0): {0: 2, 1: 6, 2: 10, 3: -9, 4: -8, 5: 6, 6: -2, 7: -2, 8: -7, 9: -8, -2: -8, -1: 8}, (-1, 0): {0: -1, 1: 8, 2: -4, 3: 4, 4: -7, 5: -3, 6: 7, 7: 1, 8: -9, 9: 4, -2: 10, -1: -5}}
        sage: indefinite_positive_diagonal_part__diagonal_precision(coefficients, 2, 4, -2, -3)[1]
        2*ae*b1e^6*b2e^3*ce^3 - 5*ae*b1e^5*b2e^3*ce^2 - 5*ae*b1e^5*b2e^2*ce^2 - 5*ae*b1e^4*b2e^3*ce^2 + 2*ae*b1e^3*b2e^3*ce^3 - 5*ae*b1e^4*b2e^2*ce - 5*ae*b1e^4*b2e*ce^2 + 2*ae*b1e^3*b2e^2*ce + 2*ae*b1e^3*ce^3 + 2*ae*b1e^3*b2e*ce - 5*ae*b1e^2*b2e^2*ce - 5*ae*b1e*b2e^2*ce^2 - 5*ae*b1e^2*ce + 2*ae*b2e*ce - 5*ae*b1e*b2e^-1*ce^2 - 5*ae*b1e^-1*b2e*ce^2 + 2*ae*b2e^-1*ce + 2*ae*b1e^-3*ce^3 - 5*ae*b1e^-2*ce - 5*ae*b1e^-1*b2e^-2*ce^2 - 5*ae*b1e^-2*b2e^-2*ce + 2*ae*b1e^-3*b2e^-1*ce - 5*ae*b1e^-4*b2e^-1*ce^2 + 2*ae*b1e^-3*b2e^-3*ce^3 + 2*ae*b1e^-3*b2e^-2*ce - 5*ae*b1e^-4*b2e^-2*ce - 5*ae*b1e^-4*b2e^-3*ce^2 - 5*ae*b1e^-5*b2e^-2*ce^2 - 5*ae*b1e^-5*b2e^-3*ce^2 + 2*ae*b1e^-6*b2e^-3*ce^3
        sage: coefficients = {(1, 0): {-1: 2, -5: -10, -4: -4, -3: 4, -2: 8}, (0, 0): {-1: -8, -5: -6, -4: 6, -3: -6, -2: 1}, (-1, 0): {-1: 2, -5: -10, -4: -4, -3: 4, -2: 8}}
        sage: indefinite_positive_diagonal_part__diagonal_precision(coefficients, 2, 4, -3, -3)[1]
        2*ae*b1e^5*b2e^3*ce^2 + 2*ae*b1e^5*b2e^2*ce^2 + 2*ae*b1e^4*b2e^3*ce^2 + 2*ae*b1e^4*b2e^2*ce + 2*ae*b1e^4*b2e*ce^2 + 2*ae*b1e^2*b2e^2*ce + 2*ae*b1e*b2e^2*ce^2 + 2*ae*b1e^2*ce + 2*ae*b1e*b2e^-1*ce^2 + 2*ae*b1e^-1*b2e*ce^2 + 2*ae*b1e^-2*ce + 2*ae*b1e^-1*b2e^-2*ce^2 + 2*ae*b1e^-2*b2e^-2*ce + 2*ae*b1e^-4*b2e^-1*ce^2 + 2*ae*b1e^-4*b2e^-2*ce + 2*ae*b1e^-4*b2e^-3*ce^2 + 2*ae*b1e^-5*b2e^-2*ce^2 + 2*ae*b1e^-5*b2e^-3*ce^2
        sage: indefinite_positive_diagonal_part__diagonal_precision(coefficients, 4, 4, -3, -3)[1]
        2/3*ae^3*b1e^12*b2e^6*ce^3 + 2*ae^3*b1e^10*b2e^6*ce^3 + 2*ae^3*b1e^10*b2e^4*ce^3 + 2*ae^3*b1e^8*b2e^6*ce^3 + 2*ae^3*b1e^8*b2e^5*ce^2 + 2*ae^2*b1e^8*b2e^5*ce^3 + 2/3*ae^3*b1e^6*b2e^6*ce^3 + 2*ae^3*b1e^7*b2e^5*ce^2 + 2*ae^2*b1e^7*b2e^5*ce^3 + 2*ae^3*b1e^8*b2e^3*ce^2 + ae^2*b1e^8*b2e^4*ce^2 + 2*ae^3*b1e^8*b2e^2*ce^3 + 2*ae^2*b1e^8*b2e^3*ce^3 + 2*ae^2*b1e^7*b2e^4*ce^2 + 2*ae^3*b1e^7*b2e^2*ce^2 + 2*ae^2*b1e^7*b2e^3*ce^2 + 2*ae^2*b1e^7*b2e^2*ce^3 + 2*ae^2*b1e^5*b2e^4*ce^2 + ae^2*b1e^4*b2e^4*ce^2 + 2/3*ae^3*b1e^6*ce^3 + 2*ae^3*b1e^2*b2e^4*ce^3 + 2*ae^2*b1e^5*b2e^3*ce + 2*ae*b1e^5*b2e^3*ce^2 + 2*ae^2*b1e^5*b2e^2*ce + 2*ae^2*b1e^4*b2e^3*ce + 2*ae^2*b1e^5*b2e*ce^2 + 2*ae*b1e^5*b2e^2*ce^2 + 2*ae*b1e^4*b2e^3*ce^2 + 2*ae^3*b1e*b2e^3*ce^2 + 2*ae^2*b1e^2*b2e^3*ce^2 + 2*ae^2*b1e*b2e^3*ce^3 + 2*ae^2*b1e^4*b2e*ce + 2*ae*b1e^4*b2e^2*ce + ae^2*b1e^4*ce^2 + 2*ae*b1e^4*b2e*ce^2 + 2*ae^2*b1e*b2e^2*ce + 2*ae*b1e^2*b2e^2*ce + 2*ae^3*b1e^-1*b2e^2*ce^2 + 2*ae*b1e*b2e^2*ce^2 + 2*ae^3*b1e^2*b2e^-2*ce^3 + 2*ae^3*b1e^-2*b2e^2*ce^3 + 2*ae^2*b1e^-1*b2e^2*ce^3 + 2*ae^2*b1e^2*b2e^-1*ce^2 + 2*ae*b1e^2*ce + 2*ae^3*b1e*b2e^-2*ce^2 + 2*ae^2*b1e*b2e^-2*ce^3 + 2*ae^2*b1e*b2e^-1*ce + 2*ae^2*b1e^-1*b2e*ce + 2*ae*b1e*b2e^-1*ce^2 + 2*ae^2*b1e^-2*b2e*ce^2 + 2*ae*b1e^-1*b2e*ce^2 + 2*ae^3*b1e^-1*b2e^-3*ce^2 + 2*ae^2*b1e^-1*b2e^-3*ce^3 + 2*ae^2*b1e^-1*b2e^-2*ce + 2*ae*b1e^-2*ce + 2*ae*b1e^-1*b2e^-2*ce^2 + ae^2*b1e^-4*ce^2 + 2*ae^3*b1e^-2*b2e^-4*ce^3 + 2/3*ae^3*b1e^-6*ce^3 + 2*ae^2*b1e^-2*b2e^-3*ce^2 + 2*ae*b1e^-2*b2e^-2*ce + 2*ae^2*b1e^-4*b2e^-1*ce + 2*ae^2*b1e^-5*b2e^-1*ce^2 + 2*ae*b1e^-4*b2e^-1*ce^2 + 2*ae^2*b1e^-4*b2e^-3*ce + 2*ae^2*b1e^-5*b2e^-2*ce + 2*ae*b1e^-4*b2e^-2*ce + ae^2*b1e^-4*b2e^-4*ce^2 + 2*ae*b1e^-4*b2e^-3*ce^2 + 2*ae^3*b1e^-7*b2e^-2*ce^2 + 2*ae*b1e^-5*b2e^-2*ce^2 + 2*ae^3*b1e^-8*b2e^-2*ce^3 + 2*ae^2*b1e^-7*b2e^-2*ce^3 + 2*ae^2*b1e^-5*b2e^-3*ce + 2*ae^2*b1e^-5*b2e^-4*ce^2 + 2*ae*b1e^-5*b2e^-3*ce^2 + 2*ae^3*b1e^-8*b2e^-3*ce^2 + 2*ae^2*b1e^-7*b2e^-3*ce^2 + 2/3*ae^3*b1e^-6*b2e^-6*ce^3 + 2*ae^2*b1e^-8*b2e^-3*ce^3 + 2*ae^3*b1e^-7*b2e^-5*ce^2 + 2*ae^2*b1e^-7*b2e^-4*ce^2 + 2*ae^2*b1e^-7*b2e^-5*ce^3 + 2*ae^3*b1e^-8*b2e^-5*ce^2 + ae^2*b1e^-8*b2e^-4*ce^2 + 2*ae^3*b1e^-8*b2e^-6*ce^3 + 2*ae^2*b1e^-8*b2e^-5*ce^3 + 2*ae^3*b1e^-10*b2e^-4*ce^3 + 2*ae^3*b1e^-10*b2e^-6*ce^3 + 2/3*ae^3*b1e^-12*b2e^-6*ce^3
        sage: indefinite_positive_diagonal_part__diagonal_precision(coefficients, 4, 4, -3, -3)[2]
        4*ae^3*b1e^12*b2e^6*ce^3 + 8*ae^3*b1e^11*b2e^6*ce^3 + 8*ae^3*b1e^11*b2e^5*ce^3 + 12*ae^3*b1e^10*b2e^6*ce^3 + 16*ae^3*b1e^10*b2e^5*ce^3 + 32*ae^3*b1e^9*b2e^6*ce^3 + 12*ae^3*b1e^10*b2e^4*ce^3 + 24*ae^3*b1e^9*b2e^5*ce^3 + 12*ae^3*b1e^8*b2e^6*ce^3 + 8*ae^3*b1e^9*b2e^5*ce^2 + 24*ae^3*b1e^9*b2e^4*ce^3 + 8*ae^2*b1e^9*b2e^5*ce^3 + 8*ae^3*b1e^7*b2e^6*ce^3 + 8*ae^3*b1e^9*b2e^4*ce^2 + 8*ae^3*b1e^8*b2e^5*ce^2 + 32*ae^3*b1e^9*b2e^3*ce^3 + 16*ae^3*b1e^8*b2e^4*ce^3 + 8*ae^2*b1e^9*b2e^4*ce^3 + 8*ae^2*b1e^8*b2e^5*ce^3 + 4*ae^3*b1e^6*b2e^6*ce^3 + 8*ae^3*b1e^7*b2e^5*ce^2 + 8*ae^3*b1e^7*b2e^4*ce^3 + 24*ae^3*b1e^6*b2e^5*ce^3 + 8*ae^2*b1e^7*b2e^5*ce^3 + 8*ae^3*b1e^8*b2e^3*ce^2 + 8*ae^3*b1e^7*b2e^4*ce^2 + 4*ae^2*b1e^8*b2e^4*ce^2 + 8*ae^3*b1e^6*b2e^5*ce^2 + 12*ae^3*b1e^8*b2e^2*ce^3 + 8*ae^3*b1e^7*b2e^3*ce^3 + 8*ae^2*b1e^8*b2e^3*ce^3 + 24*ae^3*b1e^6*b2e^4*ce^3 + 8*ae^2*b1e^7*b2e^4*ce^3 + 16*ae^3*b1e^5*b2e^5*ce^3 + 8*ae^2*b1e^6*b2e^5*ce^3 + 8*ae^3*b1e^7*b2e^3*ce^2 + 8*ae^2*b1e^7*b2e^3*ce^3 + 8*ae^3*b1e^5*b2e^4*ce^3 + 8*ae^3*b1e^4*b2e^5*ce^3 + 8*ae^3*b1e^7*b2e^2*ce^2 + 16*ae^3*b1e^6*b2e^3*ce^2 + 8*ae^3*b1e^5*b2e^4*ce^2 + 8*ae^2*b1e^6*b2e^4*ce^2 + 8*ae^3*b1e^7*b2e*ce^3 + 24*ae^3*b1e^6*b2e^2*ce^3 + 8*ae^2*b1e^7*b2e^2*ce^3 + 24*ae^3*b1e^5*b2e^3*ce^3 + 16*ae^2*b1e^6*b2e^3*ce^3 + 16*ae^3*b1e^4*b2e^4*ce^3 + 8*ae^2*b1e^5*b2e^4*ce^3 + 24*ae^3*b1e^6*b2e*ce^3 + 24*ae^3*b1e^5*b2e^2*ce^3 + 24*ae^3*b1e^4*b2e^3*ce^3 + 24*ae^3*b1e^3*b2e^4*ce^3 + 8*ae^3*b1e^6*b2e*ce^2 + 8*ae^2*b1e^6*b2e^2*ce^2 + 8*ae^3*b1e^3*b2e^4*ce^2 + 4*ae^2*b1e^4*b2e^4*ce^2 + 4*ae^3*b1e^6*ce^3 + 8*ae^3*b1e^5*b2e*ce^3 + 8*ae^2*b1e^6*b2e*ce^3 + 4*ae^3*b1e^4*b2e^2*ce^3 + 12*ae^3*b1e^2*b2e^4*ce^3 + 8*ae^2*b1e^3*b2e^4*ce^3 + 8*ae^3*b1e^5*b2e*ce^2 + 16*ae^3*b1e^3*b2e^3*ce^2 + 16*ae^3*b1e^5*ce^3 + 24*ae^3*b1e^4*b2e*ce^3 + 8*ae^2*b1e^5*b2e*ce^3 + 32*ae^3*b1e^3*b2e^2*ce^3 + 8*ae^3*b1e^2*b2e^3*ce^3 + 16*ae^2*b1e^3*b2e^3*ce^3 + 16*ae^3*b1e^3*b2e^2*ce^2 + 8*ae^2*b1e^4*b2e^2*ce^2 + 8*ae^3*b1e^2*b2e^3*ce^2 + 16*ae^3*b1e^4*ce^3 + 32*ae^3*b1e^3*b2e*ce^3 + 4*ae^3*b1e^2*b2e^2*ce^3 + 16*ae^2*b1e^3*b2e^2*ce^3 + 8*ae^2*b1e^2*b2e^3*ce^3 + 16*ae^3*b1e^3*b2e*ce^2 + 8*ae^3*b1e*b2e^3*ce^2 + 8*ae^3*b1e^4*b2e^-1*ce^3 + 16*ae^3*b1e^2*b2e*ce^3 + 16*ae^2*b1e^3*b2e*ce^3 + 24*ae^3*b1e*b2e^2*ce^3 + 32*ae^3*b2e^3*ce^3 + 8*ae^2*b1e*b2e^3*ce^3 + 16*ae^3*b1e^3*ce^2 + 4*ae^2*b1e^4*ce^2 + 16*ae^3*b1e^2*b2e*ce^2 + 8*ae^2*b1e^2*b2e^2*ce^2 + 24*ae^3*b1e^3*b2e^-1*ce^3 + 4*ae^3*b1e^2*ce^3 + 16*ae^2*b1e^3*ce^3 + 16*ae^3*b1e*b2e*ce^3 + 16*ae^2*b1e^2*b2e*ce^3 + 24*ae^3*b2e^2*ce^3 + 8*ae^3*b1e^3*b2e^-1*ce^2 + 16*ae^3*b1e*b2e*ce^2 + 8*ae^3*b1e^2*b2e^-1*ce^3 + 8*ae^2*b1e^3*b2e^-1*ce^3 + 16*ae^3*b1e*ce^3 + 32*ae^3*b2e*ce^3 + 16*ae^2*b1e*b2e*ce^3 + 8*ae^3*b1e^2*b2e^-1*ce^2 + 16*ae^3*b1e*ce^2 + 8*ae^2*b1e^2*ce^2 + 16*ae^3*b2e*ce^2 + 8*ae^3*b1e^-1*b2e^2*ce^2 + 8*ae^2*b2e^2*ce^2 + 12*ae^3*b1e^2*b2e^-2*ce^3 + 24*ae^3*b1e*b2e^-1*ce^3 + 8*ae^2*b1e^2*b2e^-1*ce^3 + 96*ae^3*ce^3 + 16*ae^2*b1e*ce^3 + 24*ae^3*b1e^-1*b2e*ce^3 + 16*ae^2*b2e*ce^3 + 12*ae^3*b1e^-2*b2e^2*ce^3 + 8*ae^2*b1e^-1*b2e^2*ce^3 + 32*ae^3*b2e^-1*ce^3 + 16*ae^3*b1e^-1*ce^3 + 8*ae^3*b1e^-2*b2e*ce^3 + 8*ae^3*b1e*b2e^-2*ce^2 + 16*ae^3*b2e^-1*ce^2 + 16*ae^3*b1e^-1*ce^2 + 24*ae^2*ce^2 + 8*ae^3*b1e^-2*b2e*ce^2 + 24*ae^3*b2e^-2*ce^3 + 8*ae^2*b1e*b2e^-2*ce^3 + 16*ae^3*b1e^-1*b2e^-1*ce^3 + 16*ae^2*b2e^-1*ce^3 + 4*ae^3*b1e^-2*ce^3 + 16*ae^2*b1e^-1*ce^3 + 24*ae^3*b1e^-3*b2e*ce^3 + 8*ae^2*b1e^-2*b2e*ce^3 + 16*ae^3*b1e^-1*b2e^-1*ce^2 + 8*ae^3*b1e^-3*b2e*ce^2 + 32*ae^3*b2e^-3*ce^3 + 24*ae^3*b1e^-1*b2e^-2*ce^3 + 16*ae^3*b1e^-2*b2e^-1*ce^3 + 16*ae^2*b1e^-1*b2e^-1*ce^3 + 8*ae^3*b1e^-4*b2e*ce^3 + 8*ae^2*b1e^-3*b2e*ce^3 + 8*ae^2*b2e^-2*ce^2 + 16*ae^3*b1e^-2*b2e^-1*ce^2 + 16*ae^3*b1e^-3*ce^2 + 8*ae^2*b1e^-2*ce^2 + 4*ae^3*b1e^-2*b2e^-2*ce^3 + 32*ae^3*b1e^-3*b2e^-1*ce^3 + 16*ae^2*b1e^-2*b2e^-1*ce^3 + 16*ae^3*b1e^-4*ce^3 + 16*ae^2*b1e^-3*ce^3 + 8*ae^3*b1e^-1*b2e^-3*ce^2 + 16*ae^3*b1e^-3*b2e^-1*ce^2 + 8*ae^3*b1e^-2*b2e^-3*ce^3 + 8*ae^2*b1e^-1*b2e^-3*ce^3 + 32*ae^3*b1e^-3*b2e^-2*ce^3 + 24*ae^3*b1e^-4*b2e^-1*ce^3 + 16*ae^2*b1e^-3*b2e^-1*ce^3 + 16*ae^3*b1e^-5*ce^3 + 8*ae^3*b1e^-2*b2e^-3*ce^2 + 16*ae^3*b1e^-3*b2e^-2*ce^2 + 8*ae^2*b1e^-2*b2e^-2*ce^2 + 4*ae^2*b1e^-4*ce^2 + 12*ae^3*b1e^-2*b2e^-4*ce^3 + 8*ae^2*b1e^-2*b2e^-3*ce^3 + 4*ae^3*b1e^-4*b2e^-2*ce^3 + 16*ae^2*b1e^-3*b2e^-2*ce^3 + 8*ae^3*b1e^-5*b2e^-1*ce^3 + 4*ae^3*b1e^-6*ce^3 + 16*ae^3*b1e^-3*b2e^-3*ce^2 + 8*ae^3*b1e^-5*b2e^-1*ce^2 + 24*ae^3*b1e^-3*b2e^-4*ce^3 + 24*ae^3*b1e^-4*b2e^-3*ce^3 + 16*ae^2*b1e^-3*b2e^-3*ce^3 + 24*ae^3*b1e^-5*b2e^-2*ce^3 + 24*ae^3*b1e^-6*b2e^-1*ce^3 + 8*ae^2*b1e^-5*b2e^-1*ce^3 + 8*ae^3*b1e^-3*b2e^-4*ce^2 + 8*ae^2*b1e^-4*b2e^-2*ce^2 + 8*ae^3*b1e^-6*b2e^-1*ce^2 + 16*ae^3*b1e^-4*b2e^-4*ce^3 + 8*ae^2*b1e^-3*b2e^-4*ce^3 + 24*ae^3*b1e^-5*b2e^-3*ce^3 + 24*ae^3*b1e^-6*b2e^-2*ce^3 + 8*ae^3*b1e^-7*b2e^-1*ce^3 + 8*ae^2*b1e^-6*b2e^-1*ce^3 + 8*ae^3*b1e^-4*b2e^-5*ce^3 + 8*ae^3*b1e^-5*b2e^-4*ce^3 + 8*ae^3*b1e^-5*b2e^-4*ce^2 + 4*ae^2*b1e^-4*b2e^-4*ce^2 + 16*ae^3*b1e^-6*b2e^-3*ce^2 + 8*ae^3*b1e^-7*b2e^-2*ce^2 + 8*ae^2*b1e^-6*b2e^-2*ce^2 + 16*ae^3*b1e^-5*b2e^-5*ce^3 + 24*ae^3*b1e^-6*b2e^-4*ce^3 + 8*ae^2*b1e^-5*b2e^-4*ce^3 + 8*ae^3*b1e^-7*b2e^-3*ce^3 + 16*ae^2*b1e^-6*b2e^-3*ce^3 + 12*ae^3*b1e^-8*b2e^-2*ce^3 + 8*ae^2*b1e^-7*b2e^-2*ce^3 + 8*ae^3*b1e^-7*b2e^-3*ce^2 + 24*ae^3*b1e^-6*b2e^-5*ce^3 + 8*ae^3*b1e^-7*b2e^-4*ce^3 + 8*ae^2*b1e^-7*b2e^-3*ce^3 + 8*ae^3*b1e^-6*b2e^-5*ce^2 + 8*ae^3*b1e^-7*b2e^-4*ce^2 + 8*ae^2*b1e^-6*b2e^-4*ce^2 + 8*ae^3*b1e^-8*b2e^-3*ce^2 + 4*ae^3*b1e^-6*b2e^-6*ce^3 + 8*ae^2*b1e^-6*b2e^-5*ce^3 + 16*ae^3*b1e^-8*b2e^-4*ce^3 + 8*ae^2*b1e^-7*b2e^-4*ce^3 + 32*ae^3*b1e^-9*b2e^-3*ce^3 + 8*ae^2*b1e^-8*b2e^-3*ce^3 + 8*ae^3*b1e^-7*b2e^-5*ce^2 + 8*ae^3*b1e^-7*b2e^-6*ce^3 + 8*ae^2*b1e^-7*b2e^-5*ce^3 + 24*ae^3*b1e^-9*b2e^-4*ce^3 + 8*ae^3*b1e^-8*b2e^-5*ce^2 + 8*ae^3*b1e^-9*b2e^-4*ce^2 + 4*ae^2*b1e^-8*b2e^-4*ce^2 + 12*ae^3*b1e^-8*b2e^-6*ce^3 + 24*ae^3*b1e^-9*b2e^-5*ce^3 + 8*ae^2*b1e^-8*b2e^-5*ce^3 + 12*ae^3*b1e^-10*b2e^-4*ce^3 + 8*ae^2*b1e^-9*b2e^-4*ce^3 + 8*ae^3*b1e^-9*b2e^-5*ce^2 + 32*ae^3*b1e^-9*b2e^-6*ce^3 + 16*ae^3*b1e^-10*b2e^-5*ce^3 + 8*ae^2*b1e^-9*b2e^-5*ce^3 + 12*ae^3*b1e^-10*b2e^-6*ce^3 + 8*ae^3*b1e^-11*b2e^-5*ce^3 + 8*ae^3*b1e^-11*b2e^-6*ce^3 + 4*ae^3*b1e^-12*b2e^-6*ce^3
    """
    reduce_b = HermitianModularFormD2Factory_class(HermitianModularFormD2Filter_diagonal(0, D = D))._reduce_vector_valued_index

    exponents = dict()

    filter_indef = HermitianModularFormD2Filter_diagonal_borcherds(a_precision, c_precision, discriminant_bound, D)
    for ((a, b1, b2, c),_,disc) in filter_indef.iter_positive_diagonal_indefinite_forms_with_content_and_discriminant() :
        for n in range(1, min((a_precision - 1) // a, (c_precision - 1) // c) + 1) :
            try :
                exponents[(n*a, n*b1, n*b2, n*c)] += -Integer(coefficients[reduce_b((b1, b2))][disc]) / n
            except KeyError :
                try :
                    exponents[(n*a, n*b1, n*b2, n*c)] = -Integer(coefficients[reduce_b((b1, b2))][disc]) / n
                except KeyError :
                    pass

    
    P = LaurentPolynomialRing(QQ, ['ae', 'b1e', 'b2e', 'ce'])
    truncate = lambda p : P(dict((e,c) for (e,c) in p.dict().iteritems() if e[0] < a_precision and e[3] < c_precision))
    (ae, b1e, b2e, ce) = P.gens()
    
    p = sum(exp * ae**a * b1e**b1 * b2e**b2 * ce**c for ((a,b1,b2,c),exp) in exponents.iteritems())
    
    result = [P(1)]
    for _ in range(1, min(a_precision, c_precision)) :
        result.append( truncate(result[-1] * p) )
    
    return result

#===============================================================================
# indefinite_diagonal_part__diagonal_precision
#===============================================================================

def indefinite_diagonal_part__diagonal_precision(coefficients, precision, discriminant_bound, D) :
    """
    The part `\mathfrak{C}` in [GKR] and its powers. It subsumes all indices `(a,b,c)` that satisfy `c > 0,\,a \le 0`.
       
    INPUT:
        - ``coefficients``       -- A dictionary tuple -> (dictionary Integer -> ring element);
                                    The coefficients separated by components of a vector valued elliptic
                                    modular form.
        - ``precision``          -- A positive integer; An upper bound for `c` in the result's indices.
        - ``discriminant_bound`` -- A negative integer; A lower bound for the coefficients that do
                                    not vanish.
        - `D`                    -- A negative integer; A discriminant of an imaginary quadratic field.
    
    OUTPUT:
        A list of Laurent polynomial.
    
    TESTS::
        sage: indefinite_diagonal_part__diagonal_precision({}, 4, -10, -3)
        [1, 0, 0, 0]
        sage: coefficients = {(1, 0): {-1: 2, -5: -10, -4: -4, -3: 4, -2: 8}, (0, 0): {-1: -8, -5: -6, -4: 6, -3: -6, -2: 1}, (-1, 0): {-1: 2, -5: -10, -4: -4, -3: 4, -2: 8}}
        sage: indefinite_diagonal_part__diagonal_precision(coefficients, 4, -7, -3)[1]
        -4/3*b1e^12*b2e^6*ce^3 - 2*b1e^9*b2e^6*ce^3 - 2*b1e^9*b2e^3*ce^3 - 4/3*b1e^6*b2e^6*ce^3 - 2*b1e^8*b2e^4*ce^2 - 3*b1e^6*b2e^4*ce^2 + 2/3*b1e^6*b2e^3*ce^3 - 3*b1e^6*b2e^2*ce^2 - 2*b1e^4*b2e^4*ce^2 - 4/3*b1e^6*ce^3 - 4*b1e^4*b2e^2*ce^3 + 2/3*b1e^3*b2e^3*ce^3 - 4/3*ae^-3*b1e^6*b2e^3*ce^3 - 3*b1e^4*b2e^2*ce^2 - 6*b1e^3*b2e^2*ce^3 - 4*b1e^4*b2e^2*ce - 6*b1e^3*b2e^2*ce^2 - 6*b1e^3*b2e*ce^3 - 4*b1e^2*b2e^2*ce^3 - 6*b1e^3*b2e^2*ce - 2*b1e^4*ce^2 - 6*b1e^3*b2e*ce^2 - 3*b1e^2*b2e^2*ce^2 - 2*ae^-2*b1e^4*b2e^2*ce^2 + 2/3*b1e^3*ce^3 + 2*b1e^2*b2e*ce^3 - 2*b2e^3*ce^3 - 4/3*ae^-3*b1e^3*b2e^3*ce^3 - 6*b1e^3*b2e*ce - 4*b1e^2*b2e^2*ce + 2*b1e^2*b2e*ce^2 - 4*b1e^2*ce^3 + 2*b1e*b2e*ce^3 + 2*b1e^2*b2e*ce - 3*b1e^2*ce^2 + 2*b1e*b2e*ce^2 - 3*b2e^2*ce^2 - 2*ae^-2*b1e^2*b2e^2*ce^2 + 2*b1e*ce^3 - 6*b2e*ce^3 - 4*b1e^2*ce + 2*b1e*b2e*ce - 4*ae^-1*b1e^2*b2e*ce + 2*b1e*ce^2 - 6*b2e*ce^2 - 4/3*ae^-3*b1e^3*ce^3 + 2*b1e*ce - 6*b2e*ce - 4*ae^-1*b1e*b2e*ce - 2*ae^-2*b1e^2*ce^2 - 6*b2e^-1*ce^3 + 2*b1e^-1*ce^3 - 4*ae^-1*b1e*ce - 6*b2e^-1*ce^2 + 2*b1e^-1*ce^2 + 2*b1e^-1*b2e^-1*ce^3 - 4*b1e^-2*ce^3 - 6*b2e^-1*ce + 2*b1e^-1*ce - 6*ae^-1*ce - 3*b2e^-2*ce^2 + 2*b1e^-1*b2e^-1*ce^2 - 3*b1e^-2*ce^2 - 3*ae^-2*ce^2 - 2*b2e^-3*ce^3 + 2*b1e^-2*b2e^-1*ce^3 + 2/3*b1e^-3*ce^3 - 2*ae^-3*ce^3 + 2*b1e^-1*b2e^-1*ce - 4*b1e^-2*ce - 4*ae^-1*b1e^-1*ce + 2*b1e^-2*b2e^-1*ce^2 - 4*b1e^-2*b2e^-2*ce^3 - 6*b1e^-3*b2e^-1*ce^3 + 2*b1e^-2*b2e^-1*ce - 4*ae^-1*b1e^-1*b2e^-1*ce - 3*b1e^-2*b2e^-2*ce^2 - 6*b1e^-3*b2e^-1*ce^2 - 2*b1e^-4*ce^2 - 2*ae^-2*b1e^-2*ce^2 - 6*b1e^-3*b2e^-2*ce^3 - 4*b1e^-2*b2e^-2*ce - 6*b1e^-3*b2e^-1*ce - 4*ae^-1*b1e^-2*b2e^-1*ce - 6*b1e^-3*b2e^-2*ce^2 + 2/3*b1e^-3*b2e^-3*ce^3 - 4*b1e^-4*b2e^-2*ce^3 - 4/3*b1e^-6*ce^3 - 4/3*ae^-3*b1e^-3*ce^3 - 6*b1e^-3*b2e^-2*ce - 3*b1e^-4*b2e^-2*ce^2 - 2*ae^-2*b1e^-2*b2e^-2*ce^2 - 4*b1e^-4*b2e^-2*ce - 2*b1e^-4*b2e^-4*ce^2 - 3*b1e^-6*b2e^-2*ce^2 - 2*ae^-2*b1e^-4*b2e^-2*ce^2 + 2/3*b1e^-6*b2e^-3*ce^3 - 4/3*ae^-3*b1e^-3*b2e^-3*ce^3 - 3*b1e^-6*b2e^-4*ce^2 - 4/3*b1e^-6*b2e^-6*ce^3 - 2*b1e^-9*b2e^-3*ce^3 - 4/3*ae^-3*b1e^-6*b2e^-3*ce^3 - 2*b1e^-8*b2e^-4*ce^2 - 2*b1e^-9*b2e^-6*ce^3 - 4/3*b1e^-12*b2e^-6*ce^3
        sage: indefinite_diagonal_part__diagonal_precision(coefficients, 4, -7, -3)[2]
        16*b1e^12*b2e^6*ce^3 + 24*b1e^11*b2e^6*ce^3 + 24*b1e^11*b2e^5*ce^3 + 40*b1e^10*b2e^6*ce^3 - 8*b1e^10*b2e^5*ce^3 + 36*b1e^9*b2e^6*ce^3 + 40*b1e^10*b2e^4*ce^3 + 28*b1e^9*b2e^5*ce^3 + 16*ae^-1*b1e^10*b2e^5*ce^3 + 40*b1e^8*b2e^6*ce^3 + 28*b1e^9*b2e^4*ce^3 + 12*b1e^8*b2e^5*ce^3 + 16*ae^-1*b1e^9*b2e^5*ce^3 + 24*b1e^7*b2e^6*ce^3 + 36*b1e^9*b2e^3*ce^3 + 72*b1e^8*b2e^4*ce^3 + 16*ae^-1*b1e^9*b2e^4*ce^3 + 12*b1e^7*b2e^5*ce^3 + 24*ae^-1*b1e^8*b2e^5*ce^3 + 16*b1e^6*b2e^6*ce^3 + 16*b1e^8*b2e^4*ce^2 + 12*b1e^8*b2e^3*ce^3 + 64*b1e^7*b2e^4*ce^3 + 24*ae^-1*b1e^8*b2e^4*ce^3 + 28*b1e^6*b2e^5*ce^3 + 24*ae^-1*b1e^7*b2e^5*ce^3 + 48*b1e^7*b2e^4*ce^2 + 40*b1e^8*b2e^2*ce^3 + 64*b1e^7*b2e^3*ce^3 + 24*ae^-1*b1e^8*b2e^3*ce^3 + 152*b1e^6*b2e^4*ce^3 + 40*ae^-1*b1e^7*b2e^4*ce^3 + 16*ae^-2*b1e^8*b2e^4*ce^3 - 8*b1e^5*b2e^5*ce^3 + 16*ae^-1*b1e^6*b2e^5*ce^3 + 48*b1e^7*b2e^3*ce^2 + 68*b1e^6*b2e^4*ce^2 + 12*b1e^7*b2e^2*ce^3 + 180*b1e^6*b2e^3*ce^3 + 40*ae^-1*b1e^7*b2e^3*ce^3 + 64*b1e^5*b2e^4*ce^3 + 36*ae^-1*b1e^6*b2e^4*ce^3 + 24*ae^-2*b1e^7*b2e^4*ce^3 + 24*b1e^4*b2e^5*ce^3 + 16*ae^-1*b1e^5*b2e^5*ce^3 + 56*b1e^6*b2e^3*ce^2 + 48*b1e^5*b2e^4*ce^2 + 24*b1e^7*b2e*ce^3 + 152*b1e^6*b2e^2*ce^3 + 24*ae^-1*b1e^7*b2e^2*ce^3 + 20*b1e^5*b2e^3*ce^3 + 40*ae^-1*b1e^6*b2e^3*ce^3 + 24*ae^-2*b1e^7*b2e^3*ce^3 + 72*b1e^4*b2e^4*ce^3 + 40*ae^-1*b1e^5*b2e^4*ce^3 + 32*ae^-2*b1e^6*b2e^4*ce^3 + 68*b1e^6*b2e^2*ce^2 + 8*b1e^5*b2e^3*ce^2 + 32*ae^-1*b1e^6*b2e^3*ce^2 + 16*b1e^4*b2e^4*ce^2 + 28*b1e^6*b2e*ce^3 + 20*b1e^5*b2e^2*ce^3 + 36*ae^-1*b1e^6*b2e^2*ce^3 + 20*b1e^4*b2e^3*ce^3 + 96*ae^-1*b1e^5*b2e^3*ce^3 - 8*ae^-2*b1e^6*b2e^3*ce^3 + 28*b1e^3*b2e^4*ce^3 + 24*ae^-1*b1e^4*b2e^4*ce^3 + 24*ae^-2*b1e^5*b2e^4*ce^3 + 8*b1e^5*b2e^2*ce^2 + 8*b1e^4*b2e^3*ce^2 + 80*ae^-1*b1e^5*b2e^3*ce^2 + 16*b1e^6*ce^3 + 64*b1e^5*b2e*ce^3 + 16*ae^-1*b1e^6*b2e*ce^3 + 24*b1e^4*b2e^2*ce^3 + 96*ae^-1*b1e^5*b2e^2*ce^3 + 32*ae^-2*b1e^6*b2e^2*ce^3 + 180*b1e^3*b2e^3*ce^3 + 96*ae^-1*b1e^4*b2e^3*ce^3 + 16*ae^-2*b1e^5*b2e^3*ce^3 + 16*ae^-3*b1e^6*b2e^3*ce^3 + 40*b1e^2*b2e^4*ce^3 + 16*ae^-1*b1e^3*b2e^4*ce^3 + 16*ae^-2*b1e^4*b2e^4*ce^3 + 48*b1e^5*b2e*ce^2 - 12*b1e^4*b2e^2*ce^2 + 80*ae^-1*b1e^5*b2e^2*ce^2 + 56*b1e^3*b2e^3*ce^2 + 80*ae^-1*b1e^4*b2e^3*ce^2 - 8*b1e^5*ce^3 + 20*b1e^4*b2e*ce^3 + 40*ae^-1*b1e^5*b2e*ce^3 + 140*b1e^3*b2e^2*ce^3 + 116*ae^-1*b1e^4*b2e^2*ce^3 + 16*ae^-2*b1e^5*b2e^2*ce^3 + 64*b1e^2*b2e^3*ce^3 + 40*ae^-1*b1e^3*b2e^3*ce^3 + 16*ae^-2*b1e^4*b2e^3*ce^3 + 16*ae^-3*b1e^5*b2e^3*ce^3 + 8*b1e^4*b2e*ce^2 + 48*b1e^3*b2e^2*ce^2 + 128*ae^-1*b1e^4*b2e^2*ce^2 + 48*b1e^2*b2e^3*ce^2 + 32*ae^-1*b1e^3*b2e^3*ce^2 + 72*b1e^4*ce^3 + 16*ae^-1*b1e^5*ce^3 + 140*b1e^3*b2e*ce^3 + 96*ae^-1*b1e^4*b2e*ce^3 + 24*ae^-2*b1e^5*b2e*ce^3 + 24*b1e^2*b2e^2*ce^3 + 88*ae^-1*b1e^3*b2e^2*ce^3 + 56*ae^-2*b1e^4*b2e^2*ce^3 + 16*ae^-3*b1e^5*b2e^2*ce^3 + 12*b1e*b2e^3*ce^3 + 40*ae^-1*b1e^2*b2e^3*ce^3 - 8*ae^-2*b1e^3*b2e^3*ce^3 + 16*ae^-3*b1e^4*b2e^3*ce^3 + 16*b1e^4*ce^2 + 48*b1e^3*b2e*ce^2 + 80*ae^-1*b1e^4*b2e*ce^2 - 12*b1e^2*b2e^2*ce^2 + 104*ae^-1*b1e^3*b2e^2*ce^2 + 16*ae^-2*b1e^4*b2e^2*ce^2 + 24*b1e^4*b2e^-1*ce^3 + 180*b1e^3*ce^3 + 24*ae^-1*b1e^4*ce^3 + 60*b1e^2*b2e*ce^3 + 88*ae^-1*b1e^3*b2e*ce^3 + 16*ae^-2*b1e^4*b2e*ce^3 + 20*b1e*b2e^2*ce^3 + 116*ae^-1*b1e^2*b2e^2*ce^3 + 20*ae^-2*b1e^3*b2e^2*ce^3 + 24*ae^-3*b1e^4*b2e^2*ce^3 + 36*b2e^3*ce^3 + 24*ae^-1*b1e*b2e^3*ce^3 + 24*ae^-2*b1e^2*b2e^3*ce^3 + 16*ae^-3*b1e^3*b2e^3*ce^3 + 56*b1e^3*ce^2 + 40*b1e^2*b2e*ce^2 + 104*ae^-1*b1e^3*b2e*ce^2 + 8*b1e*b2e^2*ce^2 + 128*ae^-1*b1e^2*b2e^2*ce^2 + 32*ae^-2*b1e^3*b2e^2*ce^2 + 28*b1e^3*b2e^-1*ce^3 + 24*b1e^2*ce^3 + 40*ae^-1*b1e^3*ce^3 + 16*ae^-2*b1e^4*ce^3 + 60*b1e*b2e*ce^3 + 64*ae^-1*b1e^2*b2e*ce^3 + 20*ae^-2*b1e^3*b2e*ce^3 + 16*ae^-3*b1e^4*b2e*ce^3 + 152*b2e^2*ce^3 + 96*ae^-1*b1e*b2e^2*ce^3 + 56*ae^-2*b1e^2*b2e^2*ce^3 + 32*ae^-3*b1e^3*b2e^2*ce^3 - 12*b1e^2*ce^2 + 32*ae^-1*b1e^3*ce^2 + 40*b1e*b2e*ce^2 + 72*ae^-1*b1e^2*b2e*ce^2 + 32*ae^-2*b1e^3*b2e*ce^2 + 68*b2e^2*ce^2 + 80*ae^-1*b1e*b2e^2*ce^2 + 16*ae^-2*b1e^2*b2e^2*ce^2 + 64*b1e^2*b2e^-1*ce^3 + 16*ae^-1*b1e^3*b2e^-1*ce^3 + 60*b1e*ce^3 + 116*ae^-1*b1e^2*ce^3 - 8*ae^-2*b1e^3*ce^3 + 140*b2e*ce^3 + 64*ae^-1*b1e*b2e*ce^3 + 28*ae^-2*b1e^2*b2e*ce^3 + 32*ae^-3*b1e^3*b2e*ce^3 + 12*b1e^-1*b2e^2*ce^3 + 36*ae^-1*b2e^2*ce^3 + 16*ae^-2*b1e*b2e^2*ce^3 + 24*ae^-3*b1e^2*b2e^2*ce^3 + 48*b1e^2*b2e^-1*ce^2 + 40*b1e*ce^2 + 128*ae^-1*b1e^2*ce^2 + 48*b2e*ce^2 + 72*ae^-1*b1e*b2e*ce^2 + 80*ae^-2*b1e^2*b2e*ce^2 + 40*b1e^2*b2e^-2*ce^3 + 20*b1e*b2e^-1*ce^3 + 40*ae^-1*b1e^2*b2e^-1*ce^3 + 624*ce^3 + 64*ae^-1*b1e*ce^3 + 56*ae^-2*b1e^2*ce^3 + 16*ae^-3*b1e^3*ce^3 + 20*b1e^-1*b2e*ce^3 + 88*ae^-1*b2e*ce^3 + 28*ae^-2*b1e*b2e*ce^3 + 40*ae^-3*b1e^2*b2e*ce^3 + 40*b1e^-2*b2e^2*ce^3 + 24*ae^-1*b1e^-1*b2e^2*ce^3 + 32*ae^-2*b2e^2*ce^3 + 16*ae^-3*b1e*b2e^2*ce^3 + 8*b1e*b2e^-1*ce^2 + 336*ce^2 + 72*ae^-1*b1e*ce^2 + 16*ae^-2*b1e^2*ce^2 + 8*b1e^-1*b2e*ce^2 + 104*ae^-1*b2e*ce^2 + 80*ae^-2*b1e*b2e*ce^2 + 12*b1e*b2e^-2*ce^3 + 140*b2e^-1*ce^3 + 96*ae^-1*b1e*b2e^-1*ce^3 + 24*ae^-2*b1e^2*b2e^-1*ce^3 + 60*b1e^-1*ce^3 - 96*ae^-1*ce^3 + 28*ae^-2*b1e*ce^3 + 24*ae^-3*b1e^2*ce^3 + 64*b1e^-2*b2e*ce^3 + 96*ae^-1*b1e^-1*b2e*ce^3 + 20*ae^-2*b2e*ce^3 + 40*ae^-3*b1e*b2e*ce^3 + 48*b2e^-1*ce^2 + 80*ae^-1*b1e*b2e^-1*ce^2 + 40*b1e^-1*ce^2 - 96*ae^-1*ce^2 + 80*ae^-2*b1e*ce^2 + 48*b1e^-2*b2e*ce^2 + 80*ae^-1*b1e^-1*b2e*ce^2 + 32*ae^-2*b2e*ce^2 + 152*b2e^-2*ce^3 + 24*ae^-1*b1e*b2e^-2*ce^3 + 60*b1e^-1*b2e^-1*ce^3 + 88*ae^-1*b2e^-1*ce^3 + 16*ae^-2*b1e*b2e^-1*ce^3 + 24*b1e^-2*ce^3 + 64*ae^-1*b1e^-1*ce^3 + 96*ae^-2*ce^3 + 40*ae^-3*b1e*ce^3 + 28*b1e^-3*b2e*ce^3 + 40*ae^-1*b1e^-2*b2e*ce^3 + 16*ae^-2*b1e^-1*b2e*ce^3 + 32*ae^-3*b2e*ce^3 + 68*b2e^-2*ce^2 + 40*b1e^-1*b2e^-1*ce^2 + 104*ae^-1*b2e^-1*ce^2 - 12*b1e^-2*ce^2 + 72*ae^-1*b1e^-1*ce^2 + 132*ae^-2*ce^2 + 36*b2e^-3*ce^3 + 20*b1e^-1*b2e^-2*ce^3 + 36*ae^-1*b2e^-2*ce^3 + 60*b1e^-2*b2e^-1*ce^3 + 64*ae^-1*b1e^-1*b2e^-1*ce^3 + 20*ae^-2*b2e^-1*ce^3 + 16*ae^-3*b1e*b2e^-1*ce^3 + 180*b1e^-3*ce^3 + 116*ae^-1*b1e^-2*ce^3 + 28*ae^-2*b1e^-1*ce^3 + 36*ae^-3*ce^3 + 24*b1e^-4*b2e*ce^3 + 16*ae^-1*b1e^-3*b2e*ce^3 + 24*ae^-2*b1e^-2*b2e*ce^3 + 16*ae^-3*b1e^-1*b2e*ce^3 + 8*b1e^-1*b2e^-2*ce^2 + 40*b1e^-2*b2e^-1*ce^2 + 72*ae^-1*b1e^-1*b2e^-1*ce^2 + 32*ae^-2*b2e^-1*ce^2 + 56*b1e^-3*ce^2 + 128*ae^-1*b1e^-2*ce^2 + 80*ae^-2*b1e^-1*ce^2 + 12*b1e^-1*b2e^-3*ce^3 + 24*b1e^-2*b2e^-2*ce^3 + 96*ae^-1*b1e^-1*b2e^-2*ce^3 + 32*ae^-2*b2e^-2*ce^3 + 140*b1e^-3*b2e^-1*ce^3 + 64*ae^-1*b1e^-2*b2e^-1*ce^3 + 28*ae^-2*b1e^-1*b2e^-1*ce^3 + 32*ae^-3*b2e^-1*ce^3 + 72*b1e^-4*ce^3 + 40*ae^-1*b1e^-3*ce^3 + 56*ae^-2*b1e^-2*ce^3 + 40*ae^-3*b1e^-1*ce^3 - 12*b1e^-2*b2e^-2*ce^2 + 80*ae^-1*b1e^-1*b2e^-2*ce^2 + 48*b1e^-3*b2e^-1*ce^2 + 72*ae^-1*b1e^-2*b2e^-1*ce^2 + 80*ae^-2*b1e^-1*b2e^-1*ce^2 + 16*b1e^-4*ce^2 + 32*ae^-1*b1e^-3*ce^2 + 16*ae^-2*b1e^-2*ce^2 + 64*b1e^-2*b2e^-3*ce^3 + 24*ae^-1*b1e^-1*b2e^-3*ce^3 + 140*b1e^-3*b2e^-2*ce^3 + 116*ae^-1*b1e^-2*b2e^-2*ce^3 + 16*ae^-2*b1e^-1*b2e^-2*ce^3 + 20*b1e^-4*b2e^-1*ce^3 + 88*ae^-1*b1e^-3*b2e^-1*ce^3 + 28*ae^-2*b1e^-2*b2e^-1*ce^3 + 40*ae^-3*b1e^-1*b2e^-1*ce^3 - 8*b1e^-5*ce^3 + 24*ae^-1*b1e^-4*ce^3 - 8*ae^-2*b1e^-3*ce^3 + 24*ae^-3*b1e^-2*ce^3 + 48*b1e^-2*b2e^-3*ce^2 + 48*b1e^-3*b2e^-2*ce^2 + 128*ae^-1*b1e^-2*b2e^-2*ce^2 + 8*b1e^-4*b2e^-1*ce^2 + 104*ae^-1*b1e^-3*b2e^-1*ce^2 + 80*ae^-2*b1e^-2*b2e^-1*ce^2 + 40*b1e^-2*b2e^-4*ce^3 + 180*b1e^-3*b2e^-3*ce^3 + 40*ae^-1*b1e^-2*b2e^-3*ce^3 + 24*b1e^-4*b2e^-2*ce^3 + 88*ae^-1*b1e^-3*b2e^-2*ce^3 + 56*ae^-2*b1e^-2*b2e^-2*ce^3 + 16*ae^-3*b1e^-1*b2e^-2*ce^3 + 64*b1e^-5*b2e^-1*ce^3 + 96*ae^-1*b1e^-4*b2e^-1*ce^3 + 20*ae^-2*b1e^-3*b2e^-1*ce^3 + 40*ae^-3*b1e^-2*b2e^-1*ce^3 + 16*b1e^-6*ce^3 + 16*ae^-1*b1e^-5*ce^3 + 16*ae^-2*b1e^-4*ce^3 + 16*ae^-3*b1e^-3*ce^3 + 56*b1e^-3*b2e^-3*ce^2 - 12*b1e^-4*b2e^-2*ce^2 + 104*ae^-1*b1e^-3*b2e^-2*ce^2 + 16*ae^-2*b1e^-2*b2e^-2*ce^2 + 48*b1e^-5*b2e^-1*ce^2 + 80*ae^-1*b1e^-4*b2e^-1*ce^2 + 32*ae^-2*b1e^-3*b2e^-1*ce^2 + 28*b1e^-3*b2e^-4*ce^3 + 20*b1e^-4*b2e^-3*ce^3 + 40*ae^-1*b1e^-3*b2e^-3*ce^3 + 24*ae^-2*b1e^-2*b2e^-3*ce^3 + 20*b1e^-5*b2e^-2*ce^3 + 116*ae^-1*b1e^-4*b2e^-2*ce^3 + 20*ae^-2*b1e^-3*b2e^-2*ce^3 + 24*ae^-3*b1e^-2*b2e^-2*ce^3 + 28*b1e^-6*b2e^-1*ce^3 + 40*ae^-1*b1e^-5*b2e^-1*ce^3 + 16*ae^-2*b1e^-4*b2e^-1*ce^3 + 32*ae^-3*b1e^-3*b2e^-1*ce^3 + 8*b1e^-4*b2e^-3*ce^2 + 32*ae^-1*b1e^-3*b2e^-3*ce^2 + 8*b1e^-5*b2e^-2*ce^2 + 128*ae^-1*b1e^-4*b2e^-2*ce^2 + 32*ae^-2*b1e^-3*b2e^-2*ce^2 + 72*b1e^-4*b2e^-4*ce^3 + 16*ae^-1*b1e^-3*b2e^-4*ce^3 + 20*b1e^-5*b2e^-3*ce^3 + 96*ae^-1*b1e^-4*b2e^-3*ce^3 - 8*ae^-2*b1e^-3*b2e^-3*ce^3 + 152*b1e^-6*b2e^-2*ce^3 + 96*ae^-1*b1e^-5*b2e^-2*ce^3 + 56*ae^-2*b1e^-4*b2e^-2*ce^3 + 32*ae^-3*b1e^-3*b2e^-2*ce^3 + 24*b1e^-7*b2e^-1*ce^3 + 16*ae^-1*b1e^-6*b2e^-1*ce^3 + 24*ae^-2*b1e^-5*b2e^-1*ce^3 + 16*ae^-3*b1e^-4*b2e^-1*ce^3 + 16*b1e^-4*b2e^-4*ce^2 + 8*b1e^-5*b2e^-3*ce^2 + 80*ae^-1*b1e^-4*b2e^-3*ce^2 + 68*b1e^-6*b2e^-2*ce^2 + 80*ae^-1*b1e^-5*b2e^-2*ce^2 + 16*ae^-2*b1e^-4*b2e^-2*ce^2 + 24*b1e^-4*b2e^-5*ce^3 + 64*b1e^-5*b2e^-4*ce^3 + 24*ae^-1*b1e^-4*b2e^-4*ce^3 + 180*b1e^-6*b2e^-3*ce^3 + 96*ae^-1*b1e^-5*b2e^-3*ce^3 + 16*ae^-2*b1e^-4*b2e^-3*ce^3 + 16*ae^-3*b1e^-3*b2e^-3*ce^3 + 12*b1e^-7*b2e^-2*ce^3 + 36*ae^-1*b1e^-6*b2e^-2*ce^3 + 16*ae^-2*b1e^-5*b2e^-2*ce^3 + 24*ae^-3*b1e^-4*b2e^-2*ce^3 + 48*b1e^-5*b2e^-4*ce^2 + 56*b1e^-6*b2e^-3*ce^2 + 80*ae^-1*b1e^-5*b2e^-3*ce^2 - 8*b1e^-5*b2e^-5*ce^3 + 152*b1e^-6*b2e^-4*ce^3 + 40*ae^-1*b1e^-5*b2e^-4*ce^3 + 16*ae^-2*b1e^-4*b2e^-4*ce^3 + 64*b1e^-7*b2e^-3*ce^3 + 40*ae^-1*b1e^-6*b2e^-3*ce^3 + 16*ae^-2*b1e^-5*b2e^-3*ce^3 + 16*ae^-3*b1e^-4*b2e^-3*ce^3 + 40*b1e^-8*b2e^-2*ce^3 + 24*ae^-1*b1e^-7*b2e^-2*ce^3 + 32*ae^-2*b1e^-6*b2e^-2*ce^3 + 16*ae^-3*b1e^-5*b2e^-2*ce^3 + 68*b1e^-6*b2e^-4*ce^2 + 48*b1e^-7*b2e^-3*ce^2 + 32*ae^-1*b1e^-6*b2e^-3*ce^2 + 28*b1e^-6*b2e^-5*ce^3 + 16*ae^-1*b1e^-5*b2e^-5*ce^3 + 64*b1e^-7*b2e^-4*ce^3 + 36*ae^-1*b1e^-6*b2e^-4*ce^3 + 24*ae^-2*b1e^-5*b2e^-4*ce^3 + 12*b1e^-8*b2e^-3*ce^3 + 40*ae^-1*b1e^-7*b2e^-3*ce^3 - 8*ae^-2*b1e^-6*b2e^-3*ce^3 + 16*ae^-3*b1e^-5*b2e^-3*ce^3 + 48*b1e^-7*b2e^-4*ce^2 + 16*b1e^-6*b2e^-6*ce^3 + 12*b1e^-7*b2e^-5*ce^3 + 16*ae^-1*b1e^-6*b2e^-5*ce^3 + 72*b1e^-8*b2e^-4*ce^3 + 40*ae^-1*b1e^-7*b2e^-4*ce^3 + 32*ae^-2*b1e^-6*b2e^-4*ce^3 + 36*b1e^-9*b2e^-3*ce^3 + 24*ae^-1*b1e^-8*b2e^-3*ce^3 + 24*ae^-2*b1e^-7*b2e^-3*ce^3 + 16*ae^-3*b1e^-6*b2e^-3*ce^3 + 16*b1e^-8*b2e^-4*ce^2 + 24*b1e^-7*b2e^-6*ce^3 + 12*b1e^-8*b2e^-5*ce^3 + 24*ae^-1*b1e^-7*b2e^-5*ce^3 + 28*b1e^-9*b2e^-4*ce^3 + 24*ae^-1*b1e^-8*b2e^-4*ce^3 + 24*ae^-2*b1e^-7*b2e^-4*ce^3 + 40*b1e^-8*b2e^-6*ce^3 + 28*b1e^-9*b2e^-5*ce^3 + 24*ae^-1*b1e^-8*b2e^-5*ce^3 + 40*b1e^-10*b2e^-4*ce^3 + 16*ae^-1*b1e^-9*b2e^-4*ce^3 + 16*ae^-2*b1e^-8*b2e^-4*ce^3 + 36*b1e^-9*b2e^-6*ce^3 - 8*b1e^-10*b2e^-5*ce^3 + 16*ae^-1*b1e^-9*b2e^-5*ce^3 + 40*b1e^-10*b2e^-6*ce^3 + 24*b1e^-11*b2e^-5*ce^3 + 16*ae^-1*b1e^-10*b2e^-5*ce^3 + 24*b1e^-11*b2e^-6*ce^3 + 16*b1e^-12*b2e^-6*ce^3
    """
    reduce_b = HermitianModularFormD2Factory_class(HermitianModularFormD2Filter_diagonal(0, D = D))._reduce_vector_valued_index

    exponents = dict()

    filter_indef = HermitianModularFormD2Filter_diagonal_borcherds(precision, d = discriminant_bound, D = D)
    for ((a, b1, b2, c),_,disc) in filter_indef.iter_indefinite_diagonal_indefinite_forms_with_content_and_discriminant() :
        for n in range(1, (precision - 1) // c + 1) : 
            try :
                exponents[(n*a, n*b1, n*b2, n*c)] += -Integer(coefficients[reduce_b((b1, b2))][disc]) / n
            except KeyError :
                try :
                    exponents[(n*a, n*b1, n*b2, n*c)] = -Integer(coefficients[reduce_b((b1, b2))][disc]) / n
                except KeyError :
                    pass
    
    P = LaurentPolynomialRing(QQ, ['ae', 'b1e', 'b2e', 'ce'])
    truncate = lambda p : P(dict((e,c) for (e,c) in p.dict().iteritems() if e[3] < precision))
    (ae, b1e, b2e, ce) = P.gens()
    
    p = sum(exp * ae**a * b1e**b1 * b2e**b2 * ce**c for ((a,b1,b2,c),exp) in exponents.iteritems())
    
    result = [P(1)]
    for _ in range(1, precision) :
        result.append( truncate(result[-1] * p) )

    return result 

#===============================================================================
# semidefinite_diagonal_part__diagonal_precision
#===============================================================================

def semidefinite_diagonal_part__diagonal_precision(coefficients, precision, discriminant_bound, D) :
    """
    The part `\mathfrak{D}` in [GKR] and its powers. It subsumes all indices `(a,b,c)` that satisfy `c = 0,\,a > 0`.
       
    INPUT:
        - ``coefficients``       -- A dictionary tuple -> (dictionary Integer -> ring element);
                                    The coefficients separated by components of a vector valued elliptic
                                    modular form.
        - ``precision``          -- A positive integer; An upper bound for `c` in the result's indices.
        - ``discriminant_bound`` -- A negative integer; A lower bound for the coefficients that do
                                    not vanish.
        - `D`                    -- A negative integer; A discriminant of an imaginary quadratic field.
    
    OUTPUT:
        A list of Laurent polynomial.
    
    TESTS::
        sage: semidefinite_diagonal_part__diagonal_precision({}, 4, -10, -3)
        [1, 0, 0, 0]
        sage: coefficients = {(1, 0): {0: -1, 1: 8, 2: -4, 3: 4, 4: -7, 5: -3, 6: 7, 7: 1, 8: -9, 9: 4, -2: 10, -1: -5}, (0, 0): {0: 2, 1: 6, 2: 10, 3: -9, 4: -8, 5: 6, 6: -2, 7: -2, 8: -7, 9: -8, -2: -8, -1: 8}, (-1, 0): {0: -1, 1: 8, 2: -4, 3: 4, 4: -7, 5: -3, 6: 7, 7: 1, 8: -9, 9: 4, -2: 10, -1: -5}}
        sage: semidefinite_diagonal_part__diagonal_precision(coefficients, 4, -3, -3)[1]
        -5/3*ae^3*b1e^6*b2e^3 - 5/3*ae^3*b1e^3*b2e^3 - 5/2*ae^2*b1e^4*b2e^2 - 5/3*ae^3*b1e^3 - 5*ae^3*b1e^2*b2e - 5/2*ae^2*b1e^2*b2e^2 - 5*ae^3*b1e*b2e - 5*ae^2*b1e^2*b2e - 5*ae^3*b1e - 5/2*ae^2*b1e^2 - 5*ae^2*b1e*b2e - 5*ae*b1e^2*b2e + 8/3*ae^3 - 5*ae^2*b1e - 5*ae*b1e*b2e - 5*ae^3*b1e^-1 + 3*ae^2 - 5*ae*b1e - 5*ae^3*b1e^-1*b2e^-1 - 5*ae^2*b1e^-1 + 2*ae - 5*ae^3*b1e^-2*b2e^-1 - 5*ae^2*b1e^-1*b2e^-1 - 5/3*ae^3*b1e^-3 - 5/2*ae^2*b1e^-2 - 5*ae*b1e^-1 - 5*ae^2*b1e^-2*b2e^-1 - 5*ae*b1e^-1*b2e^-1 - 5/2*ae^2*b1e^-2*b2e^-2 - 5*ae*b1e^-2*b2e^-1 - 5/3*ae^3*b1e^-3*b2e^-3 - 5/2*ae^2*b1e^-4*b2e^-2 - 5/3*ae^3*b1e^-6*b2e^-3
        sage: coefficients = {(1, 0): {-1: 2, -5: -10, -4: -4, -3: 4, -2: 8}, (0, 0): {-1: -8, -5: -6, -4: 6, -3: -6, -2: 1}, (-1, 0): {-1: 2, -5: -10, -4: -4, -3: 4, -2: 8}}
        sage: semidefinite_diagonal_part__diagonal_precision(coefficients, 4, -7, -3)[1]
        -4/3*ae^3*b1e^12*b2e^6 - 2*ae^3*b1e^9*b2e^6 - 2*ae^3*b1e^9*b2e^3 - 4/3*ae^3*b1e^6*b2e^6 - 2*ae^2*b1e^8*b2e^4 + 2/3*ae^3*b1e^6*b2e^3 - 3*ae^2*b1e^6*b2e^4 - 3*ae^2*b1e^6*b2e^2 - 2*ae^2*b1e^4*b2e^4 - 4/3*ae^3*b1e^6 - 4*ae^3*b1e^4*b2e^2 + 2/3*ae^3*b1e^3*b2e^3 - 6*ae^3*b1e^3*b2e^2 - 3*ae^2*b1e^4*b2e^2 - 6*ae^3*b1e^3*b2e - 4*ae^3*b1e^2*b2e^2 - 6*ae^2*b1e^3*b2e^2 - 4*ae*b1e^4*b2e^2 + 2/3*ae^3*b1e^3 - 2*ae^2*b1e^4 + 2*ae^3*b1e^2*b2e - 6*ae^2*b1e^3*b2e - 3*ae^2*b1e^2*b2e^2 - 6*ae*b1e^3*b2e^2 - 2*ae^3*b2e^3 - 4*ae^3*b1e^2 + 2*ae^3*b1e*b2e + 2*ae^2*b1e^2*b2e - 6*ae*b1e^3*b2e - 4*ae*b1e^2*b2e^2 + 2*ae^3*b1e - 3*ae^2*b1e^2 - 6*ae^3*b2e + 2*ae^2*b1e*b2e + 2*ae*b1e^2*b2e - 3*ae^2*b2e^2 + 2*ae^2*b1e - 4*ae*b1e^2 - 6*ae^2*b2e + 2*ae*b1e*b2e - 6*ae^3*b2e^-1 + 2*ae^3*b1e^-1 + 2*ae*b1e - 6*ae*b2e + 2*ae^3*b1e^-1*b2e^-1 - 6*ae^2*b2e^-1 - 4*ae^3*b1e^-2 + 2*ae^2*b1e^-1 - 2*ae^3*b2e^-3 - 3*ae^2*b2e^-2 + 2*ae^3*b1e^-2*b2e^-1 + 2*ae^2*b1e^-1*b2e^-1 - 6*ae*b2e^-1 + 2/3*ae^3*b1e^-3 - 3*ae^2*b1e^-2 + 2*ae*b1e^-1 - 4*ae^3*b1e^-2*b2e^-2 - 6*ae^3*b1e^-3*b2e^-1 + 2*ae^2*b1e^-2*b2e^-1 + 2*ae*b1e^-1*b2e^-1 - 4*ae*b1e^-2 - 6*ae^3*b1e^-3*b2e^-2 - 3*ae^2*b1e^-2*b2e^-2 - 6*ae^2*b1e^-3*b2e^-1 + 2*ae*b1e^-2*b2e^-1 - 2*ae^2*b1e^-4 + 2/3*ae^3*b1e^-3*b2e^-3 - 4*ae^3*b1e^-4*b2e^-2 - 6*ae^2*b1e^-3*b2e^-2 - 4*ae*b1e^-2*b2e^-2 - 6*ae*b1e^-3*b2e^-1 - 4/3*ae^3*b1e^-6 - 3*ae^2*b1e^-4*b2e^-2 - 6*ae*b1e^-3*b2e^-2 - 4*ae*b1e^-4*b2e^-2 - 2*ae^2*b1e^-4*b2e^-4 + 2/3*ae^3*b1e^-6*b2e^-3 - 3*ae^2*b1e^-6*b2e^-2 - 3*ae^2*b1e^-6*b2e^-4 - 4/3*ae^3*b1e^-6*b2e^-6 - 2*ae^3*b1e^-9*b2e^-3 - 2*ae^2*b1e^-8*b2e^-4 - 2*ae^3*b1e^-9*b2e^-6 - 4/3*ae^3*b1e^-12*b2e^-6
        sage: semidefinite_diagonal_part__diagonal_precision(coefficients, 4, -7, -3)[2]
        16*ae^3*b1e^12*b2e^6 + 24*ae^3*b1e^11*b2e^6 + 24*ae^3*b1e^11*b2e^5 + 40*ae^3*b1e^10*b2e^6 - 8*ae^3*b1e^10*b2e^5 + 36*ae^3*b1e^9*b2e^6 + 40*ae^3*b1e^10*b2e^4 + 28*ae^3*b1e^9*b2e^5 + 40*ae^3*b1e^8*b2e^6 + 28*ae^3*b1e^9*b2e^4 + 12*ae^3*b1e^8*b2e^5 + 24*ae^3*b1e^7*b2e^6 + 36*ae^3*b1e^9*b2e^3 + 72*ae^3*b1e^8*b2e^4 + 12*ae^3*b1e^7*b2e^5 + 16*ae^3*b1e^6*b2e^6 + 12*ae^3*b1e^8*b2e^3 + 64*ae^3*b1e^7*b2e^4 + 16*ae^2*b1e^8*b2e^4 + 28*ae^3*b1e^6*b2e^5 + 40*ae^3*b1e^8*b2e^2 + 64*ae^3*b1e^7*b2e^3 + 152*ae^3*b1e^6*b2e^4 + 48*ae^2*b1e^7*b2e^4 - 8*ae^3*b1e^5*b2e^5 + 12*ae^3*b1e^7*b2e^2 + 180*ae^3*b1e^6*b2e^3 + 48*ae^2*b1e^7*b2e^3 + 64*ae^3*b1e^5*b2e^4 + 68*ae^2*b1e^6*b2e^4 + 24*ae^3*b1e^4*b2e^5 + 24*ae^3*b1e^7*b2e + 152*ae^3*b1e^6*b2e^2 + 20*ae^3*b1e^5*b2e^3 + 56*ae^2*b1e^6*b2e^3 + 72*ae^3*b1e^4*b2e^4 + 48*ae^2*b1e^5*b2e^4 + 28*ae^3*b1e^6*b2e + 20*ae^3*b1e^5*b2e^2 + 68*ae^2*b1e^6*b2e^2 + 20*ae^3*b1e^4*b2e^3 + 8*ae^2*b1e^5*b2e^3 + 28*ae^3*b1e^3*b2e^4 + 16*ae^2*b1e^4*b2e^4 + 16*ae^3*b1e^6 + 64*ae^3*b1e^5*b2e + 24*ae^3*b1e^4*b2e^2 + 8*ae^2*b1e^5*b2e^2 + 180*ae^3*b1e^3*b2e^3 + 8*ae^2*b1e^4*b2e^3 + 40*ae^3*b1e^2*b2e^4 - 8*ae^3*b1e^5 + 20*ae^3*b1e^4*b2e + 48*ae^2*b1e^5*b2e + 140*ae^3*b1e^3*b2e^2 - 12*ae^2*b1e^4*b2e^2 + 64*ae^3*b1e^2*b2e^3 + 56*ae^2*b1e^3*b2e^3 + 72*ae^3*b1e^4 + 140*ae^3*b1e^3*b2e + 8*ae^2*b1e^4*b2e + 24*ae^3*b1e^2*b2e^2 + 48*ae^2*b1e^3*b2e^2 + 12*ae^3*b1e*b2e^3 + 48*ae^2*b1e^2*b2e^3 + 24*ae^3*b1e^4*b2e^-1 + 180*ae^3*b1e^3 + 16*ae^2*b1e^4 + 60*ae^3*b1e^2*b2e + 48*ae^2*b1e^3*b2e + 20*ae^3*b1e*b2e^2 - 12*ae^2*b1e^2*b2e^2 + 36*ae^3*b2e^3 + 28*ae^3*b1e^3*b2e^-1 + 24*ae^3*b1e^2 + 56*ae^2*b1e^3 + 60*ae^3*b1e*b2e + 40*ae^2*b1e^2*b2e + 152*ae^3*b2e^2 + 8*ae^2*b1e*b2e^2 + 64*ae^3*b1e^2*b2e^-1 + 60*ae^3*b1e - 12*ae^2*b1e^2 + 140*ae^3*b2e + 40*ae^2*b1e*b2e + 12*ae^3*b1e^-1*b2e^2 + 68*ae^2*b2e^2 + 40*ae^3*b1e^2*b2e^-2 + 20*ae^3*b1e*b2e^-1 + 48*ae^2*b1e^2*b2e^-1 + 624*ae^3 + 40*ae^2*b1e + 20*ae^3*b1e^-1*b2e + 48*ae^2*b2e + 40*ae^3*b1e^-2*b2e^2 + 12*ae^3*b1e*b2e^-2 + 140*ae^3*b2e^-1 + 8*ae^2*b1e*b2e^-1 + 60*ae^3*b1e^-1 + 336*ae^2 + 64*ae^3*b1e^-2*b2e + 8*ae^2*b1e^-1*b2e + 152*ae^3*b2e^-2 + 60*ae^3*b1e^-1*b2e^-1 + 48*ae^2*b2e^-1 + 24*ae^3*b1e^-2 + 40*ae^2*b1e^-1 + 28*ae^3*b1e^-3*b2e + 48*ae^2*b1e^-2*b2e + 36*ae^3*b2e^-3 + 20*ae^3*b1e^-1*b2e^-2 + 68*ae^2*b2e^-2 + 60*ae^3*b1e^-2*b2e^-1 + 40*ae^2*b1e^-1*b2e^-1 + 180*ae^3*b1e^-3 - 12*ae^2*b1e^-2 + 24*ae^3*b1e^-4*b2e + 12*ae^3*b1e^-1*b2e^-3 + 24*ae^3*b1e^-2*b2e^-2 + 8*ae^2*b1e^-1*b2e^-2 + 140*ae^3*b1e^-3*b2e^-1 + 40*ae^2*b1e^-2*b2e^-1 + 72*ae^3*b1e^-4 + 56*ae^2*b1e^-3 + 64*ae^3*b1e^-2*b2e^-3 + 140*ae^3*b1e^-3*b2e^-2 - 12*ae^2*b1e^-2*b2e^-2 + 20*ae^3*b1e^-4*b2e^-1 + 48*ae^2*b1e^-3*b2e^-1 - 8*ae^3*b1e^-5 + 16*ae^2*b1e^-4 + 40*ae^3*b1e^-2*b2e^-4 + 180*ae^3*b1e^-3*b2e^-3 + 48*ae^2*b1e^-2*b2e^-3 + 24*ae^3*b1e^-4*b2e^-2 + 48*ae^2*b1e^-3*b2e^-2 + 64*ae^3*b1e^-5*b2e^-1 + 8*ae^2*b1e^-4*b2e^-1 + 16*ae^3*b1e^-6 + 28*ae^3*b1e^-3*b2e^-4 + 20*ae^3*b1e^-4*b2e^-3 + 56*ae^2*b1e^-3*b2e^-3 + 20*ae^3*b1e^-5*b2e^-2 - 12*ae^2*b1e^-4*b2e^-2 + 28*ae^3*b1e^-6*b2e^-1 + 48*ae^2*b1e^-5*b2e^-1 + 72*ae^3*b1e^-4*b2e^-4 + 20*ae^3*b1e^-5*b2e^-3 + 8*ae^2*b1e^-4*b2e^-3 + 152*ae^3*b1e^-6*b2e^-2 + 8*ae^2*b1e^-5*b2e^-2 + 24*ae^3*b1e^-7*b2e^-1 + 24*ae^3*b1e^-4*b2e^-5 + 64*ae^3*b1e^-5*b2e^-4 + 16*ae^2*b1e^-4*b2e^-4 + 180*ae^3*b1e^-6*b2e^-3 + 8*ae^2*b1e^-5*b2e^-3 + 12*ae^3*b1e^-7*b2e^-2 + 68*ae^2*b1e^-6*b2e^-2 - 8*ae^3*b1e^-5*b2e^-5 + 152*ae^3*b1e^-6*b2e^-4 + 48*ae^2*b1e^-5*b2e^-4 + 64*ae^3*b1e^-7*b2e^-3 + 56*ae^2*b1e^-6*b2e^-3 + 40*ae^3*b1e^-8*b2e^-2 + 28*ae^3*b1e^-6*b2e^-5 + 64*ae^3*b1e^-7*b2e^-4 + 68*ae^2*b1e^-6*b2e^-4 + 12*ae^3*b1e^-8*b2e^-3 + 48*ae^2*b1e^-7*b2e^-3 + 16*ae^3*b1e^-6*b2e^-6 + 12*ae^3*b1e^-7*b2e^-5 + 72*ae^3*b1e^-8*b2e^-4 + 48*ae^2*b1e^-7*b2e^-4 + 36*ae^3*b1e^-9*b2e^-3 + 24*ae^3*b1e^-7*b2e^-6 + 12*ae^3*b1e^-8*b2e^-5 + 28*ae^3*b1e^-9*b2e^-4 + 16*ae^2*b1e^-8*b2e^-4 + 40*ae^3*b1e^-8*b2e^-6 + 28*ae^3*b1e^-9*b2e^-5 + 40*ae^3*b1e^-10*b2e^-4 + 36*ae^3*b1e^-9*b2e^-6 - 8*ae^3*b1e^-10*b2e^-5 + 40*ae^3*b1e^-10*b2e^-6 + 24*ae^3*b1e^-11*b2e^-5 + 24*ae^3*b1e^-11*b2e^-6 + 16*ae^3*b1e^-12*b2e^-6
    """
    reduce_b = HermitianModularFormD2Factory_class(HermitianModularFormD2Filter_diagonal(0, D = D))._reduce_vector_valued_index

    exponents = dict()
    
    filter_indef = HermitianModularFormD2Filter_diagonal_borcherds(precision, d = discriminant_bound, D = D)
    for ((a, b1, b2, _),_,disc) in filter_indef.iter_semidefinite_diagonal_indefinite_forms_with_content_and_discriminant() :
        for n in range(1, (precision - 1) // a + 1) :
            try :
                exponents[(n*a, n*b1, n*b2)] += -Integer(coefficients[reduce_b((b1, b2))][disc]) / n
            except KeyError :
                try :
                    exponents[(n*a, n*b1, n*b2)] = -Integer(coefficients[reduce_b((b1, b2))][disc]) / n
                except KeyError :
                    pass


    P = LaurentPolynomialRing(QQ, ['ae', 'b1e', 'b2e', 'ce'])
    truncate = lambda p : P(dict((e,c) for (e,c) in p.dict().iteritems() if e[0] < precision))
    (ae, b1e, b2e, _) = P.gens()
    
    p = sum(exp * ae**a * b1e**b1 * b2e**b2 for ((a,b1,b2),exp) in exponents.iteritems())
    
    result = [P(1)]
    for _ in range(1, precision) :
        result.append( truncate(result[-1] * p) )
    
    return result
    
#===============================================================================
# zero_diagonal_negative_b2_part__b2_precision
#===============================================================================

def zero_diagonal_negative_b2_part__b2_precision(coefficients, precision, discriminant_bound, D) :
    """
    The part `\mathfrak{E}` in [GKR] restricted to all `b` with negative second component and its powers.
    I.e the indices `(a,b,c)` that we take into account satisfy `a = c = 0\, b_2 < 0`.
    
        INPUT:
        - ``coefficients``       -- A dictionary tuple -> (dictionary Integer -> ring element);
                                    The coefficients separated by components of a vector valued elliptic
                                    modular form.
        - ``precision``          -- A positive integer; An upper bound for `-b_2` in the result's indices.
        - ``discriminant_bound`` -- A negative integer; A lower bound for the coefficients that do
                                    not vanish.
        - `D`                    -- A negative integer; A discriminant of an imaginary quadratic field.
    
    OUTPUT:
        A list of Laurent polynomial.
    
    TESTS::
        sage: zero_diagonal_negative_b2_part__b2_precision({}, 4, -10, -3)
        [1, 0, 0, 0]
        sage: coefficients = {(1, 0): {0: -1, 1: 8, 2: -4, 3: 4, 4: -7, 5: -3, 6: 7, 7: 1, 8: -9, 9: 4, -2: 10, -1: -5}, (0, 0): {0: 2, 1: 6, 2: 10, 3: -9, 4: -8, 5: 6, 6: -2, 7: -2, 8: -7, 9: -8, -2: -8, -1: 8}, (-1, 0): {0: -1, 1: 8, 2: -4, 3: 4, 4: -7, 5: -3, 6: 7, 7: 1, 8: -9, 9: 4, -2: 10, -1: -5}}
        sage: zero_diagonal_negative_b2_part__b2_precision(coefficients, 4, -3, -3)[1]
        -5*b1e^-1*b2e^-1 - 5*b1e^-2*b2e^-1 - 5/2*b1e^-2*b2e^-2 - 5/3*b1e^-3*b2e^-3 - 5/2*b1e^-4*b2e^-2 - 5/3*b1e^-6*b2e^-3
        sage: coefficients = {(1, 0): {-1: 2, -5: -10, -4: -4, -3: 4, -2: 8}, (0, 0): {-1: -8, -5: -6, -4: 6, -3: -6, -2: 1}, (-1, 0): {-1: 2, -5: -10, -4: -4, -3: 4, -2: 8}}
        sage: zero_diagonal_negative_b2_part__b2_precision(coefficients, 4, -7, -3)[1]
        -6*b2e^-1 - 3*b2e^-2 + 2*b1e^-1*b2e^-1 - 2*b2e^-3 + 2*b1e^-2*b2e^-1 - 3*b1e^-2*b2e^-2 - 6*b1e^-3*b2e^-1 - 6*b1e^-3*b2e^-2 + 2/3*b1e^-3*b2e^-3 - 3*b1e^-4*b2e^-2 - 3*b1e^-6*b2e^-2 + 2/3*b1e^-6*b2e^-3 - 2*b1e^-9*b2e^-3
        sage: zero_diagonal_negative_b2_part__b2_precision(coefficients, 4, -7, -3)[2]
        36*b2e^-2 + 36*b2e^-3 - 24*b1e^-1*b2e^-2 - 12*b1e^-1*b2e^-3 - 20*b1e^-2*b2e^-2 + 24*b1e^-2*b2e^-3 + 80*b1e^-3*b2e^-2 + 96*b1e^-3*b2e^-3 - 20*b1e^-4*b2e^-2 - 24*b1e^-5*b2e^-2 + 36*b1e^-6*b2e^-2 + 96*b1e^-6*b2e^-3 + 24*b1e^-7*b2e^-3 - 12*b1e^-8*b2e^-3 + 36*b1e^-9*b2e^-3
    """
    reduce_b = HermitianModularFormD2Factory_class(HermitianModularFormD2Filter_diagonal(0, D = D))._reduce_vector_valued_index

    exponents = dict()
    filter_indef = HermitianModularFormD2Filter_diagonal_borcherds(precision, d = discriminant_bound, D = D)
    for ((_, b1, b2, _),_,disc) in filter_indef.iter_zero_diagonal_indefinite_forms_with_content_and_discriminant() :
        if b2 >= 0 : continue
        for n in range(1, (precision - 1) // (-b2) + 1) : 
            try :
                exponents[(n*b1, n*b2)] += -Integer(coefficients[reduce_b((b1, b2))][disc]) / n
            except KeyError :
                try :
                    exponents[(n*b1, n*b2)] = -Integer(coefficients[reduce_b((b1, b2))][disc]) / n
                except KeyError :
                    pass

#------------------------------------------------------------------------------ 
    P = LaurentPolynomialRing(QQ, ['ae', 'b1e', 'b2e', 'ce'])
    truncate = lambda p : P(dict((e,c) for (e,c) in p.dict().iteritems() if e[2] > -precision))
    (_, b1e, b2e, _) = P.gens()

    p = sum(e * b1e**b1 * b2e**b2 for ((b1, b2), e) in exponents.iteritems())
    
    result = [P(1)]
    for _ in range(1, precision) :
        result.append( truncate(result[-1] * p) )

    return result

#===============================================================================
# zero_diagonal_zero_b2_part__b1_precision
#===============================================================================

def zero_diagonal_zero_b2_part__b1_precision(coefficients, precision, discriminant_bound, D) :
    """
    The part `\mathfrak{E}` in [GKR] restricted to all `b` with vanishing second component and its powers.
    I.e the indices `(a,b,c)` that we take into account satisfy `a = c = b_2 = 0\, b_1 < 0`.
    
        INPUT:
        - ``coefficients``       -- A dictionary tuple -> (dictionary Integer -> ring element);
                                    The coefficients separated by components of a vector valued elliptic
                                    modular form.
        - ``precision``          -- A positive integer; An upper bound for `-b_1` in the result's indices.
        - ``discriminant_bound`` -- A negative integer; A lower bound for the coefficients that do
                                    not vanish.
        - `D`                    -- A negative integer; A discriminant of an imaginary quadratic field.
    
    OUTPUT:
        A list of Laurent polynomial.
    
    TESTS::
        sage: zero_diagonal_zero_b2_part__b1_precision({}, 4, -10, -3)
        [1, 0, 0, 0]
        sage: coefficients = {(1, 0): {0: -1, 1: 8, 2: -4, 3: 4, 4: -7, 5: -3, 6: 7, 7: 1, 8: -9, 9: 4, -2: 10, -1: -5}, (0, 0): {0: 2, 1: 6, 2: 10, 3: -9, 4: -8, 5: 6, 6: -2, 7: -2, 8: -7, 9: -8, -2: -8, -1: 8}, (-1, 0): {0: -1, 1: 8, 2: -4, 3: 4, 4: -7, 5: -3, 6: 7, 7: 1, 8: -9, 9: 4, -2: 10, -1: -5}}
        sage: zero_diagonal_zero_b2_part__b1_precision(coefficients, 4, -3, -3)[1]
        -5*b1e^-1 - 5/2*b1e^-2 - 5/3*b1e^-3
        sage: coefficients = {(1, 0): {-1: 2, -5: -10, -4: -4, -3: 4, -2: 8}, (0, 0): {-1: -8, -5: -6, -4: 6, -3: -6, -2: 1}, (-1, 0): {-1: 2, -5: -10, -4: -4, -3: 4, -2: 8}}
        sage: zero_diagonal_zero_b2_part__b1_precision(coefficients, 4, -7, -3)[1]
        2*b1e^-1 - 3*b1e^-2 + 2/3*b1e^-3
        sage: zero_diagonal_zero_b2_part__b1_precision(coefficients, 4, -7, -3)[2]
        4*b1e^-2 - 12*b1e^-3
    """
    reduce_b = HermitianModularFormD2Factory_class(HermitianModularFormD2Filter_diagonal(0, D = D))._reduce_vector_valued_index

    exponents = dict()
    
    filter_indef = HermitianModularFormD2Filter_diagonal_borcherds(precision, d = discriminant_bound, D = D)
    for ((_, b1, b2, _),_,disc) in filter_indef.iter_zero_diagonal_indefinite_forms_with_content_and_discriminant() :
        if b2 != 0 or b1 >= 0 : continue
        
        for n in range(1, (precision - 1) // (-b1) + 1) : 
            try :
                exponents[n*b1] += -Integer(coefficients[reduce_b((b1, b2))][disc]) / n
            except KeyError :
                try :
                    exponents[n*b1] = -Integer(coefficients[reduce_b((b1, b2))][disc]) / n
                except KeyError :
                    pass
            

    P = LaurentPolynomialRing(QQ, ['ae', 'b1e', 'b2e', 'ce'])
    truncate = lambda p : P(dict((e,c) for (e,c) in p.dict().iteritems() if e[1] > -precision))
    (_, b1e, _, _) = P.gens()

    p = sum(e * b1e**b1 for (b1, e) in exponents.iteritems())
    
    result = [P(1)]
    for _ in range(1, precision) :
        result.append( truncate(result[-1] * p) )

    return result

def lattice_for_hermitian_dual_maximal_order(D) :
    """
    Return the Gram matrix that is associated to the trace form of the the Dedeking module
    in an hermitian field.

    INPUT:
    
    - `D` -- A negative integer; The discriminant of a hermitian field.

    OUTPUT:
    
    - A quadratic form `Q` that assignes to each element of `\mathcal{o}_{\QQ(D)}`
      its discriminant. The quadratic form is formed with respect to the
      basis `1 / \sqrt{D},\, (1 + \sqrt{D})/2`.

    NOTE:

        We use the calculations done by Dern p. 106.
        
    TESTS::

        sage: from hermitianmodularforms.hermitianmodularformd2_borcherdsproducts import *
        sage: lattice_for_hermitian_dual_maximal_order(-3)
        Quadratic form in 2 variables over Integer Ring with coefficients: 
        [ 1 -3 ]
        [ * 3 ]

        sage: lattice_for_hermitian_dual_maximal_order(-4)
        Quadratic form in 2 variables over Integer Ring with coefficients: 
        [ 1 -4 ]
        [ * 5 ]

    """
    
    return QuadraticForm(matrix(2, [2, D, D, (D**2 - D)//2]))

def weyl_vector(coefficients, discriminant_bound, D) :
    """
    We use Dern's formula, but in his notation m and n are interchanged.

    INPUT:
    
    - ``coefficients``       -- A dictionary (theta1, theta2) -> (ZZ -> ZZ) storing all
                                coefficients of a vector valued modular form by index in
                                the discriminant group and discrimnant.
                                definite indices.
        
    - ``discriminant_bound`` -- A positive integer; The negative of a lower bound on the discriminants
                                that occur in dictionary ``coefficients``.
    
    - `D`                    -- A negative integer; The discriminant of a hermitian field.
    
    OUTPUT:
    
    - A tuple `(a, b1, b2, c)`.

    REFERENCE:

    - Dern, Hermitsche Modulformen zweiten Grades, PhD thesis, RWTH Aachen University

        
    TESTS:
        sage: from hermitianmodularforms.hermitianmodularformd2_borcherdsproducts import weyl_vector
        sage: weyl_vector({(0,0) : {-3: 1, 0: 90}, (1,0): {}, (-1,0): {}}, -4, -3)
        (4, 3, 2, 3)
    """
    L = lattice_for_hermitian_dual_maximal_order(D)
    L_vecs = list(enumerate(L.short_vector_list_up_to_length(D * discriminant_bound)))
    
    reduce_b = HermitianModularFormD2Factory_class(HermitianModularFormD2Filter_diagonal(0, D = D))._reduce_vector_valued_index

    a_weyl = 0
    c_weyl = 0
    b1_weyl = 0
    b2_weyl = 0
    for (l, vs) in L_vecs :
        for v in vs :
            vr = reduce_b(v)
            try :
                a_weyl += Rational((1,24)) * coefficients[vr][-l]
                if v[1] < 0 or (v[1] == 0 and v[0] < 0) : 
                    b1_weyl += -Rational((1,2)) * coefficients[vr][-l] * v[0]
                    b2_weyl += -Rational((1,2)) * coefficients[vr][-l] * v[1]
            except KeyError :
                pass
            
            for n in range(1, (-discriminant_bound - 1 - l) // (-D) + 1) :
                try :
                    c_weyl += sigma(n, 1) * coefficients[vr][D * n - l]
                except KeyError :
                    pass
    
    c_weyl = a_weyl - c_weyl

    return tuple(map(ZZ, (a_weyl, b1_weyl, b2_weyl, c_weyl)))

#===============================================================================
# borcherds_product__by_logarithm
#===============================================================================

def borcherds_product__by_logarithm(coefficients, precision) :
    """
    Compute the Borcherds products with the algorithm described in [GKR].

    INPUT:

    - ``coefficients``       -- A dictionary tuple -> (dictionary Integer -> ring element);
                                The coefficients separated by components of a vector valued elliptic
                                modular form.
    
    - ``precision``          -- A filter instance.
    
    OUTPUT:
    
    - An equivariant monoid power series.
    
    EXAMPLES::

        from hermitianmodularforms.hermitianmodularformd2_borcherdsproducts import *
        from hermitianmodularforms.hermitianmodularformd2_fourierexpansion import *
        sage: vv = {(0,0):{-3:1,0:90,3:100116},(1,0):{-1:0,2:16038,5:2125035},(-1,0):{-1:0,2:16038,5:2125035}}
        sage: index_filter = HermitianModularFormD2Filter_diagonal(5,-3)
        sage: borcherds_product__by_logarithm(vv, index_filter).coefficients()
        {f1: {(3, 3, 2, 4): -1}, 1: {(0, 0, 0, 0): 0}, f0*f1: {}}
    """
    ## We first extract some basic data from the precision
    D = precision.D()
    discriminant_bound = min([0] + [d for cdict in coefficients.values() for d in cdict.keys() if d < 0]) - 1
    
    ## Compute the Weyl vector
    (weyl_a, weyl_b1, weyl_b2, weyl_c) = weyl_vector(coefficients, discriminant_bound, D)
    
    ## We compute three parts `\frakA`, `\frakB`, and `\frakC` of the logarithm of the Borcherds product.
    ## See the paper for more details.
    
    ## The bound precision.index() has to be imposed on the component `c` of any Fourier index in the
    ## Borcherds products is bounded above by `precision.index() - weyl_c`.
    
    ## The negative entry a that occur in indefinite_diagonal need to be taken into account when
    ## computing the bounds for the first two factors of the Borcherds product. Let
    ## `l [[a, b], [\bar b, c]]` be such a indefinite_diagonal form, where the matrix without the
    ## factor `l` satisfies `-D a c + D ||b||^2 > discriminant_bound`.
    ## Since we have `lc < prec - weyl_c` by the bound on `c` we deduce the following bound
    ## on the absolute value of negative entries `l a` in such matrix.
    maxa_negative_contribution = (-discriminant_bound * ( precision.index() - weyl_c - 1 )) // (-D)
    
    definite_positive_diagonal = definite_positive_diagonal_part__diagonal_precision(coefficients,
                                     ceil(precision.index() - weyl_a + maxa_negative_contribution),
                                     ceil(precision.index() - weyl_c), D)
    indefinite_positive_diagonal = indefinite_positive_diagonal_part__diagonal_precision(coefficients,
                                     precision.index() - weyl_a + maxa_negative_contribution,
                                     precision.index() - weyl_c,
                                     discriminant_bound, D)
    indefinite_diagonal = indefinite_diagonal_part__diagonal_precision(coefficients,
                            precision.index() - weyl_c, discriminant_bound, D)
    
    P = LaurentPolynomialRing(QQ, ['ae', 'b1e', 'b2e', 'ce'])
    
    ## The last step of the implementation will be to multiply the approbriate powers of
    ## `\frakA` with the ``poly_factors``, that will contain all other factors `\frakB` to `\frakE`.
    ## Multiplication of `\frakB` and `\frakC`. Since the component `c` is always positive, we
    ## can use it to bound the enumeration.
    poly_factors = dict( (alpha, P(0)) for alpha in range(precision.index() - weyl_c) )
    for alpha in range(precision.index() - weyl_c) :
        for beta in range(precision.index() - weyl_c - alpha) :
            for gamma in range(precision.index() - weyl_c - alpha - beta) :
                poly_factors[alpha] +=   (factorial(alpha) * factorial(beta) * factorial(gamma))**-1 \
                                       * indefinite_positive_diagonal[beta] * indefinite_diagonal[gamma]

    ## We filter all monomials exceeding the bound on `a` and `c`.
    ## The bound on c comes from the bound `precision.index()` on the Fourier indices in the Borcherds products.
    ## In contrast to the computation of `definite_positive_diagonal` and `indefinte_positive_diagonal`, the
    ## according bound for `a` need not be increased, since indices with negative diagonal
    ## have already been multiplied with the polynomial.
    for alpha in poly_factors.keys() :
        poly_factors[alpha] = \
            P(dict( filter( lambda ((a, b1, b2, c), _): a < precision.index() - weyl_a - alpha and c < precision.index() - weyl_c - alpha,
                            poly_factors[alpha].dict().iteritems() ) ))
        if poly_factors[alpha].is_zero() :
            del poly_factors[alpha]
    if len(poly_factors) == 0 :
        return EquivariantMonoidPowerSeries( definite_positive_diagonal[0].parent(), dict(), precision )
    
    ## The exponent of `a` tells us the maximal power of `\frakD`, which features only Fourier indices
    ## with positive component `a`. We determine the potentially negative minimal value of `a` that
    ## occurs in the monomials of `poly_factors`. From this we can compute an upper bound for the entry
    ## `a` that can be added without exceeding the bound `precision.index() - weyl_a`. 
    min_a_exponents = dict( (alpha, min(a for (a,_,_,_) in p.exponents()))
                             for (alpha, p) in poly_factors.iteritems() )
    semidefinite_diagonal_part = semidefinite_diagonal_part__diagonal_precision(coefficients,
                                   ceil(precision.index() - weyl_a - min(min_a_exponents.values())) ,
                                   discriminant_bound, D)
    
    ## Multiply the elements of `poly_factors` with `\frakD`.
    for alpha in poly_factors.keys() :
        poly_factors[alpha] *= \
            sum( factorial(delta)**-1 * semidefinite_diagonal_part[delta]
                 for delta in range(precision.index() - weyl_a - min_a_exponents[alpha]) ) 
    
    ## We filter all monomials exceeding the bound `precision.index() - weyl_a` on `a`.
    for alpha in poly_factors.keys() :
        poly_factors[alpha] = \
            P(dict( filter( lambda ((a, b1, b2, c), _): a < precision.index() - weyl_a - alpha,
                            poly_factors[alpha].dict().iteritems() ) ))
        if poly_factors[alpha].is_zero() :
            del poly_factors[alpha]
    if len(poly_factors) == 0 :
        return EquivariantMonoidPowerSeries( definite_positive_diagonal[0].parent(), dict(), precision )

    ## We treat the factor `\frakE` in more detail. We split it in a part with `b2 < 0` and `b2 = 0`.
    ## This way we can better controle the number of multiplications performed.
    ## The maximal `b2`-entry that occurs in elements of `poly_factors` gives rise to a lower bound on the
    ## `b2`-entry of elements of `\frakE` that we need to consider. We deduce it by considering the discriminant
    ## of a potential, final index `[[a, b], [\bar b, c]]` of the Borcherds product. We have upper bounds on
    ## the entries `a` and `c` of this final index, and from this we deduce a lower bound for `b2` of this final index.
    ## By using upper bounds on the `b2` entries of the indices in `\frakA` and `poly_factors` we optain the
    ## final bound that we use.
    amin_b2max_cmin = dict( (alpha, ( min(a for (a, _, _, _) in p.exponents()), max(b2 for (_, _, b2, _) in p.exponents()),
                                      min(c for (_, _, _, c) in p.exponents()) ))
                            for (alpha, p) in poly_factors.iteritems() )
    ## contributions to the bound on b2:
    ## first line: contribution from `poly_factors` + contribution from weyl vector 
    ## second line: contribution from `\frakA`
    ## third line: bound on `b2` coming from positive definiteness of the final index
    max_zero_diagonal_power = dict( (alpha, b2max + weyl_b2
                                             + isqrt(4 * (precision.index() - 1 - weyl_a - amin) * (precision.index() - 1 - weyl_c - cmin))
                                             + 2 * (precision.index() - 1))
                                    for (alpha, (amin, b2max, cmin)) in amin_b2max_cmin.iteritems() ) 
    zero_diagonal_negative_b2_part = zero_diagonal_negative_b2_part__b2_precision( coefficients,
                                        max(max_zero_diagonal_power.values()) + 1,
                                        discriminant_bound, D )
    
    ## Multiply the elements of `poly_factors` with the first part of `\frakE`.
    for alpha in poly_factors.keys() :
        poly_factors[alpha] *= sum( factorial(eta1)**-1 * zero_diagonal_negative_b2_part[eta1]
                                    for eta1 in range(max_zero_diagonal_power[alpha] + 1) ) 
    ## We filter all monomials exceeding the `b2` bounds for positive definiteness of the final index.
    ## The inequality which is tested is composed of the currect value of `b2`, the contribution of the
    ## Weyl vector, the maximal positive contribution of `\frakA` and, on the right hand side, the minimal
    ## value of `b2` for any final index.  
    min_b2 = - 2 * (precision.index() - 1)
    for alpha in poly_factors.keys() :
        poly_factors[alpha] = \
            P(dict( filter( lambda ((a, b1, b2, c), _): b2 + weyl_b2 + isqrt(4 * (precision.index() - 1 - weyl_a - a) * (precision.index() - 1 - weyl_c - c)) >= min_b2,
                            poly_factors[alpha].dict().iteritems() ) ))
        if poly_factors[alpha].is_zero() :
            del poly_factors[alpha]
    if len(poly_factors) == 0 :
        return EquivariantMonoidPowerSeries( definite_positive_diagonal[0].parent(), dict(), precision )

    ## We deal with the part of `\frakE` that satisfies `b2 = 0`. To deduce a bound on `b1` we need
    ## to consider the contributions from the currect index, the Weyl vector, and `\frakA`. Notice
    ## that the `b1` entry of a positive definite index is bounded by `-(1 + \sqrt{-D}) \sqrt{a c}` from
    ## below. It is bounded by the negative from above.
    amin_b1max_cmin = dict( (alpha, (min(a for (a, _, _, _) in p.exponents()), max(b1 for (_, _, b1, _) in p.exponents()),
                                     min(c for (_, _, _, c) in p.exponents())))
                            for (alpha, p) in poly_factors.iteritems() )
    ## contributions to the bound on b1:
    ## first line: contribution from `poly_factors` + contribution from Weyl vector 
    ## second and third line: contribution from `\frakA`
    ## fourth line: bound on `b1` coming from positive definiteness of the final index
    max_zero_diagonal_power = dict( (alpha, b1max + weyl_b1
                                            + ( isqrt(-D) - D + 1 ) *
                                              ( isqrt((precision.index() - 1 - weyl_a - amin) * (precision.index() - 1 - weyl_c - cmin)) + 1 )
                                            + ( isqrt(-D) - D + 1 ) * (precision.index() - 1))
                                    for (alpha, (amin, b1max, cmin)) in amin_b1max_cmin.iteritems() )
    zero_diagonal_zero_b2_part = zero_diagonal_zero_b2_part__b1_precision( coefficients,
                                   max(max_zero_diagonal_power.values()) + 1,
                                   discriminant_bound, D )
    
    for alpha in poly_factors.keys() :
        poly_factors[alpha] *= sum( factorial(eta2)**-1 * zero_diagonal_zero_b2_part[eta2]
                                    for eta2 in range(max_zero_diagonal_power[alpha] + 1) )
    
    ## We filter all monomials exceeding the `b1` bounds for positive definiteness of the final index.
    
    ## The inequality which is tested is composed of the currect value of `b1`, the contribution of the
    ## Weyl vector, the maximal positive contribution of `\frakA` and, on the right hand side, the minimal
    ## value of `b1` for any final index.  
    min_b1 = - ((isqrt(-D) - D + 1) * (precision.index() - 1) + 1)
    for alpha in poly_factors.keys() :
        poly_factors[alpha] = \
            P(dict( filter( lambda ((a, b1, b2, c), _): b1 + weyl_b1 + ( isqrt(-D) - D + 1 ) * ( isqrt((precision.index() - 1 - weyl_a - a) * (precision.index() - 1 - weyl_c - c)) + 1 ) >= min_b1,
                            poly_factors[alpha].dict().iteritems() ) ))
        if poly_factors[alpha].is_zero() :
            del poly_factors[alpha]
    if len(poly_factors) == 0 :
        return EquivariantMonoidPowerSeries( definite_positive_diagonal[0].parent(), dict(), precision )

    ## We prepare the polynomials in `poly_factors` for multiplication with `\frakA`.
    weyl_p = prod(map(operator.pow, P.gens(), (weyl_a, weyl_b1, weyl_b2, weyl_c)))
    for alpha in poly_factors.keys() :
        poly_factors[alpha] *= weyl_p

    ## determine the weight of the Borcherds product
    reduce_index = HermitianModularFormD2Factory_class(HermitianModularFormD2Filter_diagonal(0, D = D))._reduce_vector_valued_index
    weight = coefficients[reduce_index((0,0))][0]/2
    
    ## We multiply `poly_factors` with `\frakA`.
    res = definite_positive_diagonal[0].parent()(0)
    for alpha in poly_factors :
        res += multiply_polynomial_with_definite_positive_diagonal_part__diagonal_precision(
                               definite_positive_diagonal[alpha], poly_factors[alpha].dict(), precision, weight )

    res._set_precision(precision)
    
    return res

#===============================================================================
# multiply_polynomial_with_definite_positive_diagonal_part__diagonal_precision
#===============================================================================

def multiply_polynomial_with_definite_positive_diagonal_part__diagonal_precision(positive_expansion, other_expansion, precision, weight_parity) :
    """
    Multiply a equiavariant expansion with a polynomial. It is assumed that the result is equivariant.
    
    INPUT:

    - ``positive_expansion`` -- An equivariant monoid power series with exactly one
                                nonvanishing character component.

    - ``other_expansion``    -- A dictionary (ae, b1e, b2e, ce) -> ZZ.

    - ``precision``          -- A filter instance.

    - ``weight_parity``      -- An integer (default: `0`); The parity of the
                                associated weight
    
    OUTPUT:

    - An equivariant monoid power series.
    
    TESTS::

        sage: from hermitianmodularforms.hermitianmodularformd2_borcherdsproducts import *
        sage: from hermitianmodularforms.hermitianmodularformd2_fourierexpansion import *
        sage: from psage.modform.fourier_expansion_framework.monoidpowerseries.monoidpowerseries_element import *                                                   
        sage: from psage.modform.fourier_expansion_framework.monoidpowerseries.monoidpowerseries_ring import *                                                      
        sage: from psage.modform.fourier_expansion_framework.monoidpowerseries.monoidpowerseries_basicmonoids import *                                              
        sage: fering = HermitianModularFormD2FourierExpansionRing(QQ, -3)
        sage: chsym = HermitianModularFormD2FourierExpansionTrivialCharacter(-4, ZZ)
        sage: chantisym = HermitianModularFormD2FourierExpansionTransposeCharacter(-4, ZZ)
        sage: pe = EquivariantMonoidPowerSeries_algebraelement(fering, {chsym: {(1,0,0,1): 1, (2, 3, 2, 3): 4}}, fering.action().filter(5), False, False)
        sage: oe = {(1, 0, 0, 1): 1}
        sage: multiply_polynomial_with_definite_positive_diagonal_part__diagonal_precision(pe, oe, pe.precision(), 0).coefficients()
        {1: {(3, 3, 2, 4): 4, (2, 3, 2, 3): 1, (4, 6, 4, 4): 4, (3, 3, 3, 3): 1, (2, 0, 0, 2): 1}, f0: {}}
    """
    sym_result = dict()
    antisym_result = dict()
    pmonoid = positive_expansion.parent().monoid()


    for oexp in other_expansion :
        (oa, ob1, ob2, oc) = oexp
        for ind in precision :
            (a, b1, b2, c) = ind
            pexp = (a - oa, b1 - ob1, b2 - ob2, c - oc)
            ptrexp = (a - oa, - b1 - precision.D() * b2 - ob1, b2 - ob2, c - oc)
            
            if pexp in pmonoid and pexp in precision :
                pres = other_expansion[oexp] * positive_expansion[pexp]
            else :
                pres = 0
            if ptrexp in pmonoid and ptrexp in precision :
                ptrres = other_expansion[oexp] * positive_expansion[ptrexp]
            else :
                ptrres = 0
                
            if pres != 0 or ptrres != 0 :
                try :
                    sym_result[ind] += (pres + ptrres) / 2
                except KeyError :
                    sym_result[ind] = (pres + ptrres) / 2
    
                try :
                    antisym_result[ind] += (pres - ptrres) / 2
                except KeyError :
                    antisym_result[ind] = (pres - ptrres) / 2

    for k in sym_result.keys() :
        if sym_result[k].is_zero() :
            del sym_result[k]
    for k in antisym_result.keys() :
        if antisym_result[k].is_zero() :
            del antisym_result[k]

    sym_character = HermitianModularFormD2FourierExpansionTrivialCharacter(precision.D(), ZZ, weight_parity)
    antisym_character = HermitianModularFormD2FourierExpansionTransposeCharacter(precision.D(), ZZ, weight_parity)
                    
    return EquivariantMonoidPowerSeries( positive_expansion.parent(),
                                          {sym_character: sym_result, antisym_character: antisym_result},
                                          precision )
