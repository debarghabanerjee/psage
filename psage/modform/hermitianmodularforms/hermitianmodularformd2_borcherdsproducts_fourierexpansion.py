"""
Functions for treating the fourier expansion of hermitian modular forms. 

AUTHOR :
    -- Martin Raum (2009 - 08 - 31) Initial version
    -- Dominic Gehre (2010 - 05 - 12) Change iterations
"""

#===============================================================================
# 
# Copyright (C) 2009 Dominic Gehre, Martin Raum
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

from psage.modform.fourier_expansion_framework.monoidpowerseries.monoidpowerseries_basicmonoids import TrivialCharacterMonoid,\
    TrivialRepresentation
from psage.modform.fourier_expansion_framework.monoidpowerseries.monoidpowerseries_ring import EquivariantMonoidPowerSeriesRing
from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import *
from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion_cython import reduce_GL
from sage.functions.other import ceil, floor
from sage.functions.other import sqrt
from sage.groups.all import AbelianGroup
from sage.misc.functional import isqrt
from sage.misc.latex import latex
from sage.rings.arith import gcd
from sage.rings.infinity import infinity
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.rational import Rational
from sage.structure.sage_object import SageObject
import itertools
from sage.functions.other import sqrt
from sage.rings.rational import Rational
from sage.functions.other import ceil, floor
from sage.rings.all import QQ

## TODO: Transposition must not be used for Borcherds Fourier expansions

#===============================================================================
# HermitianModularFormD2Filter_diagonal
#===============================================================================

class HermitianModularFormD2Filter_diagonal_borcherds ( HermitianModularFormD2Filter_diagonal ) :
    r"""
    This class implements a filter on the monoid of
    integral hermitian quadratic forms `[a, b, c]` in Gauss notation over
    `\mathbb{Q}(\sqrt{D})`, an imaginary quadratic field with discriminant
    `D`. It implements a bound on the diagonal entries in matrix notation,
    namely `a, c < \textrm{bound}` as well as `c \geq 0` and disc[a, b, c] = D (a c - |b|^2) > d.
    The main aspect of this class is to give an iteration over all reduced
    (or non-reduced, which isn't implemented for positive forms) indefinite
    and positive definite hermitian quadratic forms up to a given bound of
    the diagonal. These iterations are needed to work with hermitian
    Borcherds products using their Fourier expansions.
    By initiating an object of this class, most importantly the said bounds and the
    discriminant (of the imaginary quadratic number field in consideration) are
    specified.
    
    Furthermore it is possible to consider a filter for non-trivial
    characters.  See :class:~`HermitianModularFormD2Indices_diagonal_borcherds`
    for details.
    """

    def __init__(self, bound_a, bound_c = None, d = None, D = None, with_character = None, reduced = None) :
        """
        INPUT:
            - ``bound_a``         -- Positive integer or a filter for Borcherds products; bound for
                                     the diagonal element `a`.
            - ``bound_c``         -- Positive integer or ``None``; bound for the diagonal element c
                                     if bound_c is None, bound_a is used as a bound for c
            - `d`                 -- Negative rational; lower bound for the negative discriminant
                                     of quadratic forms.
                                     Must be ``None``, if ``bound_a`` is a filter.
            - `D`                 -- Negative integer; A discriminant of an imaginary quadratic field.
                                     Must be ``None``, if ``bound_a`` is a filter.
            - ``with_character``  -- Boolean (optional: default is False); Filter for modular forms
                                     with character `\nu` or not.
                                     Must be ``None``, if ``bound_a`` is a filter.
            - ``reduced`` --         Boolean (optional: default is True); Reduced filters only iterate
                                     over reduced hermitian quadratic forms.

        EXAMPLES::
            sage: HermitianModularFormD2Filter_diagonal_borcherds(4,D=-3,d=-1)
            Reduced diagonal Borcherds product filter for discriminant -3 with bounds 4 for a and 4 for c and discriminant bound -1
            sage: HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1,-4,True,False)
            Diagonal Borcherds product filter for discriminant -4 respecting characters with bounds 10 for a and 8 for c and discriminant bound -4

        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1/2,-3)
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1/2,-4,True)
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1/2,-4,True,False)
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1/2,-3,reduced=False)
            sage: HermitianModularFormD2Filter_diagonal_borcherds(5,d=-1/2,D=-3)
            Reduced diagonal Borcherds product filter for discriminant -3 with bounds 5 for a and 5 for c and discriminant bound -1/2
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1/2,-3,True)
            Traceback (most recent call last):
            ...
            ValueError: Characters are admissable only if 4 | D.
            sage: HermitianModularFormD2Filter_diagonal_borcherds(4)
            Traceback (most recent call last):
            ...
            TypeError: If bound_a is not a filter, D has to be assigned.
            sage: HermitianModularFormD2Filter_diagonal_borcherds(0.5,4,-1,-3)
            Traceback (most recent call last):
            ...
            TypeError: If bound_a is not a filter, bound_a has to be an integer or infinity.
            sage: HermitianModularFormD2Filter_diagonal_borcherds(5,0.5,-1,-3)
            Traceback (most recent call last):
            ...
            TypeError: If bound_a is not a filter, bound_c has to be None, an integer or infinity.
            sage: HermitianModularFormD2Filter_diagonal_borcherds(5,4,-sqrt(2),-3)
            Traceback (most recent call last):
            ...
            TypeError: If bound is not a filter, d has to be a rational.
            sage: HermitianModularFormD2Filter_diagonal_borcherds(-1,4,-1,-3)
            Traceback (most recent call last):
            ...
            TypeError: If bound_a is not a filter, bound_a has to be a positive integer or infinity.
            sage: HermitianModularFormD2Filter_diagonal_borcherds(5,-1,-1,-3)
            Traceback (most recent call last):
            ...
            TypeError: If bound_a is not a filter, bound_c has to be None, a positive integer or infinity.
            sage: HermitianModularFormD2Filter_diagonal_borcherds(5,4,1,-3)
            Traceback (most recent call last):
            ...
            TypeError: If bound is not a filter, d has to be a negative rational.
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1,-3); filter
            Reduced diagonal Borcherds product filter for discriminant -3 with bounds 5 for a and 4 for c and discriminant bound -1

            sage: HermitianModularFormD2Filter_diagonal_borcherds(filter,reduced=False)
            Diagonal Borcherds product filter for discriminant -3 with bounds 5 for a and 4 for c and discriminant bound -1
            sage: HermitianModularFormD2Filter_diagonal_borcherds(filter,3)
            Traceback (most recent call last):
            ...
            ValueError: bound_c cannot be reassigned.
            sage: HermitianModularFormD2Filter_diagonal_borcherds(filter,D=-4)
            Traceback (most recent call last):
            ...
            ValueError: D cannot be reassigned.
            sage: HermitianModularFormD2Filter_diagonal_borcherds(filter,d=-2)
            Traceback (most recent call last):
            ...
            ValueError: d cannot be reassigned.
            sage: HermitianModularFormD2Filter_diagonal_borcherds(filter,with_character=True)
            Traceback (most recent call last):
            ...
            ValueError: with_character cannot be reassigned.
        """
        if isinstance(bound_a, HermitianModularFormD2Filter_diagonal_borcherds) :
            if not D is None and D != bound_a.discriminant() :
                raise ValueError( "D cannot be reassigned." )
            if not d is None and d != bound_a.lower_discriminant_bound() :
                raise ValueError( "d cannot be reassigned." )
            if not with_character is None and with_character != bound_a.is_with_character() :
                raise ValueError( "with_character cannot be reassigned." )
            if not bound_c is None and bound_c != bound_a.index()[1] :
                raise ValueError( "bound_c cannot be reassigned." )
            
            self.__bound_a = bound_a.index()[0]
            self.__bound_c = bound_a.index()[1]
            self.__d = bound_a.d()
            self.__D = bound_a.D()
            self.__with_character = bound_a.is_with_character()
                
            if reduced is None :
                self.__reduced = bound_a.is_reduced()
            else :
                self.__reduced = reduced
        else :
            if D is None :
                raise TypeError( "If bound_a is not a filter, D has to be assigned." )
            if not isinstance(bound_a,(int,Integer)) and not bound_a is infinity :
                raise TypeError( "If bound_a is not a filter, bound_a has to be an integer or infinity." )
            if bound_a < 0 :
                raise TypeError( "If bound_a is not a filter, bound_a has to be a positive integer or infinity." )
            if not bound_c is None :
                if not isinstance(bound_c,(int,Integer)) and not bound_c is infinity :
                    raise TypeError( "If bound_a is not a filter, bound_c has to be None, an integer or infinity." )
                if bound_c < 0 :
                    raise TypeError( "If bound_a is not a filter, bound_c has to be None, a positive integer or infinity." )
            if not d in QQ :
                raise TypeError( "If bound is not a filter, d has to be a rational." )
            if d >= 0 :
                raise TypeError( "If bound is not a filter, d has to be a negative rational." )
            
            if with_character is None : 
                self.__with_character = False
            else :
                self.__with_character = with_character

            if self.__with_character and D % 4 != 0 :
                raise ValueError( "Characters are admissable only if 4 | D." )

            if reduced is None :
                self.__reduced = True
            else :
                self.__reduced = reduced
                
            if not self.__with_character :
                self.__bound_a = bound_a
                if bound_c is None :
                    self.__bound_c = bound_a
                else :
                    self.__bound_c = bound_c
            else :
                self.__bound_a = 2*bound_a
                if bound_c is None :
                    self.__bound_c = 2*bound_a
                else :
                    self.__bound_c = 2*bound_c
                if D % 4 != 0 or D // 4 % 4 not in [2,3] :
                    raise ValueError( "Characters are admissable only if 4 | D." )
            if not self.__with_character :
                self.__d = d
            else :
                self.__d = 4*d
                if D % 4 != 0 or D // 4 % 4 not in [2,3] :
                    raise ValueError( "Characters are admissable only if 4 | D." )
            self.__D = D
            
    def is_infinite(self) :
        """
        Returns whether the filter contains infinitely many elements or not.

        OUTPUT:
            - A boolean

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1,-3)
            sage: filter.is_infinite()
            False
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(infinity,4,-1,-3)
            sage: filter.is_infinite()
            True
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,infinity,-1,-3)
            sage: filter.is_infinite()
            True
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(infinity,infinity,-1,-3)
            sage: filter.is_infinite()
            True
        """
        return self.__bound_a is infinity or self.__bound_c is infinity
    
    def is_all(self) :
        """
        Returns whether the filter contains all elements or not.

        OUTPUT:
            - A boolean

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1,-3)
            sage: filter.is_all()
            False
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(infinity,4,-1,-3)
            sage: filter.is_all()
            False
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,infinity,-1,-3)
            sage: filter.is_all()
            False
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(infinity,infinity,-1,-3)
            sage: filter.is_all()
            True
        """
        return self.__bound_a is infinity and self.__bound_c is infinity

    def index(self) :
        """
        Return the vitual index, namely if the filter respects characters,
        return twice the bounds, since we save twice the Fourier indices.

        OUTPUT:
            - a tuple of two positive integers

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1,-4)
            sage: filter.index()
            (5, 4)
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1,-4,with_character=True)
            sage: filter.index()
            (10, 8)

        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(infinity,4,-1,-3)
            sage: filter.index()
            (+Infinity, 4)
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,infinity,-1,-3)
            sage: filter.index()
            (5, +Infinity)
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(infinity,infinity,-1,-3)
            sage: filter.index()
            (+Infinity, +Infinity)
        """
        return (self.__bound_a,self.__bound_c)

    def _enveloping_content_bound(self) :
        """
        Return a bound `B` such that any indefinite element of ``self`` has at most
        content `B`. The bound will not necessarily be attained.

        OUTPUT:
            - a positive integer

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1,-4)
            sage: filter._enveloping_content_bound()
            5
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,5,-1,-4)
            sage: filter._enveloping_content_bound()
            5
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1,-4,with_character=True)
            sage: filter._enveloping_content_bound()
            10

        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(infinity,4,-1,-3)
            sage: filter._enveloping_content_bound()
            +Infinity
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,infinity,-1,-3)
            sage: filter._enveloping_content_bound()
            +Infinity
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(infinity,infinity,-1,-3)
            sage: filter._enveloping_content_bound()
            +Infinity
        """
        return max(self.__bound_a,self.__bound_c)
    
    def _enveloping_discriminant_bound(self) :
        """
        Returns an enveloping discriminant bound. Namely a maximal discriminant
        for elements in this filter. The discriminant is D * det T for an index T.

        NOTES:
            The virtual bound for the diagonal of T is twice the bound if the filter
            respects characters, since we save twice the Fourier indices.

        OUTPUT:
            - a positive integer

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1,-4)
            sage: filter._enveloping_discriminant_bound()
            49
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1,-4,with_character=True)
            sage: filter._enveloping_discriminant_bound()
            253
        """
        return - self.__D * (self.__bound_a - 1) * (self.__bound_c - 1) + 1
    
    def lower_discriminant_bound(self) :
        """
        Returns the lower discriminant bound of the filter.

        OUTPUT:
            - A negative rational

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-2,-4)
            sage: filter.lower_discriminant_bound()
            -2
        """
        return self.__d
    
    d = lower_discriminant_bound
    
    def discriminant(self) :
        """
        Returns the discriminant of the imaginary quadratic field associated to ``self``.

        OUTPUT:
            - A negative integer

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1/2,-4)
            sage: filter.discriminant()
            -4
        """
        return self.__D
    
    D = discriminant
    
    def is_with_character(self) :
        """
        Returns whether the filter is associated to Fourier expansions
        with a character `\nu` or not.

        OUTPUT:
            - A boolean

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1/2,-4)
            sage: filter.is_with_character()
            False
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1/2,-4,with_character=True)
            sage: filter.is_with_character()
            True
        """
        return self.__with_character
    
    def is_reduced(self) :
        """
        Returns whether the filter respects the action of `GL_2(\mathfrak{o}_{\mathbb{Q}(\sqrt{D})})`.

        OUTPUT:
            - A boolean

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1/2,-4)
            sage: filter.is_reduced()
            True
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(5,4,-1/2,-4,reduced=False)
            sage: filter.is_reduced()
            False
        """
        return self.__reduced
    
    def __contains__(self, f) :
        """
        Returns whether the filter contains `f=[a,b,c]` with `b=b_1/\sqrt{D}+b2(1+\sqrt{D})/2`
        or not. Look into the code for details.

        INPUT:
            - `f` -- A tuple of four integers (representation of a matrix given via the tuple)

        OUTPUT:
            - A boolean

        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4)
            sage: filter.__contains__((1,0,0,1))
            True
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4)
            sage: filter.__contains__((4,0,0,1))
            False
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4,with_character=True)
            sage: filter.__contains__((4,0,0,1))
            True
        """
        if self.__bound is infinity :
            return True
        
        (a, b1, b2, c) = f
        return a < self.__bound_a and c < self.__bound_c and a >= 0 and c >= 0 and (a > 0 or c > 0) and (- self.__D * a * c - b1**2 - self.__D * b1 * b2 - (self.__D**2 - self.__D) // 4 * b2**2 > self.__d)
    
    def __iter__(self) :
        """
        Iterate over all integral hermitian quadratic forms contained
        in the filter, possibly up to reduction. The content and discriminant of
        the form will be given, too.

        NOTES:
            You may not iterate over infinite filters.
            The iteration over none reduced forms is not implemented yet.

        OUTPUT:
            - A generator over 3-tuples of a 4-tuple of integers and 2 integers.

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-3,d=-2)
            sage: list(filter)
[((0, 0, 0, 0), 0, 0), ((0, 0, 0, 1), 1, 0), ((0, 0, 0, 2), 2, 0), ((0, 0, 0, 3), 3, 0), ((1, 0, 0, 1), 1, 3), ((1, 1, 1, 1), 1, 2), ((1, 0, 0, 2), 1, 6), ((1, 1, 1, 2), 1, 5), ((1, 0, 0, 3), 1, 9), ((1, 1, 1, 3), 1, 8), ((2, 0, 0, 2), 2, 12), ((2, 1, 1, 2), 1, 11), ((2, 2, 2, 2), 2, 8), ((2, 3, 2, 2), 1, 9), ((2, 0, 0, 3), 1, 18), ((2, 1, 1, 3), 1, 17), ((2, 2, 2, 3), 1, 14), ((2, 3, 2, 3), 1, 15), ((3, 0, 0, 3), 3, 27), ((3, 1, 1, 3), 1, 26), ((3, 2, 2, 3), 1, 23), ((3, 3, 2, 3), 1, 24), ((3, 3, 3, 3), 3, 18), ((3, 4, 3, 3), 1, 20), ((1, -4, -2, 1), 1, -1), ((1, -3, -2, 1), 1, 0), ((1, -2, -2, 1), 1, -1), ((1, -3, -1, 1), 1, 0), ((1, 0, -1, 1), 1, 0), ((1, -2, 0, 1), 1, -1), ((1, 2, 0, 1), 1, -1), ((1, 0, 1, 1), 1, 0), ((1, 3, 1, 1), 1, 0), ((1, 2, 2, 1), 1, -1), ((1, 3, 2, 1), 1, 0), ((1, 4, 2, 1), 1, -1), ((1, -5, -3, 2), 1, -1), ((1, -4, -3, 2), 1, -1), ((1, -5, -2, 2), 1, -1), ((1, -1, -2, 2), 1, -1), ((1, -4, -1, 2), 1, -1), ((1, 1, -1, 2), 1, -1), ((1, -1, 1, 2), 1, -1), ((1, 4, 1, 2), 1, -1), ((1, 1, 2, 2), 1, -1), ((1, 5, 2, 2), 1, -1), ((1, 4, 3, 2), 1, -1), ((1, 5, 3, 2), 1, -1), ((1, -6, -3, 3), 1, 0), ((1, -3, -3, 3), 1, 0), ((1, -3, 0, 3), 1, 0), ((1, 3, 0, 3), 1, 0), ((1, 3, 3, 3), 1, 0), ((1, 6, 3, 3), 1, 0), ((2, -5, -3, 1), 1, -1), ((2, -4, -3, 1), 1, -1), ((2, -5, -2, 1), 1, -1), ((2, -1, -2, 1), 1, -1), ((2, -4, -1, 1), 1, -1), ((2, 1, -1, 1), 1, -1), ((2, -1, 1, 1), 1, -1), ((2, 4, 1, 1), 1, -1), ((2, 1, 2, 1), 1, -1), ((2, 5, 2, 1), 1, -1), ((2, 4, 3, 1), 1, -1), ((2, 5, 3, 1), 1, -1), ((2, -7, -4, 2), 1, -1), ((2, -6, -4, 2), 2, 0), ((2, -5, -4, 2), 1, -1), ((2, -7, -3, 2), 1, -1), ((2, -2, -3, 2), 1, -1), ((2, -6, -2, 2), 2, 0), ((2, 0, -2, 2), 2, 0), ((2, -5, -1, 2), 1, -1), ((2, 2, -1, 2), 1, -1), ((2, -2, 1, 2), 1, -1), ((2, 5, 1, 2), 1, -1), ((2, 0, 2, 2), 2, 0), ((2, 6, 2, 2), 2, 0), ((2, 2, 3, 2), 1, -1), ((2, 7, 3, 2), 1, -1), ((2, 5, 4, 2), 1, -1), ((2, 6, 4, 2), 2, 0), ((2, 7, 4, 2), 1, -1), ((2, -8, -5, 3), 1, -1), ((2, -7, -5, 3), 1, -1), ((2, -8, -3, 3), 1, -1), ((2, -1, -3, 3), 1, -1), ((2, -7, -2, 3), 1, -1), ((2, 1, -2, 3), 1, -1), ((2, -1, 2, 3), 1, -1), ((2, 7, 2, 3), 1, -1), ((2, 1, 3, 3), 1, -1), ((2, 8, 3, 3), 1, -1), ((2, 7, 5, 3), 1, -1), ((2, 8, 5, 3), 1, -1), ((3, -6, -3, 1), 1, 0), ((3, -3, -3, 1), 1, 0), ((3, -3, 0, 1), 1, 0), ((3, 3, 0, 1), 1, 0), ((3, 3, 3, 1), 1, 0), ((3, 6, 3, 1), 1, 0), ((3, -8, -5, 2), 1, -1), ((3, -7, -5, 2), 1, -1), ((3, -8, -3, 2), 1, -1), ((3, -1, -3, 2), 1, -1), ((3, -7, -2, 2), 1, -1), ((3, 1, -2, 2), 1, -1), ((3, -1, 2, 2), 1, -1), ((3, 7, 2, 2), 1, -1), ((3, 1, 3, 2), 1, -1), ((3, 8, 3, 2), 1, -1), ((3, 7, 5, 2), 1, -1), ((3, 8, 5, 2), 1, -1), ((3, -10, -6, 3), 1, -1), ((3, -9, -6, 3), 3, 0), ((3, -8, -6, 3), 1, -1), ((3, -10, -4, 3), 1, -1), ((3, -2, -4, 3), 1, -1), ((3, -9, -3, 3), 3, 0), ((3, 0, -3, 3), 3, 0), ((3, -8, -2, 3), 1, -1), ((3, 2, -2, 3), 1, -1), ((3, -2, 2, 3), 1, -1), ((3, 8, 2, 3), 1, -1), ((3, 0, 3, 3), 3, 0), ((3, 9, 3, 3), 3, 0), ((3, 2, 4, 3), 1, -1), ((3, 10, 4, 3), 1, -1), ((3, 8, 6, 3), 1, -1), ((3, 9, 6, 3), 3, 0), ((3, 10, 6, 3), 1, -1), ((0, -2, -1, 1), 1, -1), ((0, -1, -1, 1), 1, -1), ((0, -1, 0, 1), 1, -1), ((0, 0, 0, 1), 1, 0), ((0, 1, 0, 1), 1, -1), ((0, 1, 1, 1), 1, -1), ((0, 2, 1, 1), 1, -1), ((0, -2, -1, 2), 1, -1), ((0, -1, -1, 2), 1, -1), ((0, -1, 0, 2), 1, -1), ((0, 0, 0, 2), 2, 0), ((0, 1, 0, 2), 1, -1), ((0, 1, 1, 2), 1, -1), ((0, 2, 1, 2), 1, -1), ((0, -2, -1, 3), 1, -1), ((0, -1, -1, 3), 1, -1), ((0, -1, 0, 3), 1, -1), ((0, 0, 0, 3), 3, 0), ((0, 1, 0, 3), 1, -1), ((0, 1, 1, 3), 1, -1), ((0, 2, 1, 3), 1, -1), ((1, -2, -1, 0), 1, -1), ((1, -1, -1, 0), 1, -1), ((1, -1, 0, 0), 1, -1), ((1, 0, 0, 0), 1, 0), ((1, 1, 0, 0), 1, -1), ((1, 1, 1, 0), 1, -1), ((1, 2, 1, 0), 1, -1), ((2, -2, -1, 0), 1, -1), ((2, -1, -1, 0), 1, -1), ((2, -1, 0, 0), 1, -1), ((2, 0, 0, 0), 2, 0), ((2, 1, 0, 0), 1, -1), ((2, 1, 1, 0), 1, -1), ((2, 2, 1, 0), 1, -1), ((3, -2, -1, 0), 1, -1), ((3, -1, -1, 0), 1, -1), ((3, -1, 0, 0), 1, -1), ((3, 0, 0, 0), 3, 0), ((3, 1, 0, 0), 1, -1), ((3, 1, 1, 0), 1, -1), ((3, 2, 1, 0), 1, -1), ((0, -1, 0, 0), 1, -1), ((0, -2, -1, 0), 1, -1), ((0, -1, -1, 0), 1, -1)]

        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(infinity,D=-3,d=-1)
            sage: filter.__iter__()
            Traceback (most recent call last):
            ...
            ArithmeticError: Cannot iterate over infinite filters.
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(70,D=-3,d=-1)   # long time
            sage: indices = list(form[0] for form in filter if (form[2]>1))                # long time
            sage: list(reduce_GL(form,-3)[0] for form in indices) == indices               # long time
            True
        """
        return itertools.chain( self.iter_semidefinite_forms_for_character_with_content_and_discriminant(False),
                                self.iter_semidefinite_forms_for_character_with_content_and_discriminant(True),
                                self.iter_positive_forms_for_character_with_content_and_discriminant(False),
                                self.iter_positive_forms_for_character_with_content_and_discriminant(True),
                                self.iter_positive_diagonal_indefinite_forms_with_content_and_discriminant(),
                                self.iter_indefinite_diagonal_indefinite_forms_with_content_and_discriminant(),
                                self.iter_semidefinite_diagonal_indefinite_forms_with_content_and_discriminant(),
                                self.iter_zero_diagonal_indefinite_forms_with_content_and_discriminant() )
    
    def iter_semidefinite_forms_for_character_with_content_and_discriminant( self, for_character = False) :
        """
        Iterate over all semidefinite, non-definite integral hermitian
        quadratic forms contained in the filter. The content and
        discriminant of the form will be given, too.

        NOTES:
            If ``for_character`` is True, the iteration will be empty.
            You may not iterate over infinite filters.
            The iteration over none reduced forms is not implemented yet.

        INPUT:
            - ``for_character`` -- A boolean (optinal: default is False)
                                   If False only those forms will be iterated
                                   which can occure in the Fourier expansion
                                   of a form without character. If True only
                                   those will be iterated which occure in the
                                   Fourier expansion of a form with character.

        OUTPUT:
            - A generator over 3-tuples of a 4-tuple of integers and 2 integers.

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-3,d=-1)
            sage: list( filter.iter_semidefinite_forms_for_character_with_content_and_discriminant() )
            [((0, 0, 0, 0), 0, 0), ((0, 0, 0, 1), 1, 0), ((0, 0, 0, 2), 2, 0), ((0, 0, 0, 3), 3, 0)]
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-4,d=-1,with_character=True)
            sage: list( filter.iter_semidefinite_forms_for_character_with_content_and_discriminant() )
            [((0, 0, 0, 0), 0, 0), ((0, 0, 0, 2), 1, 0), ((0, 0, 0, 4), 2, 0), ((0, 0, 0, 6), 3, 0)]

        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(infinity,D=-3,d=-1)
            sage: filter.iter_semidefinite_forms_for_character_with_content_and_discriminant()
            Traceback (most recent call last):
            ...
            ArithmeticError: Cannot iterate over infinite filters.
        """
        if self.is_infinite() :
            raise ArithmeticError( "Cannot iterate over infinite filters." )
        if self.__with_character and not for_character :
            return (((0,0,0,c), c//2, 0) for c in xrange(0, self.__bound_c, 2))
        elif not self.__with_character and not for_character :
            return (((0,0,0,c), c, 0) for c in xrange(0, self.__bound_c))
        else :
            return (a for a in [])
    
    def iter_positive_forms_for_character_with_content_and_discriminant(self, for_character = False) :
        """
        Iterate over all positive definite integral hermitian quadratic
        forms contained in the filter. The content and discriminant of
        the form will be given, too.
        
        INPUT:
            - ``for_character`` -- A boolean (optinal: default is False)
                                   If False only those forms will be iterated
                                   which can occure in the Fourier expansion
                                   of a form without character. If True only
                                   those will be iterated which occure in the
                                   Fourier expansion of a form with character.

        NOTES:
            You may not iterate over infinite filters.
            The iteration over none reduced forms is not implemented yet.

        OUTPUT:
            - A generator over 3-tuples of a 4-tuple of integers and 2 integers.

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-3,d=-1)
            sage: list(filter.iter_positive_forms_for_character_with_content_and_discriminant())
            [((1, 0, 0, 1), 1, 3), ((1, 1, 1, 1), 1, 2), ((1, 0, 0, 2), 1, 6), ((1, 1, 1, 2), 1, 5), ((1, 0, 0, 3), 1, 9), ((1, 1, 1, 3), 1, 8), ((2, 0, 0, 2), 2, 12), ((2, 1, 1, 2), 1, 11), ((2, 2, 2, 2), 2, 8), ((2, 3, 2, 2), 1, 9), ((2, 0, 0, 3), 1, 18), ((2, 1, 1, 3), 1, 17), ((2, 2, 2, 3), 1, 14), ((2, 3, 2, 3), 1, 15), ((3, 0, 0, 3), 3, 27), ((3, 1, 1, 3), 1, 26), ((3, 2, 2, 3), 1, 23), ((3, 3, 2, 3), 1, 24), ((3, 3, 3, 3), 3, 18), ((3, 4, 3, 3), 1, 20)]

        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds =  HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-3,d=-1)
            sage: iterated = map(lambda s: s[0], filter.iter_positive_forms_for_character_with_content_and_discriminant())
            sage: map(lambda s: inds.reduce(s)[0], iterated) == iterated
            True
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(infinity,D=-3,d=-1)
            sage: list(filter.iter_positive_forms_for_character_with_content_and_discriminant())
            Traceback (most recent call last):
            ...
            ArithmeticError: Cannot iterate over infinite filters.
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-3,d=-1,reduced=False)
            sage: list(filter.iter_positive_forms_for_character_with_content_and_discriminant())
            Traceback (most recent call last):
            ...
            NotImplementedError: Iteration over non reduced forms
        """
        if self.is_infinite() :
            raise ArithmeticError( "Cannot iterate over infinite filters." )

        if not self.__with_character and for_character :
            raise StopIteration

        D = self.__D
        DsqmD = (D**2 - D) // 4

        if self.__with_character :
            stepsize = 2
        else :
            stepsize = 1
        
        if self.__reduced :
            for a in xrange(1 if for_character or not self.__with_character else 2, self.__bound_a, stepsize) :
                for c in xrange(a, self.__bound_c, stepsize) :
                    ac_gcd = gcd(a,c)
                    ac_disc = -D * a * c
                    ## B1 ensures that we can take the square root of B2
                    B1 = isqrt(4 * a * c)
                    
                    if D == -3 :                   
                        for b2 in xrange(1 if for_character else 0, min(a, B1) + 1, stepsize) :
                            acb2_gcd = gcd(ac_gcd, b2)                            
                            
                            ## *_red : bounds given by general reduction of forms
                            ## *_redex : bounds given by sqrt(-3) reduction of forms
                            ## *_def : bounds given by positive definiteness of forms
                            b1min_red = max( 0, - ((2 * D * b2 - D * a) // 4) )
                            b1max_red = (-2 * D * b2 - D * a) // 4 + 1
                            b1min_redex = b2
                            b1max_redex = (-D * b2) // 2 + 1 

                            B2 = D * (b2**2 - 4 * a * c)
                            b1min_def = (-D * b2 - isqrt(B2)) // 2
                            b1max_def = (-D * b2 + isqrt(B2 - 1)) // 2 + 1 \
                                        if B2 > 0 else \
                                        (-D * b2 ) // 2
                            
                            b1min = max(0, b1min_red, b1min_redex, b1min_def)
                            if stepsize == 2 :
                                if for_character :
                                    b1min = b1min + 1 - (b1min % 2)
                                else :
                                    b1min = b1min + (b1min % 2)
                            for b1 in xrange( b1min, min(b1max_red, b1max_def, b1max_redex), stepsize ) :
                                yield ( (a, b1, b2, c), gcd(acb2_gcd, b1),
                                        ac_disc - DsqmD * b2**2 - b1**2 - D * b1 * b2 ) 
                    #! if D == -3
                    elif D == -4 :
                        ## TODO: Check the calculations and document them. 
                        raise NotImplementedError

                        b2min = -min(a, B1)
                        if stepsize == 2 :
                            if for_character :
                                b2min = b2min + 1 - (b2min % 2)
                            else :
                                b2min = b2min + (b2min % 2)
                        for b2 in xrange(b2min, min(a, B1) + 1, stepsize) :
                            acb2_gcd = gcd(ac_gcd, b2)                            
                            
                            ## *_red : bounds given by general reduction of forms
                            ## *_redex : bounds given by sqrt(-4) reduction of forms
                            ## *_def : bounds given by positive definiteness of forms
                            B2 = D * (b2**2 - 4 * a * c)
                            b1min_def = (-D * b2 - isqrt(B2)) // 2
                            b1max_def = (-D * b2 + isqrt(B2 - 1)) // 2 + 1 \
                                        if B2 > 0 else \
                                        (-D * b2 ) // 2

                            b1min_red = (-2 * D * b2 + D * a - 1) // 4 + 1
                            b1max_red = (-2 * D * b2 - D * a) // 4 + 1

                            b1min_redex = 2 * b2
                            b1max_redex = 3 * b2 + 1


                            b1min = max( b1min_def, b1min_red, b1min_redex )
                            if stepsize == 2 :
                                if for_character :
                                    b1min = b1min + 1 - (b1min % 2)
                                else :
                                    b1min = b1min + (b1min % 2)
                            for b1 in xrange( b1min, min( b1min_def, b1min_red, b1min_redex ), stepsize ) :
                                yield ( (a, b1, b2, c), gcd(acb2_gcd, b1),
                                        ac_disc - DsqmD * b2**2 - b1**2 - D * b1 * b2 ) 
                    #! if D == -4
                    else :
                        ## TODO: Check the calculations and document them. 
                        raise NotImplementedError

                        b2min = -min(a, B1)
                        if stepsize == 2 :
                            if for_character :
                                b2min = b2min + 1 - (b2min % 2)
                            else :
                                b2min = b2min + (b2min % 2)
                        for b2 in xrange(b2min, min(a, B1) + 1, 2) :
                            acb2_gcd = gcd(ac_gcd, b2)
                            
                            ## *_red : bounds given by general reduction of forms
                            ## *_def : bounds given by positive definiteness of forms
                            B2 = D * (b2**2 - 4 * a * c)
                            b1min_def = (-D * b2 - isqrt(B2)) // 2
                            b1max_def = (-D * b2 + isqrt(B2 - 1)) // 2 + 1 \
                                        if B2 > 0 else \
                                        (-D * b2 ) // 2

                            b1min_red = (-2 * D * b2 + D * a - 1) // 4 + 1
                            b1max_red = (-2 * D * b2 - D * a) // 4 + 1


                            b1min = max( b1min_def, b1min_red )
                            if stepsize == 2 :
                                if for_character :
                                    b1min = b1min + 1 - (b1min % 2)
                                else :
                                    b1min = b1min + (b1min % 2)
                            for b1 in xrange( b1min, min( b1min_def, b1min_red, b1min_redex ), stepsize ) :
                                yield ( (a, b1, b2, c), gcd(acb2_gcd, b1),
                                        ac_disc - DsqmD * b2**2 - b1**2 - D * b1 * b2 ) 
                    #! else D == -3
                #! for c in xrange(a, self.__bound, 2)
            #! for a in xrange(0,self.__bound, 2)
        #! if self.__reduced
        else :
            raise NotImplementedError, "Iteration over non reduced forms"
        #! if self.__reduced

        raise StopIteration

    def iter_positive_diagonal_indefinite_forms_with_content_and_discriminant(self) :
        """
        Iterate over all non-positive definite integral hermitian quadratic
        forms with positive diagonal contained in the filter. The content
        and discriminant of the form will be given, too.
        

        NOTES:
            You may not iterate over infinite filters.
            The iteration over none reduced forms is not implemented yet.

        OUTPUT:
            - A generator over 3-tuples of a 4-tuple of integers and 2 integers.

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(2,D=-3,d=-5)
            sage: list(filter.iter_positive_diagonal_indefinite_forms_with_content_and_discriminant())
            [((1, -5, -3, 1), 1, -4), ((1, -4, -3, 1), 1, -4), ((1, -5, -2, 1), 1, -4), ((1, -4, -2, 1), 1, -1), ((1, -3, -2, 1), 1, 0), ((1, -2, -2, 1), 1, -1), ((1, -1, -2, 1), 1, -4), ((1, -4, -1, 1), 1, -4), ((1, -3, -1, 1), 1, 0), ((1, 0, -1, 1), 1, 0), ((1, 1, -1, 1), 1, -4), ((1, -2, 0, 1), 1, -1), ((1, 2, 0, 1), 1, -1), ((1, -1, 1, 1), 1, -4), ((1, 0, 1, 1), 1, 0), ((1, 3, 1, 1), 1, 0), ((1, 4, 1, 1), 1, -4), ((1, 1, 2, 1), 1, -4), ((1, 2, 2, 1), 1, -1), ((1, 3, 2, 1), 1, 0), ((1, 4, 2, 1), 1, -1), ((1, 5, 2, 1), 1, -4), ((1, 4, 3, 1), 1, -4), ((1, 5, 3, 1), 1, -4)]

        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(infinity,D=-3,d=-1)
            sage: list(filter.iter_positive_diagonal_indefinite_forms_with_content_and_discriminant())
            Traceback (most recent call last):
            ...
            ArithmeticError: Cannot iterate over infinite filters.
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-3,d=-1,reduced=False)
            sage: list(filter.iter_positive_diagonal_indefinite_forms_with_content_and_discriminant())
            Traceback (most recent call last):
            ...
            NotImplementedError: Iteration over non reduced forms
        """
        if self.is_infinite() :
            raise ArithmeticError( "Cannot iterate over infinite filters." )

        if self.__reduced :
            D = self.__D
            d = self.__d
            DsqmD = (D**2 - D) // 4

            for a in xrange(1, self.__bound_a) :
                for c in xrange(1, self.__bound_c) :
                    gcd_ac = gcd(a, c)
                    disc_ac = -D * a * c
                    b2min = -isqrt(4*a*c + (4*(d+1)) // D)
                    b2max = -b2min + 1

                    for b2 in xrange(b2min, b2max) :
                        gcd_acb2 = gcd(b2, gcd_ac)
                        B2 = D * ( b2**2 - 4 * a * c) - 4 * (d+1)
                        b1min = -((D * b2 + isqrt(B2)) // 2)
                        b1max = ((-D * b2 + isqrt(B2)) // 2) + 1
                        B3 = D * (b2**2 - 4 * a * c)

                        if B3 >= 0 :
                            if (B3 - isqrt(B3)**2 == 0):
                                b1min_def = ((-D * b2 - isqrt(B3)) // 2) + 1
                                b1max_def = -(-(-D * b2 + isqrt(B3)) // 2)
                            else:
                                b1min_def = ((-D * b2 - (isqrt(B3)+1)) // 2) + 1
                                b1max_def = -(-(-D * b2 + (isqrt(B3)+1)) // 2)

                            ## To avoid double iteration we modify b1min_def
                            if b1min_def == b1max_def + 1 :
                                b1min_def -= 1
                            
                            for b1 in xrange( b1min, b1min_def ) :
                                yield ( (a, b1, b2, c), gcd(b1, gcd_acb2), disc_ac - b1**2 - D * b1 * b2 - DsqmD * b2**2)

                            for b1 in xrange( b1max_def, b1max ) :
                                yield ( (a, b1, b2, c), gcd(b1, gcd_acb2), disc_ac - b1**2 - D * b1 * b2 - DsqmD * b2**2)

                        else:
                            for b1 in xrange( b1min, b1max ) :
                                yield ( (a, b1, b2, c), gcd(b1, gcd_acb2), disc_ac - b1**2 - D * b1 * b2 - DsqmD * b2**2)

        else :
            raise NotImplementedError, "Iteration over non reduced forms"

        raise StopIteration


    def iter_indefinite_diagonal_indefinite_forms_with_content_and_discriminant(self) :
        """
        Iterate over all non-positive definite integral hermitian quadratic
        forms with first diagonal entry non-positive and second diagonal
        entry positive contained in the filter. The content and discriminant
        of the form will be given, too.
        

        NOTES:
            You may not iterate over infinite filters.
            The iteration over none reduced forms is not implemented yet.

        OUTPUT:
            - A generator over 3-tuples of a 4-tuple of integers and 2 integers.

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(2,D=-3,d=-5)
            sage: list(filter.iter_indefinite_diagonal_indefinite_forms_with_content_and_discriminant())
            [((-1, -2, -1, 1), 1, -4), ((-1, -1, -1, 1), 1, -4), ((-1, -1, 0, 1), 1, -4), ((-1, 0, 0, 1), 1, -3), ((-1, 1, 0, 1), 1, -4), ((-1, 1, 1, 1), 1, -4), ((-1, 2, 1, 1), 1, -4), ((0, -4, -2, 1), 1, -4), ((0, -3, -2, 1), 1, -3), ((0, -2, -2, 1), 1, -4), ((0, -3, -1, 1), 1, -3), ((0, -2, -1, 1), 1, -1), ((0, -1, -1, 1), 1, -1), ((0, 0, -1, 1), 1, -3), ((0, -2, 0, 1), 1, -4), ((0, -1, 0, 1), 1, -1), ((0, 0, 0, 1), 1, 0), ((0, 1, 0, 1), 1, -1), ((0, 2, 0, 1), 1, -4), ((0, 0, 1, 1), 1, -3), ((0, 1, 1, 1), 1, -1), ((0, 2, 1, 1), 1, -1), ((0, 3, 1, 1), 1, -3), ((0, 2, 2, 1), 1, -4), ((0, 3, 2, 1), 1, -3), ((0, 4, 2, 1), 1, -4)]


        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(infinity,D=-3,d=-1)
            sage: list(filter.iter_indefinite_diagonal_indefinite_forms_with_content_and_discriminant())
            Traceback (most recent call last):
            ...
            ArithmeticError: Cannot iterate over infinite filters.
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-3,d=-1,reduced=False)
            sage: list(filter.iter_indefinite_diagonal_indefinite_forms_with_content_and_discriminant())
            Traceback (most recent call last):
            ...
            NotImplementedError: Iteration over non reduced forms
        """
        if self.is_infinite() :
            raise ArithmeticError( "Cannot iterate over infinite filters." )

        if self.__reduced :
            D = self.__D
            d = self.__d
            DsqmD = (D**2 - D) // 4

            for c in xrange(1, self.__bound_c) :
                for a in xrange(-((-(d+1))//(-D * c)), 1) :
                    gcd_ac = gcd(a, c)
                    disc_ac = -D * a * c
                    b2min = -isqrt(4*a*c + (4*(d+1)) // D)
                    b2max = -b2min + 1

                    for b2 in xrange(b2min, b2max) :
                        gcd_acb2 = gcd(b2, gcd_ac)
                        B2 = D * ( b2**2 - 4 * a * c) - 4 * (d+1)
                        b1min = -((D * b2 + isqrt(B2)) // 2)
                        b1max = ((-D * b2 + isqrt(B2)) // 2) + 1
                        B3 = D * (b2**2 - 4 * a * c)

                        if B3 >= 0 :
                            if (B3 - isqrt(B3)**2 == 0):
                                b1min_def = ((-D * b2 - isqrt(B3)) // 2) + 1
                                b1max_def = -(-(-D * b2 + isqrt(B3)) // 2)
                            else:
                                b1min_def = ((-D * b2 - (isqrt(B3)+1)) // 2) + 1
                                b1max_def = -(-(-D * b2 + (isqrt(B3)+1)) // 2)

                            ## To avoid double iteration we modify b1min_def
                            if b1min_def == b1max_def + 1 :
                                b1min_def -= 1

                            ##FIXME: Iteration over b1 does not take the discriminant bound into account
                            for b1 in xrange( b1min, b1min_def ) :
                                yield ( (a, b1, b2, c), gcd(b1, gcd_acb2), disc_ac - b1**2 - D * b1 * b2 - DsqmD * b2**2)

                            for b1 in xrange( b1max_def, b1max ) :
                                yield ( (a, b1, b2, c), gcd(b1, gcd_acb2), disc_ac - b1**2 - D * b1 * b2 - DsqmD * b2**2)

                        else:
                            for b1 in xrange( b1min, b1max ) :
                                yield ( (a, b1, b2, c), gcd(b1, gcd_acb2), disc_ac - b1**2 - D * b1 * b2 - DsqmD * b2**2)

        else :
            raise NotImplementedError, "Iteration over non reduced forms"

        raise StopIteration


    def iter_semidefinite_diagonal_indefinite_forms_with_content_and_discriminant(self) :
        """
        Iterate over all non-positive definite integral hermitian quadratic
        forms with first diagonal entry positive and second diagonal entry
        zero contained in the filter. The content and discriminant of the
        form will be given, too.
        

        NOTES:
            You may not iterate over infinite filters.
            The iteration over none reduced forms is not implemented yet.

        OUTPUT:
            - A generator over 3-tuples of a 4-tuple of integers and 2 integers.

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(2,D=-3,d=-5)
            sage: list(filter.iter_semidefinite_diagonal_indefinite_forms_with_content_and_discriminant())
            [((1, -4, -2, 0), 1, -4), ((1, -3, -2, 0), 1, -3), ((1, -2, -2, 0), 1, -4), ((1, -3, -1, 0), 1, -3), ((1, -2, -1, 0), 1, -1), ((1, -1, -1, 0), 1, -1), ((1, 0, -1, 0), 1, -3), ((1, -2, 0, 0), 1, -4), ((1, -1, 0, 0), 1, -1), ((1, 0, 0, 0), 1, 0), ((1, 1, 0, 0), 1, -1), ((1, 2, 0, 0), 1, -4), ((1, 0, 1, 0), 1, -3), ((1, 1, 1, 0), 1, -1), ((1, 2, 1, 0), 1, -1), ((1, 3, 1, 0), 1, -3), ((1, 2, 2, 0), 1, -4), ((1, 3, 2, 0), 1, -3), ((1, 4, 2, 0), 1, -4)]


        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(infinity,D=-3,d=-1)
            sage: list(filter.iter_semidefinite_diagonal_indefinite_forms_with_content_and_discriminant())
            Traceback (most recent call last):
            ...
            ArithmeticError: Cannot iterate over infinite filters.
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-3,d=-1,reduced=False)
            sage: list(filter.iter_semidefinite_diagonal_indefinite_forms_with_content_and_discriminant())
            Traceback (most recent call last):
            ...
            NotImplementedError: Iteration over non reduced forms.
        """
        if self.is_infinite() :
            raise ArithmeticError( "Cannot iterate over infinite filters." )

        if self.__reduced :
            D = self.__D
            d = self.__d
            DsqmD = (D**2 - D) // 4

            for a in xrange(1, self.__bound_a) :
                c = 0
                gcd_ac = a
                disc_ac = 0
                b2min = -isqrt((4*(d+1)) // D)
                b2max = -b2min + 1

                for b2 in xrange(b2min, b2max) :
                    gcd_acb2 = gcd(b2, gcd_ac)
                    B2 = D * ( b2**2) - 4 * (d+1)
                    b1min = -((D * b2 + isqrt(B2)) // 2)
                    b1max = ((-D * b2 + isqrt(B2)) // 2) + 1
                    B3 = D * (b2**2)

                    if B3 >= 0 :
                        if (B3 - isqrt(B3)**2 == 0):
                            b1min_def = ((-D * b2 - isqrt(B3)) // 2) + 1
                            b1max_def = -(-(-D * b2 + isqrt(B3)) // 2)
                        else:
                            b1min_def = ((-D * b2 - (isqrt(B3)+1)) // 2) + 1
                            b1max_def = -(-(-D * b2 + (isqrt(B3)+1)) // 2)
                        
                        ## To avoid double iteration we modify b1min_def
                        if b1min_def == b1max_def + 1 :
                            b1min_def -= 1
                        
                        ##FIXME: Iteration over b1 does not take the discriminant bound into account
                        for b1 in xrange( b1min, b1min_def ) :
                            yield ( (a, b1, b2, c), gcd(b1, gcd_acb2), disc_ac - b1**2 - D * b1 * b2 - DsqmD * b2**2)

                        for b1 in xrange( b1max_def, b1max ) :
                            yield ( (a, b1, b2, c), gcd(b1, gcd_acb2), disc_ac - b1**2 - D * b1 * b2 - DsqmD * b2**2)

                    else:
                        for b1 in xrange( b1min, b1max ) :
                            yield ( (a, b1, b2, c), gcd(b1, gcd_acb2), disc_ac - b1**2 - D * b1 * b2 - DsqmD * b2**2)

        else :
            raise NotImplementedError( "Iteration over non reduced forms." )

        raise StopIteration


    def iter_zero_diagonal_indefinite_forms_with_content_and_discriminant(self) :
        """
        Iterate over all non-positive definite integral hermitian quadratic
        forms with zero diagonal contained in the filter. The content and
        discriminant of the form will be given, too.
        

        NOTES:
            You may not iterate over infinite filters.
            The iteration over none reduced forms is not implemented yet.

        OUTPUT:
            - A generator over 3-tuples of a 4-tuple of integers and 2 integers.

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(2,D=-3,d=-5)
            sage: list(filter.iter_zero_diagonal_indefinite_forms_with_content_and_discriminant())
            [((0, -2, 0, 0), 2, -4), ((0, -1, 0, 0), 1, -1), ((0, -4, -2, 0), 2, -4), ((0, -3, -2, 0), 1, -3), ((0, -2, -2, 0), 2, -4), ((0, -3, -1, 0), 1, -3), ((0, -2, -1, 0), 1, -1), ((0, -1, -1, 0), 1, -1), ((0, 0, -1, 0), 1, -3)]



        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(infinity,D=-3,d=-1)
            sage: list(filter.iter_zero_diagonal_indefinite_forms_with_content_and_discriminant())
            Traceback (most recent call last):
            ...
            ArithmeticError: Cannot iterate over infinite filters.
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-3,d=-1,reduced=False)
            sage: list(filter.iter_zero_diagonal_indefinite_forms_with_content_and_discriminant())
            Traceback (most recent call last):
            ...
            NotImplementedError: Iteration over non reduced forms
        """
        if self.is_infinite() :
            raise ArithmeticError( "Cannot iterate over infinite filters." )

        D = self.__D
        d = self.__d
        DsqmD = (D**2 - D) // 4

        if self.__reduced :
            ## a = 0, c = 0, b2 = 0
            b1min = -(( isqrt( - 4 * (d+1))) // 2)

            for b1 in xrange(b1min, 0) :
                yield ( (0, b1, 0, 0), -b1, -b1**2)

            ## a = 0, c = 0, b2 < 0
            b2min = -isqrt((4*(d+1)) // D)

            for b2 in xrange(b2min, 0) :
                B2 = D * ( b2**2 ) - 4 * (d+1)
                b1min = -((D * b2 + isqrt(B2)) // 2)
                b1max = ((-D * b2 + isqrt(B2)) // 2) + 1

                for b1 in xrange( b1min, b1max ) :
                    yield ( (0, b1, b2, 0), gcd(b1, b2), - b1**2 - D * b1 * b2 - DsqmD * b2**2)

        else :
            raise NotImplementedError, "Iteration over non reduced forms"

        raise StopIteration

        
    def __cmp__(self, other) :
        """
        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-3,d=-1)
            sage: filter2 = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-3,d=-1)
            sage: filter == filter2
            True
            sage: filter2 = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-4,d=-1)
            sage: filter == filter2
            False
            sage: filter2 = HermitianModularFormD2Filter_diagonal_borcherds(5,D=-3,d=-1)
            sage: filter == filter2
            False
            sage: filter2 = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-3,d=-1,reduced=False)
            sage: filter == filter2
            False
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-4,d=-1,with_character=True,reduced=False)
            sage: filter2 = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-4,d=-1,with_character=True,reduced=False)
            sage: filter == filter2
            True
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-4,d=-1)
            sage: filter2 = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-4,d=-1,with_character=True)
            sage: filter == filter2
            False
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,3,D=-3,d=-1)
            sage: filter2 = HermitianModularFormD2Filter_diagonal_borcherds(4,3,D=-3,d=-1)
            sage: filter == filter2
            True
            sage: filter2 = HermitianModularFormD2Filter_diagonal_borcherds(4,4,D=-3,d=-1)
            sage: filter == filter2
            False
            sage: filter2 = HermitianModularFormD2Filter_diagonal_borcherds(3,4,D=-3,d=-1)
            sage: filter == filter2
            False
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-3,d=-1)
            sage: filter2 = HermitianModularFormD2Filter_diagonal_borcherds(4,D=-3,d=-2)
            sage: filter == filter2
            False
        """
        c = cmp(type(self), type(other))
        if c == 0 :
            c = cmp(self.__reduced, other.__reduced)
        if c == 0 :
            c = cmp(self.__with_character, other.__with_character)
        if c == 0 :
            c = cmp(self.__D, other.__D)
        if c == 0 :
            c = cmp(self.__bound_a, other.__bound_a)
        if c == 0 :
            c = cmp(self.__bound_c, other.__bound_c)
        if c == 0 :
            c = cmp(self.__d, other.__d)
            
        return c

    def __hash__(self) :
        """
        OUTPUT:
            - An integer

        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3)
            sage: hash(filter)
            121           # 64-bit
        """
        ## TODO: What should be returned here???
        return self.__D + 19 * hash(self.__with_character) + 31 * hash(self.__bound_a) + 37 * hash(self.__d)
                   
    def _repr_(self) :
        """
        OUTPUT:
            - A string

        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,3,D=-3,d=-1);filter
            Reduced diagonal Borcherds product filter for discriminant -3 with bounds 4 for a and 3 for c and discriminant bound -1
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,3,D=-4,d=-1,with_character=True,reduced=False);filter
            Diagonal Borcherds product filter for discriminant -4 respecting characters with bounds 8 for a and 6 for c and discriminant bound -4
        """
        return "%siagonal Borcherds product filter for discriminant %s%s with bounds %s for a and %s for c and discriminant bound %s" % \
               ( "Reduced d" if self.__reduced else "D", self.__D, 
                 " respecting characters" if self.__with_character else "",
                 self.__bound_a, self.__bound_c, self.__d )
    
    def _latex_(self) :
        """
        OUTPUT:
            - A string

        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,3,D=-3,d=-1)
            sage: filter._latex_()
            'Reduced diagonal Borcherds product filter for discriminant $-3$ with bound $4$ for $a$ and $3$ for $c$ and discriminant bound $-1$'
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(4,3,D=-4,d=-1,with_character=True,reduced=False)
            sage: filter._latex_()
            'Diagonal Borcherds product filter for discriminant $-4$ respecting characters with bound $8$ for $a$ and $6$ for $c$ and discriminant bound $-4$'
        """
        return "%siagonal Borcherds product filter for discriminant $%s$%s with bound $%s$ for $a$ and $%s$ for $c$ and discriminant bound $%s$" % \
               ( "Reduced d" if self.__reduced else "D", latex(self.__D), 
                 " respecting characters" if self.__with_character else "",
                 self.__bound_a, self.__bound_c, self.__d )

#===============================================================================
# HermitianModularFormD2Indices_diagonal
#===============================================================================

class HermitianModularFormD2Indices_diagonal_borcherds( HermitianModularFormD2Indices_diagonal ) :
    """
    This class implements the monoid of all hermitian binary quadratic
    forms for Borcherds prdoducts over the integers `o_{\Q(\sqrt{D})}`
    of the imaginary quadratic field with discriminant `D`.  The
    associated filters are given in
    :class:~`.HermitianModularFormD2Filter_diagonal_borcherds` and
    restrict the diagonal entries of such a form in matrix notation as
    well as its minimal (negative) discriminant.
    
    More precicely, the class can implement this monoid or the monoid equipped
    with the action of `\mathrm{GL}_{2}(o_{\Q(\sqrt{D})}`. In the latter case generators
    are given only in reduced form.
    
    It is also possible to consider quadratic forms occuring in the Fourier expansion of
    Hermitian modular forms for the full modular group with character. Note, that this character
    has order two. In this case (see argument with_character of :meth:~`.__init__`) the
    quadratic forms `T` in this monoid represent `T / 2` within the Fourier expansion.
    
    Elements of this monoid are stored as a 4-tuple `(a, b_1, b_2, c)` of integers. Each of them
    corresponds to the quadratic form `x,y \mapsto a x^2 + b x y + c y^2` with
    `b = b_1/\sqrt{D} + b_2 (1 + \sqrt{D}) \mathop{/} 2`.   
    """
    
    def __init__(self, D, with_character = False, reduced = True) :
        """
        INPUT:
            `D`                -- A negative integer; The discriminant of an
                                  imaginary quadratic fields.
            ``with_character`` -- A boolean (default: False); Whether the associated forms
                                  are associated to modular forms with character or not.
            ``reduced``        -- A boolean (default: True); Whether orbits of quadratic forms
                                  with respect to `\mathrm{GL}_{2}(o_{\Q(\sqrt{D})}` are considered.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-3, True)
            Traceback (most recent call last):
            ...
            ValueError: Characters are admissable only if 4 | D.
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-4, True)
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-7, False, False)
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-8, True, False)
        """
        if with_character and D % 4 != 0 :
            raise ValueError( "Characters are admissable only if 4 | D." )
        
        self.__D = D
        self.__with_character = with_character
        self.__reduced = reduced

    def ngens(self) :
        """
        OUTPUT:
            An integer.
        
        NOTE:
            The result is always mathematically wrong.

        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds.ngens()
            2
        """
        ## FIXME: Calculate the correct value whenever possible
        return 2
    
    def gen(self, i = 0) :
        """
        OUTPUT:
            A 4-tuple of integers.
            
        NOTE:
            Only the two generators with non-vanishing entries on the diagonal are returned.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds.gen(0)
            (1, 0, 0, 0)
            sage: inds.gen(4)
            Traceback (most recent call last):
            ...
            ValueError: The 4-th generator is not defined.
        """
        if i == 0 :
            return (1, 0,0, 0)
        elif i == 1 :
            return (0, 0, 0, 1)
        
        raise ValueError( "The %s-th generator is not defined." % (i,) )
    
    def gens(self) :
        """
        OUTPUT:
            A list of 4-tuples of integers.

        NOTE:
            One the two generators with non-vanishing entries on the diagonal are returned.

        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds.gens()
            [(1, 0, 0, 0), (0, 0, 0, 1)]
        """
        return [self.gen(i) for i in xrange(self.ngens())]

    def is_commutative(self) :
        """
        OUTPUT:
            A boolean.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds.is_commutative()
            True
        """
        return True

    def has_reduced_filters(self) :
        """
        Return whether ``self`` respects the action of `\mathrm{GL}_{2}(o_{\Q(\sqrt{D})}`.
        
        OUTPUT:
            A boolean.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds.has_reduced_filters()
            True
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-4, reduced = False)
            sage: inds.has_reduced_filters()
            False
        """
        return self.__reduced
    
    def has_filters_with_character(self) :
        """
        Return whether ``self`` represents a monoid of half-integral hermitian forms.
        
        OUTPUT:
            A boolean.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds.has_filters_with_character()
            False
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-4, True)
            sage: inds.has_filters_with_character()
            True
        """
        return self.__with_character
    
    def monoid(self) :
        """
        If ``self`` respects the action of `\mathrm{GL}_{2}(o_{\Q(\sqrt{D})}` return the underlying
        monoid without this action. Otherwise return a copy of ``self``.
        
        OUTPUT:
            An instance of :class:`~.HermitianModularFormD2Indices_diagonal_borcherds`.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds_a = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds_woa = HermitianModularFormD2Indices_diagonal_borcherds(-3, reduced = False)
            sage: inds_a.monoid() == inds_woa
            True
            sage: inds_woa.monoid() == inds_woa
            True
            sage: inds_woa.monoid() is inds_woa
            False
        """
        
        return HermitianModularFormD2Indices_diagonal_borcherds( self.__D,
                        with_character = self.__with_character, reduced = False ) 

    def group(self) :
        """
        If ``self`` respects the action of `\mathrm{GL}_{2}(o_{\Q(\sqrt{D})}`, return this group.
        
        OUTPUT:
            A string.
        
        NOTE:
            The return value may change later to the actual group.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds_a = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds_woa = HermitianModularFormD2Indices_diagonal_borcherds(-3, reduced = False)
            sage: inds_a.group()
            'GL(2,o_QQ(sqrt -3))'
            sage: inds_woa.group()
            Traceback (most recent call last):
            ...
            ArithmeticError: Monoid is not equipped with a group action.
        """
        if self.__reduced :
            return "GL(2,o_QQ(sqrt %s))" % (self.__D,)
        else :
            raise ArithmeticError( "Monoid is not equipped with a group action." )
        
    def is_monoid_action(self) :
        """
        In case ``self`` respects the action of a group, decide whether this action is a monoid action
        on the underlying monoid.
        
        OUTPUT:
            A boolean.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds_a = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds_woa = HermitianModularFormD2Indices_diagonal_borcherds(-3, reduced = False)
            sage: inds_a.is_monoid_action()
            True
            sage: inds_woa.is_monoid_action()
            Traceback (most recent call last):
            ...
            ArithmeticError: Monoid is not equipped with a group action.
        """
        if self.__reduced :
            return True
        else :
            raise ArithmeticError( "Monoid is not equipped with a group action." )
    
    def filter(self, bound_a, bound_c = None, d = None) :
        """
        Return a filter associated to this monoid of hermitian quadratic forms with given bounds for Borcherds products.
        
        INPUT:
            ``bound_a``  -- An integer or an instance of
                            :class:`~.HermitianModularFormD2Filter_Borcherds_diagonal`;
                            A bound on the diagonal element a of the quadratic forms.
            ``bound_c``  -- An integer or None
                            A bound on the diagonal element c of the quadratic forms.
                            if bound_c is None, bound_a is chosen as a bound for c
            `d`          -- A negative Rational;
                            A lower bound for the discriminants of the elemets of the filter
        
        OUTPUT:
            An instance of :class:`~.HermitianModularFormD2Filter_Borcherds_diagonal`.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Filter_diagonal_borcherds
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds.filter(4,3,-1) == HermitianModularFormD2Filter_diagonal_borcherds(4,3,-1,-3)
            True
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-7, reduced = False)
            sage: inds.filter(3,3,-2) == HermitianModularFormD2Filter_diagonal_borcherds(3,3,-2,-7,reduced = False)
            True
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-4, True) 
            sage: inds.filter(4,3,-1) == HermitianModularFormD2Filter_diagonal_borcherds(4,3,-1,-4,True)
            True
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-8, True, reduced = False)
            sage: inds.filter(3,3,-2) == HermitianModularFormD2Filter_diagonal_borcherds(3,3,-2,-8,True,reduced = False)
            True
            sage: filter = HermitianModularFormD2Filter_diagonal_borcherds(3,3,-2,-8,True, reduced = False)
            sage: inds.filter(filter) == filter
            True
        """
        return HermitianModularFormD2Filter_diagonal_borcherds( bound_a, bound_c, d, self.__D,
                 with_character = self.__with_character, reduced = self.__reduced )
        
    def filter_all(self, d = -1) :
        """
        Return the filter associated to this monoid of quadratic forms which contains all
        elements.
        
        INPUT:
            `d` -- A negative Rational; A lower bound for the discriminants of the elemets of the filter

        OUTPUT:
            An instance of :class:`~HermitianModularFormD2Filter_Borcherds_diagonal`.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds.filter_all(-1).is_all()
            True
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-8, True, reduced = False)
            sage: fil = inds.filter_all(-1)
            sage: fil.is_with_character()
            True
            sage: fil.is_reduced()
            False
        """
        return HermitianModularFormD2Filter_diagonal_borcherds( infinity, infinity, d, self.__D,
                 with_character = self.__with_character, reduced = self.__reduced )
    
    def minimal_composition_filter(self, ls, rs) :
        """
        Given two lists `ls` and `rs` of hermitian quadratic forms return a filter that contains
        all the sums `l + r` of elements `l \in ls,\, r \in rs`.
        
        INPUT:
            `ls`  -- A list of 4-tuples of integers. Each element represents a quadratic form.
            `rs`  -- A list of 4-tuples of integers. Each element represents a quadratic form.
    
        OUTPUT:
            An instance of :class:`~HermitianModularFormD2Filter_Borcherds_diagonal`.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds.minimal_composition_filter([], []).index()
            (0, 0)
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-4, True, False)
            sage: fil = inds.minimal_composition_filter([], [])
            sage: fil.is_with_character()
            True
            sage: fil.is_reduced()
            False
            sage: inds.minimal_composition_filter([], [(1,0,0,1)]).index()
            (0, 0)
            sage: inds.minimal_composition_filter([(1,0,0,1)], []).index()
            (0, 0)
            sage: inds.minimal_composition_filter([(1,0,0,1)], [(1,0,0,1)]).index()
            (4, 4)
            sage: inds.minimal_composition_filter([(1,1,0,1)], [(1,0,0,d) for d in range(6)]).index()
            (4, 8)
            sage: inds.minimal_composition_filter([(d,1,0,1) for d in range(10)], [(1,0,0,d) for d in range(6)]).index()
            (12, 8)
            sage: inds = HermitianModularFormD2Indices_diagonal(-4)
            sage: inds.minimal_composition_filter([(1,0,0,1)], [(1,0,0,1)]).index()
            3
        """
        if len(ls) == 0 or len(rs) == 0 :
            return HermitianModularFormD2Filter_diagonal_borcherds( 0,0,-1, self.__D,
                                   with_character = self.__with_character,
                                   reduced = self.__reduced )
        
        bound_a = max(0, max(a for (a,_,_,_) in ls)) + max(0, max(a for (a,_,_,_) in rs)) + 1
        bound_c = max(0, max(c for (_,_,_,c) in ls)) + max(0, max(c for (_,_,_,c) in rs)) + 1
        d = min(min(- self.__D * (a1+a2) * (c1+c2) - (b11+b21)**2 - self.__D * (b11+b21) * (b12+b22) - (self.__D**2 - self.__D) // 4 * (b12+b22)**2 for (a1,b11,b12,c1) in ls for (a2,b21,b22,c2) in rs),-1)
        if self.__with_character :
            bound_a = -(-bound_a // 2)
            bound_c = -(-bound_c // 2)
            
        return HermitianModularFormD2Filter_diagonal_borcherds( bound_a, bound_c, d, self.__D,
                               with_character = self.__with_character,
                               reduced = self.__reduced )

    def _reduction_function(self) :
        """
        In case ``self`` respects the action of `\mathrm{GL}_{2}(o_{\Q(\sqrt{D})}`, return the
        reduction funtion for elements of this monoid.
        
        SEE::
            :meth:`~.reduce`
        
        OUTPUT:
            A function accepting one argument.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds._reduction_function() == inds.reduce
            True
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-4, reduced = False)
            sage: inds._reduction_function()
            Traceback (most recent call last):
            ...
            ArithmeticError: Monoid is not equipped with a group action.
        """
        if self.__reduced :
            return self.reduce
        else :
            raise ArithmeticError( "Monoid is not equipped with a group action." )
    
    def reduce(self, s) :
        """
        Reduce a hermitian binary quadratic form `s`.
        
        INPUT:
            - `s` -- A 4-tuple of integers; This tuple represents a quadratic form.
        
        OUTPUT:
            A pair `(s, (\mathrm{trans}, \mathrm{det}, \nu))` of a quadratic form `s`,
            which is represented by a four-tuple, and character evaluations, with
            `\nu` and `\mathrm{trans}` either `1` or `-1` and `\mathrm{det}` between
            `0` and `|\mathfrak{o}_{\Q(\sqrt{D})}^\times|`.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds.reduce((1,0,0,1))
            ((1, 0, 0, 1), (1, 0, 1))
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-4, True)
            sage: inds.reduce((2,0,0,2))
            Traceback (most recent call last):
            ...
            NotImplementedError
        """
        if not self.__with_character :
            (a, b1, b2, c) = s
            D = self.__D
            if a < 0 or c < 0 or - D * a * c - b1**2 - D * b1 * b2 - (D**2 - D) // 4 * b2**2 < 0 :
                return ( (a, b1, b2, c), (1, 0, 1) )
            else :
                (s, chs) = reduce_GL(s, self.__D)
                if chs[0] == 1 :
                    return (s, chs)
                
                (a, b1, b2, c) = s
                ## apply transposition
                b1 = -b1 + 4 * b2
            
                return ((a, b1, b2, c), (1, chs[1], chs[2]))
  
        raise NotImplementedError

    def _decompositions_with_stepsize(self, s, stepsize, offset, d1, d2) :
        """
        Decompose a quadratic form `s = s_1 + s_2` in all possible ways. Writting
        `s_1 = (a_1, b_1, c_1)`, the restriction `a_1 > offset` is imposed and
        moreover all entires vary by multiples of ``stepsize``.
        
        INPUT:
            `s`          -- A 4-tuple of integers; Represents a quadratic form.
            ``stepsize`` -- An integer which is either 1 or 2; The stepsize for `a_1`, `b_{1,1}`,
                           `b_{2,1}` and `c_1`.
            ``offset``   -- A integer which is either 0 or 1; An offset for `a_1` and `c_1`.
                            It also determines the class modulo ``stepsize`` of `b_{2,1}`.
            `d1`         -- A negative Rational; the discriminant bound of s1
            `d2`         -- A negative Rational; the discriminant bound of s2
        
        OUTPUT:
            A generator interating over pairs of 4-tuples of integers.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: len(list(inds._decompositions_with_stepsize((1,0,0,1), 1, 0, -1, -1)))
            34
            sage: ((0,0,0,0), (4,0,0,4)) in inds._decompositions_with_stepsize((4,0,0,4), 2, 1, -1, -1)
            False
            sage: list(inds._decompositions_with_stepsize((2,0,0,2), 2, 0, -1, -1))                    
            [((0, 0, 0, 0), (2, 0, 0, 2)), ((0, 0, 0, 2), (2, 0, 0, 0)), ((2, 0, 0, 0), (0, 0, 0, 2)), ((2, 0, 0, 2), (0, 0, 0, 0))]
        """
        (a, b1, b2, c) = s

        D = self.__D

        for a1 in xrange(offset, a + 1, stepsize) :
            a2 = a - a1
            for c1 in xrange(offset, c + 1, stepsize) :
                c2 = c - c1
                
                ## the positive semi definite case
                B1 = isqrt(4*a1*c1)
                B2 = isqrt(4*a2*c2)
                
                b21min = max(-B1, b2 - B2)
                if stepsize == 2 :
                    if offset == 0 :
                        b21min = b21min + (b21min % 2)
                    else :
                        b21min = b21min + 1 - (b21min % 2)
                for b21 in xrange(b21min, min(B1 + 1, b2 + B2 + 1), stepsize) :
                    b22 = b2 - b21
                    
                    
                    B3 = D * (b21**2 - 4 * a1 * c1)
                    b11min_def = -((D * b21 + isqrt(B3)) // 2)
                    b11max_def = (-D * b21 + isqrt(B3)) // 2
                                        
                    B4 = D * (b22**2 - 4 * a2 * c2)
                    b12min_def = -((D * b22 + isqrt(B4)) // 2)
                    b12max_def = (-D * b22 + isqrt(B4)) // 2
                    
                    b11min = max(b11min_def, b1 - b12max_def)
                    if stepsize == 2 :
                        if offset == 0 :
                            b11min = b11min + (b11min % 2)
                        else :
                            b11min = b11min + 1 - (b11min % 2)
                    for b11 in xrange(b11min, min(b11max_def, b1 - b12min_def) + 1, stepsize ) :
                        yield ((a1, b11, b21, c1), (a2,b1 - b11,b22, c2))

                ## the negative case
                ## TODO: what kind of decomposition is needed here? the one following is not correct!
                b21min = max(-isqrt(-4*d1 + 4*a1*c1), b2 - isqrt(-4*d2 + 4*a2*c2))
                b21max = min(isqrt(-4*d1 + 4*a1*c1) + 1, b2 + isqrt(-4*d2 + 4*a2*c2) + 1)
                if stepsize == 2 :
                    if offset == 0 :
                        b21min = b21min + (b21min % 2)
                    else :
                        b21min = b21min + 1 - (b21min % 2)
                for b21 in xrange(b21min, b21max, stepsize) :
                    b22 = b2 - b21
                    B21 = D * ( b21**2 + 4 * d1 - 4 * a1 * c1)
                    B22 = D * ( b22**2 + 4 * d2 - 4 * a2 * c2)
                    b11min_temp = (-D * b21 - sqrt(B21)) / 2
                    if b11min_temp > 0 :
                        b11min_temp = isqrt(b11min_temp**2)
                    else :
                        b11min_temp = -isqrt(b11min_temp**2)
                    b11max_temp = (-D * b21 + sqrt(B21)) / 2
                    if b11max_temp > 0 :
                        b11max_temp = isqrt(b11max_temp**2) + 1
                    else :
                        b11max_temp = -isqrt(b11max_temp**2) + 1
                    b12min_temp = (-D * b22 - sqrt(B22)) / 2
                    if b12min_temp > 0 :
                        b12min_temp = isqrt(b12min_temp**2)
                    else :
                        b12min_temp = -isqrt(b12min_temp**2)
                    b12max_temp = (-D * b22 + sqrt(B22)) / 2
                    if b12max_temp > 0 :
                        b12max_temp = isqrt(b12max_temp**2) + 1
                    else :
                        b12max_temp = -isqrt(b12max_temp**2) + 1
                    b11min = max(b11min_temp, b1 - b12max_temp)
                    b11max = min(b11max_temp + 1, b1 - b12min_temp + 1)
                    if stepsize == 2 :
                        if offset == 0 :
                            b11min = b11min + (b11min % 2)
                        else : 
                            b11min = b11min + 1 - (b11min % 2)
                    B31 = D * (b21**2 - 4 * a1 * c1)
                    B32 = D * (b22**2 - 4 * a2 * c2)
                    if B31 >= 0 :
                        b11min_def_temp = (-D * b21 - sqrt(B31)) / 2
                        if b11min_def_temp > 0 :
                            b11min_def_temp = isqrt(b11min_def_temp**2)
                        else :
                            b11min_def_temp = -isqrt(b11min_def_temp**2)
                        b11max_def_temp = (-D * b21 + sqrt(B31)) / 2
                        if b11max_def_temp > 0 :
                            b11max_def_temp = isqrt(b11max_def_temp**2)
                        else :
                            b11max_def_temp = -isqrt(b11max_def_temp**2)
                        if B32 >= 0 :
                            b12min_def_temp = (-D * b22 - sqrt(B32)) / 2
                            if b12min_def_temp > 0 :
                                b12min_def_temp = isqrt(b12min_def_temp**2)
                            else :
                                b12min_def_temp = -isqrt(b12min_def_temp**2)
                            b12max_def_temp = (-D * b22 + sqrt(B32)) / 2
                            if b12max_def_temp > 0 :
                                b12max_def_temp = isqrt(b12max_def_temp**2)
                            else :
                                b12max_def_temp = -isqrt(b12max_def_temp**2)
                            b11min_def = min(b11min_def_temp, b1 - b12max_def_temp)
                            b11max_def = max(b11max_def_temp + 1, b1 - b12min_def_temp + 1)
                            if stepsize == 2 :
                                if offset == 0 :
                                    b11max_def = b11max_def + (b11max_def % 2)
                                else : 
                                    b11max_def = b11max_def + 1 - (b11max_def % 2)
                            for b11 in xrange( b11min, b11min_def, stepsize ) :
                                yield ((a1, b11, b21, c1), (a2,b1 - b11,b22, c2))
                            for b11 in xrange( b11max_def, b11max, stepsize ) :
                                yield ((a1, b11, b21, c1), (a2,b1 - b11,b22, c2))
                        else :
                            b11min_def = b11min_def_temp
                            b11max_def = b11max_def_temp + 1
                            if stepsize == 2 :
                                if offset == 0 :
                                    b11max_def = b11max_def + (b11max_def % 2)
                                else : 
                                    b11max_def = b11max_def + 1 - (b11max_def % 2)
                            for b11 in xrange( b11min, b11min_def, stepsize ) :
                                yield ((a1, b11, b21, c1), (a2,b1 - b11,b22, c2))
                            for b11 in xrange( b11max_def, b11max, stepsize ) :
                                yield ((a1, b11, b21, c1), (a2,b1 - b11,b22, c2))
                    else :
                        if B32 >= 0 :
                            b12min_def_temp = (-D * b22 - sqrt(B32)) / 2
                            if b12min_def_temp > 0 :
                                b12min_def_temp = isqrt(b12min_def_temp**2)
                            else :
                                b12min_def_temp = -isqrt(b12min_def_temp**2)
                            b12max_def_temp = (-D * b22 + sqrt(B32)) / 2
                            if b12max_def_temp > 0 :
                                b12max_def_temp = isqrt(b12max_def_temp**2)
                            else :
                                b12max_def_temp = -isqrt(b12max_def_temp**2)
                            b11min_def = b1 - b12max_def_temp
                            b11max_def = b1 - b12min_def_temp + 1
                            if stepsize == 2 :
                                if offset == 0 :
                                    b11max_def = b11max_def + (b11max_def % 2)
                                else : 
                                    b11max_def = b11max_def + 1 - (b11max_def % 2)
                            for b11 in xrange( b11min, b11min_def, stepsize ) :
                                yield ((a1, b11, b21, c1), (a2,b1 - b11,b22, c2))
                            for b11 in xrange( b11max_def, b11max, stepsize ) :
                                yield ((a1, b11, b21, c1), (a2,b1 - b11,b22, c2))
                        else :
                            for b11 in xrange(b11min, b11max, stepsize ) :
                                yield ((a1, b11, b21, c1), (a2,b1 - b11,b22, c2))
        
        raise StopIteration
    
    def decompositions(self, s, signature, d1 ,d2) :
        """
        Decompose a quadratic form `s = s_1 + s_2` in all possible ways. If ``signature``
        is non-zero and this monoid is for characters, the iteration is restricted to all
        even `s_1` if ``signature`` is `1` and to all odd `s_1` if it is `-1`. 
        
        INPUT:
            `s`           -- A 4-tuple of integers; Represents a quadratic form.
            ``signature`` -- An integer `-1`, `0` or `1`; Determines the oddness of `s_1`.
            `d1`         -- A negative Rational; the determinant bound of s1
            `d2`         -- A negative Rational; the determinant bound of s2

        OUTPUT:
            A generator interating over pairs of 4-tuples of integers.
            
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds_a = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds_woa = HermitianModularFormD2Indices_diagonal_borcherds(-3, reduced = False)
            sage: len(list(inds_a.decompositions((2,1,0,2),0,-1,-1)))
            62
            sage: list(inds_a.decompositions((2,1,0,2),0,-1,-1)) == list(inds_woa.decompositions((2,1,0,2),0,-1,-1))
            True
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-4, True)
            sage: list(inds.decompositions((3,0,0,3), 1,-1,-1)) + list(inds.decompositions((3,0,0,3), -1,-1,-1)) == list(inds.decompositions((3,0,0,3),0,-1,-1))
            True
            sage: inds.decompositions((3,0,0,3), 2,-1,-1)
            Traceback (most recent call last):
            ...
            ValueError: Signature must be -1, 0 or 1.
        """
        if signature is None :
            signature = 0
        if not self.__with_character :
            return self._decompositions_with_stepsize(s, 1, 0, d1 ,d2)
        else :
            if signature == 0 :
                return itertools.chain(self._decompositions_with_stepsize(s, 2, 0, d1 ,d2),
                                       self._decompositions_with_stepsize(s, 2, 1, d1 ,d2))
            elif signature == 1 :
                return itertools.chain(self._decompositions_with_stepsize(s, 2, 0, d1 ,d2))
            elif signature == -1 :
                return itertools.chain(self._decompositions_with_stepsize(s, 2, 1, d1 ,d2))
            else :
                raise ValueError( "Signature must be -1, 0 or 1." )
        
        raise StopIteration
    
    def zero_element(self) :
        """
        Return the zero element of this monoid.
        
        OUTPUT:
            A 4-tuple of integers.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds.zero_element()
            (0, 0, 0, 0)
        """
        return (0,0,0,0)

    def __cmp__(self, other) :
        """
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: inds = HermitianModularFormD2Indices_diagonal_borcherds(-3)
            sage: inds == 2
            False 
            sage: inds == HermitianModularFormD2Indices_diagonal_borcherds(-4)
            False
            sage: inds == HermitianModularFormD2Indices_diagonal_borcherds(-4, True)
            False
            sage: inds == HermitianModularFormD2Indices_diagonal_borcherds(-3, reduced = False)
            False
        """
        c = cmp(type(self), type(other))
        if c == 0 :
            c = cmp(self.__reduced, other.__reduced)
        if c == 0 :
            c = cmp(self.__with_character, other.__with_character)
        if c == 0 :
            c = cmp(self.__D, other.__D)
            
        return c
    
    def __hash__(self) :
        """
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: hash( HermitianModularFormD2Indices_diagonal_borcherds(-3) )
            16
            sage: hash( HermitianModularFormD2Indices_diagonal_borcherds(-4, True) )
            28
            sage: hash( HermitianModularFormD2Indices_diagonal_borcherds(-7, reduced = False) )
            -7
            sage: hash( HermitianModularFormD2Indices_diagonal_borcherds(-8, True, False) )
            5
        """
        ##TODO: How should hash be computed?
        return hash(self.__D) + 13 * hash(self.__with_character) + 19 * hash(self.__reduced)
    
    def _repr_(self) :
        """
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: repr( HermitianModularFormD2Indices_diagonal_borcherds(-3) )
            'Reduced quadratic forms for Borcherds products over o_QQ(\\sqrt -3)'
            sage: repr( HermitianModularFormD2Indices_diagonal_borcherds(-4, True) )
            'Reduced quadratic forms for Borcherds products over o_QQ(\\sqrt -4) for character forms'
            sage: repr( HermitianModularFormD2Indices_diagonal_borcherds(-7, reduced = False) )
            'Non-reduced quadratic forms for Borcherds products over o_QQ(\\sqrt -7)'
            sage: repr( HermitianModularFormD2Indices_diagonal_borcherds(-8, True, False) )
            'Non-reduced quadratic forms for Borcherds products over o_QQ(\\sqrt -8) for character forms'
        """
        return "%seduced quadratic forms for Borcherds products over o_QQ(\sqrt %s)%s" % \
          ( "R" if self.__reduced else "Non-r", self.__D,
            " for character forms" if self.__with_character else "" )
            
    def _latex_(self) :
        """
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
            sage: latex( HermitianModularFormD2Indices_diagonal_borcherds(-3) )
            Reduced quadratic forms for Borcherds products over $\mathfrak{o}_{\mathbb{Q}(\sqrt{-3})}$
            sage: latex( HermitianModularFormD2Indices_diagonal_borcherds(-4, True) )
            Reduced quadratic forms for Borcherds products over $\mathfrak{o}_{\mathbb{Q}(\sqrt{-4})}$ for character forms
            sage: latex( HermitianModularFormD2Indices_diagonal_borcherds(-7, reduced = False) )
            Non-reduced quadratic forms for Borcherds products over $\mathfrak{o}_{\mathbb{Q}(\sqrt{-7})}$
            sage: latex( HermitianModularFormD2Indices_diagonal_borcherds(-8, True, False) )
            Non-reduced quadratic forms for Borcherds products over $\mathfrak{o}_{\mathbb{Q}(\sqrt{-8})}$ for character forms
        """
        return "%seduced quadratic forms for Borcherds products over $\mathfrak{o}_{\mathbb{Q}(\sqrt{%s})}$%s" % \
          ( "R" if self.__reduced else "Non-r", latex(self.__D),
            " for character forms" if self.__with_character else "" )

#===============================================================================
# HermitianModularFormD2FourierExpansionRing
#===============================================================================

def HermitianModularFormD2FourierExpansionRing_borcherds(K, D, with_nu_character = False) :
    """
    Return the ring of Fourier expansions of Hermitian modular forms with or without characters
    over a ring `K` associated to the full hermitian modular group for `\QQ(\sqrt{D})`.
    
    INPUT:
        `K`    -- A ring; The ring of coefficients for the Fourier expansion.
        `D`    -- A negative integer; A discriminant of an imaginary quadratic field.
        ``with_nu_character`` -- A boolean; Whether include forms with character `\nu`
                                 or not.
    
    OUTPUT:
        An instance of :class:`~fourier_expansion_framework.monoidpowerseries.monoidpowerseries_ring.EquivariantMonoidPowerSeriesRing_generic`.
    
    TESTS::
        sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2Indices_diagonal_borcherds
        sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_borcherdsproducts_fourierexpansion import HermitianModularFormD2FourierExpansionRing_borcherds
        sage: fe_ring = HermitianModularFormD2FourierExpansionRing_borcherds(ZZ, -3)
        sage: fe_ring.action() == HermitianModularFormD2Indices_diagonal_borcherds(-3)
        True
        sage: fe_ring.base_ring()
        Integer Ring
        sage: fe_ring.coefficient_domain()
        Integer Ring
        sage: fe_ring.characters()
        Character monoid over Multiplicative Abelian Group isomorphic to C2 x C2
        sage: fe_ring.representation()
        Trivial representation of GL(2,o_QQ(sqrt -3)) on Integer Ring
        sage: fe_ring = HermitianModularFormD2FourierExpansionRing_borcherds(QQ, -4, True)
        sage: fe_ring.action() == HermitianModularFormD2Indices_diagonal_borcherds(-4, True)
        True
    """
    R = EquivariantMonoidPowerSeriesRing(
         HermitianModularFormD2Indices_diagonal_borcherds(D, with_nu_character),
         HermitianModularFormD2FourierExpansionCharacterMonoid(D, ZZ),
         TrivialRepresentation("GL(2,o_QQ(sqrt %s))" % (D,), K))
        
    return R
