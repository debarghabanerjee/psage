"""
Functions for treating the fourier expansion of hermitian modular forms. 

AUTHORS:

- Martin Raum (2009 - 08 - 31) Initial version.

- Dominic Gehre (2010 - 05 - 12) Change iterations.

- Martin Raum (2011 - 05 - 30) Implement multibound filters.
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

from psage.modform.fourier_expansion_framework.monoidpowerseries.monoidpowerseries_basicmonoids import CharacterMonoidElement_class
from psage.modform.fourier_expansion_framework.monoidpowerseries.monoidpowerseries_basicmonoids import TrivialCharacterMonoid,\
    TrivialRepresentation, CharacterMonoid_class
from psage.modform.fourier_expansion_framework.monoidpowerseries.monoidpowerseries_ring import EquivariantMonoidPowerSeriesRing
from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion_cython import reduce_GL
from sage.groups.all import AbelianGroup
from sage.misc.functional import isqrt
from sage.misc.latex import latex
from sage.rings.arith import gcd
from sage.rings.infinity import infinity
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.structure.sage_object import SageObject
import itertools

#===============================================================================
# HermitianModularFormD2Filter_diagonal
#===============================================================================

class HermitianModularFormD2Filter_diagonal ( SageObject ) :
    r"""
    This class implements a filter on the monoid of positive semi-definite
    integral hermitian quadrativ forms `[a, b, c]` in Gauss notation over
    `\mathbb{Q}(\sqrt{D})`, an imaginary quadratic field with discriminant
    `D`. It implements a bound on the diagonal entries in matrix notation,
    namely `a, c < \textrm{bound}`.
    The main aspect of this class is to give an iteration over all reduced
    (or non-reduced, which isn't implemented) indefinite and positive definite
    hermitian quadratic forms up to a given bound of the diagonal.
    These iterations are needed to work with hermitian modular forms using their
    fourier expansions.
    By initiating an object of this class, most importantly the said bound and the
    dicriminant (of the imaginary quadratic number field in consideration) are
    specified. Furthermore it is possible to consider a filter for non-trivial
    characters.
    """

    ## TODO: Update documentation to new filter and add doctests.
    def __init__(self, bound, D = None, with_character = None, reduced = None) :
        """
        INPUT:
            - ``bound``           -- A filter, a nonnegative integer or a pair of nonnegative integers;
                                     Bound for the diagonal.
            - `D`                 -- Negative integer; A discriminant of an imaginary quadratic field.
                                     Illegal if ``bound`` is a filter.
            - ``with_character``  -- Boolean (optional: default is False); With character `\nu` or not.
                                     Illegal if ``bound`` is a filter.
            - ``reduced`` --         Boolean (optional: default is True); Reduced filters only iterate
                                     over reduced hermitian quadratic forms.

        EXAMPLES::
            sage: HermitianModularFormD2Filter_diagonal(4,-3)
            Reduced diagonal filter for discriminant -3 with bound 4
            sage: HermitianModularFormD2Filter_diagonal(4,-4,True,False)
            Diagonal filter for discriminant -4 respecting characters with bound 8

        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3)
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4,True)
            sage: filter = HermitianModularFormD2Filter_diagonal((1, 3),-4,True)
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4,True,False)
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3,reduced=False)
            sage: filter = HermitianModularFormD2Filter_diagonal((4, 4), -3)
            sage: filter = HermitianModularFormD2Filter_diagonal((4, 2), -3)
            Traceback (most recent call last):
            ...
            ValueError: If two bounds are given, the first must be less than or equal to the second.
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3,True)
            Traceback (most recent call last):
            ...
            ValueError: Characters are admissable only if 4 | D.
            sage: HermitianModularFormD2Filter_diagonal(4)
            Traceback (most recent call last):
            ...
            TypeError: If bound is not a filter, D has to be assigned.
            sage: HermitianModularFormD2Filter_diagonal(0.5,-3)
            Traceback (most recent call last):
            ...
            TypeError: If bound is neither a filter nor of list type, bound has to be an integer or infinity.
            sage: HermitianModularFormD2Filter_diagonal(-1,-3)
            Traceback (most recent call last):
            ...
            ValueError: Integer bounds must be non-negative.
            sage: HermitianModularFormD2Filter_diagonal((-1, 3),-3)
            Traceback (most recent call last):
            ...
            ValueError: All integer bounds must be non-negative.
            sage: HermitianModularFormD2Filter_diagonal((3, -1),-3)
            Traceback (most recent call last):
            ...
            ValueError: All integer bounds must be non-negative.
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3); filter
            Reduced diagonal filter for discriminant -3 with bound 4
            sage: HermitianModularFormD2Filter_diagonal(filter,reduced=False)
            Diagonal filter for discriminant -3 with bound 4
            sage: HermitianModularFormD2Filter_diagonal(filter,-4)
            Traceback (most recent call last):
            ...
            ValueError: D cannot be reassigned.
            sage: HermitianModularFormD2Filter_diagonal(filter,with_character=True)
            Traceback (most recent call last):
            ...
            ValueError: with_character cannot be reassigned.
        """
        if isinstance(bound, HermitianModularFormD2Filter_diagonal) :
            if not D is None and D != bound.discriminant() :
                raise ValueError( "D cannot be reassigned." )
            if not with_character is None and with_character != bound.is_with_character() :
                raise ValueError( "with_character cannot be reassigned." )
            
            self.__bound = bound.index()
            self.__has_multi_bound = isinstance(self.__bound, tuple)
            self.__D = bound.D()
            self.__with_character = bound.is_with_character()
                
            if reduced is None :
                self.__reduced = bound.is_reduced()
            else :
                self.__reduced = reduced
        else :
            if D is None :
                raise TypeError( "If bound is not a filter, D has to be assigned." )
            
            if isinstance(bound, (tuple, list)) :
                if len(bound) != 2 or \
                   not all(isinstance(b, (int, Integer)) or b is infinity for b in bound) :
                    raise TypeError( "If bound is of list type, it must have length 2 with entries integers or infinity." )
                if any(b < 0 for b in bound) :
                    raise ValueError( "All integer bounds must be non-negative." )
                if bound[0] > bound[1] :
                    raise ValueError( "If two bounds are given, the first must be less than or equal to the second." )
            else :
                if not isinstance(bound,(int,Integer)) and not bound is infinity :
                    raise TypeError( "If bound is neither a filter nor of list type, bound has to be an integer or infinity." )
                if bound < 0 :
                    raise ValueError( "Integer bounds must be non-negative." )
                
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
                if isinstance(bound, (tuple, list) ) :
                    self.__bound = tuple(bound)
                    self.__has_multi_bound = True
                else :
                    self.__bound = bound
                    self.__has_multi_bound = False
            else :
                if D % 4 != 0 or D // 4 % 4 not in [2,3] :
                    raise ValueError( "Characters are admissable only if 4 | D." )

                if isinstance(bound, (tuple, list)) :
                    self.__bound = tuple(2 * b for b in bound)
                    self.__has_multi_bound = True
                else :
                    self.__bound = 2*bound
                    self.__has_multi_bound = False
                    
            self.__D = D
            
    def is_infinite(self) :
        """
        Returns whether the filter contains infinitely many elements or not.

        OUTPUT:
            - A boolean

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3)
            sage: filter.is_infinite()
            False
            sage: filter = HermitianModularFormD2Filter_diagonal(infinity,-3)
            sage: filter.is_infinite()
            True
            sage: filter = HermitianModularFormD2Filter_diagonal((2, infinity),-3)
            sage: filter.is_infinite()
            True
            sage: filter = HermitianModularFormD2Filter_diagonal((infinity, infinity),-3)
            sage: filter.is_infinite()
            True
        """
        return self.__bound is infinity or \
               ( self.__has_multi_bound and self.__bound[1] is infinity ) 
    
    def is_all(self) :
        """
        Returns whether the filter contains all elements or not.

        OUTPUT:
            - A boolean

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3)
            sage: filter.is_all()
            False
            sage: filter = HermitianModularFormD2Filter_diagonal(infinity,-3)
            sage: filter.is_all()
            True
            sage: filter = HermitianModularFormD2Filter_diagonal((2, infinity),-3)
            sage: filter.is_all()
            False
            sage: filter = HermitianModularFormD2Filter_diagonal((infinity, infinity),-3)
            sage: filter.is_all()
            True
        """
        return self.__bound is infinity or \
               ( self.__has_multi_bound and self.__bound[0] is infinity ) 

    def index(self) :
        """
        Return the vitual index, namely if the filter respects characters,
        return twice the bound, since we save twice the Fourier indices.

        OUTPUT:
            - a positive integer

        EXAMPLES::
        
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4)
            sage: filter.index()
            4
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4,with_character=True)
            sage: filter.index()
            8
            sage: filter = HermitianModularFormD2Filter_diagonal((2, 5),-3)
            sage: filter.index()
            (2, 5)

        TESTS::
        
            sage: filter = HermitianModularFormD2Filter_diagonal(infinity,-4)
            sage: filter.index()
            +Infinity
        """
        return self.__bound

    def _enveloping_content_bound(self) :
        """
        Return a bound such that any semi-definite element of ``self`` has at most
        content. The bound must not be attained.

        OUTPUT:
            - a positive integer

        EXAMPLES::
        
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4)
            sage: filter._enveloping_content_bound()
            4
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4,with_character=True)
            sage: filter._enveloping_content_bound()
            8
            sage: filter = HermitianModularFormD2Filter_diagonal((2, 5),-3)
            sage: filter._enveloping_content_bound()
            2
            sage: filter = HermitianModularFormD2Filter_diagonal((7, infinity),-3)
            sage: filter._enveloping_content_bound()
            7

        TESTS::
        
            sage: filter = HermitianModularFormD2Filter_diagonal(infinity,-4)
            sage: filter._enveloping_content_bound()
            +Infinity
            sage: filter = HermitianModularFormD2Filter_diagonal((infinity, infinity),-4)
            sage: filter._enveloping_content_bound()
            +Infinity
        """
        return self.__bound[0] if self.__has_multi_bound else self.__bound
    
    def _enveloping_discriminant_bound(self) :
        """
        Returns an enveloping discriminant bound. Namely a maximal discriminant
        for elements in this filter. The discriminant is D * det T for an index T.

        NOTES:
            The vitual bound for the diagonal of T is twice the bound if the filter
            respects characters, since we save twice the Fourier indices.

        OUTPUT:
            - a positive integer

        EXAMPLES::
        
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4)
            sage: filter._enveloping_discriminant_bound()
            37
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4,with_character=True)
            sage: filter._enveloping_discriminant_bound()
            197
            sage: filter = HermitianModularFormD2Filter_diagonal((2, 4),-3)
            sage: filter._enveloping_discriminant_bound()
            10
        """
        return -self.__D * (self.__bound[0] - 1) * (self.__bound[1] - 1) + 1 \
               if self.__has_multi_bound \
               else -self.__D * (self.__bound - 1)**2 + 1
    
    def discriminant(self) :
        """
        Returns the discriminant of the imaginary quadratic field associated to ``self``.

        OUTPUT:
            - A negative integer

        EXAMPLES::
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3)
            sage: filter.discriminant()
            -3
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
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4)
            sage: filter.is_with_character()
            False
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4,with_character=True)
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
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4)
            sage: filter.is_reduced()
            True
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4,reduced=False)
            sage: filter.is_reduced()
            False
        """
        return self.__reduced
    
    def __contains__(self, f) :
        """
        Returns whether the filter contains `f=[a,b,c]` with `b=b_1/\sqrt{D}+b2(1+\sqrt{D})/2`
        or not. That is, if the bound is infinite, the filter always contains `f`. Otherwise,
        the filter contains `f` if and only if `a < \textrm{vitual bound}` and
        `c < \textrm{vitual bound}`. Note that `a` and `c` are the diagonal of the matrix
        which represents the hermitian quadratic forms `f`.

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
            sage: filter = HermitianModularFormD2Filter_diagonal((2, 4),-4)
            sage: filter.__contains__((1,0,0,3))
            True
            sage: filter.__contains__((3,0,0,1))
            False
            sage: filter.__contains__((2,0,0,3))
            False
            sage: filter.__contains__((1,0,0,4))
            False
        """
        if self.__bound is infinity :
            return True
        
        (a, _, _, c) = f
        
        return ( a < self.__bound[0] and c < self.__bound[1] ) \
               if self.__has_multi_bound \
               else ( a < self.__bound and c < self.__bound )
    
    def __iter__(self) :
        """
        Iterate over all semi-definite integral hermitian quadratic forms contained
        in the filter, possibly up to reduction.
        
        NOTES:
            You may not iterate over infinite filters.
            The iteration over none reduced forms is not implemented yet.

        OUTPUT:
            - A generator over 4-tuples of integers.

        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3)
            sage: list(filter)             #indirect doctest
            [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3), (1, 0, 0, 1), (1, 1, 1, 1), (1, 0, 0, 2), (1, 1, 1, 2), (1, 0, 0, 3), (1, 1, 1, 3), (2, 0, 0, 2), (2, 1, 1, 2), (2, 2, 2, 2), (2, 3, 2, 2), (2, 0, 0, 3), (2, 1, 1, 3), (2, 2, 2, 3), (2, 3, 2, 3), (3, 0, 0, 3), (3, 1, 1, 3), (3, 2, 2, 3), (3, 3, 2, 3), (3, 3, 3, 3), (3, 4, 3, 3)]
            sage: filter = HermitianModularFormD2Filter_diagonal(infinity,-3)
            sage: filter.__iter__()
            Traceback (most recent call last):
            ...
            ArithmeticError: Cannot iterate over infinite filters.
            sage: filter = HermitianModularFormD2Filter_diagonal(30,-3)               # long time
            sage: indices = list(filter)                                              # long time
            sage: list(reduce_GL(form,-3)[0] for form in indices) == indices          # long time
            True
        """
        return itertools.chain( self.iter_semidefinite_forms_for_character(False),
                                self.iter_positive_forms_for_character(False),
                                self.iter_positive_forms_for_character(True) )
            
    def iter_semidefinite_forms_for_character( self, for_character = False) :
        """
        Iterate over all indefinite integral hermitian quadratic forms
        contained in the filter.

        NOTES:
            If ``for_character`` is True, the iteration will be empty.
            You may not iterate over infinite filters.
            The iteration over none reduced forms is not implemented yet.

        INPUT:
            - ``for_character`` -- A boolean (optional: default is False)
                                   If False only those forms will be iterated
                                   which can occur in the Fourier expansion
                                   of a form without character. If True only
                                   those will be iterated which occur in the
                                   Fourier expansion of a form with character.

        OUTPUT:
            - A generator over 4-tuples of integers.

        EXAMPLES::

            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3)
            sage: list(filter.iter_semidefinite_forms_for_character())
            [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3)]
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4,with_character=True)
            sage: list(filter.iter_semidefinite_forms_for_character())
            [(0, 0, 0, 0), (0, 0, 0, 2), (0, 0, 0, 4), (0, 0, 0, 6)]
            sage: filter = HermitianModularFormD2Filter_diagonal((2, 4),-3)
            sage: list(filter.iter_semidefinite_forms_for_character())
            [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 3)]

        TESTS::
        
            sage: filter = HermitianModularFormD2Filter_diagonal(infinity,-3)
            sage: filter.iter_semidefinite_forms_for_character()
            Traceback (most recent call last):
            ...
            ArithmeticError: Cannot iterate over infinite filters.
        """
        if self.is_infinite() :
            raise ArithmeticError( "Cannot iterate over infinite filters." )
        
        if self.__has_multi_bound :
            bound = self.__bound[1]
        else :
            bound = self.__bound
        
        if self.__with_character and not for_character :
            return ((0,0,0,c) for c in xrange(0, bound, 2))
        elif not self.__with_character and not for_character :
            return ((0,0,0,c) for c in xrange(0, bound))
        else :
            return (a for a in [])
    
    def iter_positive_forms_for_character( self, for_character = False ) :
        """
        Iterate over all positive definite integral hermitian quadratic
        forms contained in the filter.
        
        INPUT:
        
        - ``for_character`` -- A boolean (optional: default is False)
                               If False only those forms will be iterated
                               which can occur in the Fourier expansion
                               of a form without character. If True only
                               those will be iterated which occur in the
                               Fourier expansion of a form with character.

        NOTES:
        
            You may not iterate over infinite filters.
            The iteration over none reduced forms is not implemented yet.

        OUTPUT:
        
        - A generator over 4-tuples of integers.

        EXAMPLES::
        
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3)
            sage: list(filter.iter_positive_forms_for_character())
            [(1, 0, 0, 1), (1, 1, 1, 1), (1, 0, 0, 2), (1, 1, 1, 2), (1, 0, 0, 3), (1, 1, 1, 3), (2, 0, 0, 2), (2, 1, 1, 2), (2, 2, 2, 2), (2, 3, 2, 2), (2, 0, 0, 3), (2, 1, 1, 3), (2, 2, 2, 3), (2, 3, 2, 3), (3, 0, 0, 3), (3, 1, 1, 3), (3, 2, 2, 3), (3, 3, 2, 3), (3, 3, 3, 3), (3, 4, 3, 3)]
            sage: all(reduce_GL(f, -3)[0] == f for f in filter.iter_positive_forms_for_character())
            True
            sage: filter = HermitianModularFormD2Filter_diagonal((2, 4),-3)
            sage: list(filter.iter_positive_forms_for_character())
            [(1, 0, 0, 1), (1, 1, 1, 1), (1, 0, 0, 2), (1, 1, 1, 2), (1, 0, 0, 3), (1, 1, 1, 3)]
            
        TESTS::
        
            sage: filter = HermitianModularFormD2Filter_diagonal(5, -4)
            sage: list(filter.iter_positive_forms_for_character())
            [(1, 0, 0, 1), (1, 1, 1, 1), (1, 2, 1, 1), (1, 0, 0, 2), (1, 1, 1, 2), (1, 2, 1, 2), (1, 0, 0, 3), (1, 1, 1, 3), (1, 2, 1, 3), (1, 0, 0, 4), (1, 1, 1, 4), (1, 2, 1, 4), (2, 0, 0, 2), (2, 1, 1, 2), (2, 2, 1, 2), (2, 2, 2, 2), (2, 3, 2, 2), (2, 4, 2, 2), (2, 0, 0, 3), (2, 1, 1, 3), (2, 2, 1, 3), (2, 2, 2, 3), (2, 3, 2, 3), (2, 4, 2, 3), (2, 0, 0, 4), (2, 1, 1, 4), (2, 2, 1, 4), (2, 2, 2, 4), (2, 3, 2, 4), (2, 4, 2, 4), (3, 0, 0, 3), (3, 1, 1, 3), (3, 2, 1, 3), (3, 2, 2, 3), (3, 3, 2, 3), (3, 4, 2, 3), (3, 3, 3, 3), (3, 4, 3, 3), (3, 5, 3, 3), (3, 6, 3, 3), (3, 0, 0, 4), (3, 1, 1, 4), (3, 2, 1, 4), (3, 2, 2, 4), (3, 3, 2, 4), (3, 4, 2, 4), (3, 3, 3, 4), (3, 4, 3, 4), (3, 5, 3, 4), (3, 6, 3, 4), (4, 0, 0, 4), (4, 1, 1, 4), (4, 2, 1, 4), (4, 2, 2, 4), (4, 3, 2, 4), (4, 4, 2, 4), (4, 3, 3, 4), (4, 4, 3, 4), (4, 5, 3, 4), (4, 6, 3, 4), (4, 4, 4, 4), (4, 5, 4, 4), (4, 6, 4, 4), (4, 7, 4, 4), (4, 8, 4, 4)]
            sage: all(reduce_GL(f, -4)[0] == f for f in filter.iter_positive_forms_for_character())
            True
            sage: list(filter.iter_positive_forms_for_character(True))
            []
            sage: filter = HermitianModularFormD2Filter_diagonal(3, -4, with_character = True)
            sage: list(filter.iter_positive_forms_for_character(True))
            [(1, 1, 1, 1), (1, 1, 1, 3), (1, 1, 1, 5), (3, 1, 1, 3), (3, 3, 3, 3), (3, 5, 3, 3), (3, 1, 1, 5), (3, 3, 3, 5), (3, 5, 3, 5), (5, 1, 1, 5), (5, 3, 3, 5), (5, 5, 3, 5), (5, 5, 5, 5), (5, 7, 5, 5), (5, 9, 5, 5)]
            sage: list(filter.iter_positive_forms_for_character(False))
            [(2, 0, 0, 2), (2, 2, 2, 2), (2, 4, 2, 2), (2, 0, 0, 4), (2, 2, 2, 4), (2, 4, 2, 4), (4, 0, 0, 4), (4, 2, 2, 4), (4, 4, 2, 4), (4, 4, 4, 4), (4, 6, 4, 4), (4, 8, 4, 4)]
            sage: filter = HermitianModularFormD2Filter_diagonal(3, -3, reduced = False)
            sage: list(filter.iter_positive_forms_for_character())
            [(1, -2, -1, 1), (1, -1, -1, 1), (1, -1, 0, 1), (1, 0, 0, 1), (1, 1, 0, 1), (1, 1, 1, 1), (1, 2, 1, 1), (1, -4, -2, 2), (1, -3, -2, 2), (1, -2, -2, 2), (1, -3, -1, 2), (1, -2, -1, 2), (1, -1, -1, 2), (1, 0, -1, 2), (1, -2, 0, 2), (1, -1, 0, 2), (1, 0, 0, 2), (1, 1, 0, 2), (1, 2, 0, 2), (1, 0, 1, 2), (1, 1, 1, 2), (1, 2, 1, 2), (1, 3, 1, 2), (1, 2, 2, 2), (1, 3, 2, 2), (1, 4, 2, 2), (2, -4, -2, 1), (2, -3, -2, 1), (2, -2, -2, 1), (2, -3, -1, 1), (2, -2, -1, 1), (2, -1, -1, 1), (2, 0, -1, 1), (2, -2, 0, 1), (2, -1, 0, 1), (2, 0, 0, 1), (2, 1, 0, 1), (2, 2, 0, 1), (2, 0, 1, 1), (2, 1, 1, 1), (2, 2, 1, 1), (2, 3, 1, 1), (2, 2, 2, 1), (2, 3, 2, 1), (2, 4, 2, 1), (2, -6, -3, 2), (2, -5, -3, 2), (2, -4, -3, 2), (2, -3, -3, 2), (2, -5, -2, 2), (2, -4, -2, 2), (2, -3, -2, 2), (2, -2, -2, 2), (2, -1, -2, 2), (2, -4, -1, 2), (2, -3, -1, 2), (2, -2, -1, 2), (2, -1, -1, 2), (2, 0, -1, 2), (2, 1, -1, 2), (2, -3, 0, 2), (2, -2, 0, 2), (2, -1, 0, 2), (2, 0, 0, 2), (2, 1, 0, 2), (2, 2, 0, 2), (2, 3, 0, 2), (2, -1, 1, 2), (2, 0, 1, 2), (2, 1, 1, 2), (2, 2, 1, 2), (2, 3, 1, 2), (2, 4, 1, 2), (2, 1, 2, 2), (2, 2, 2, 2), (2, 3, 2, 2), (2, 4, 2, 2), (2, 5, 2, 2), (2, 3, 3, 2), (2, 4, 3, 2), (2, 5, 3, 2), (2, 6, 3, 2)]

        ::
        
            sage: filter = HermitianModularFormD2Filter_diagonal(5, -7)
            sage: list(filter.iter_positive_forms_for_character())
            [(1, -1, 0, 1), (1, 0, 0, 1), (1, 2, 1, 1), (1, 3, 1, 1), (1, -1, 0, 2), (1, 0, 0, 2), (1, 2, 1, 2), (1, 3, 1, 2), (1, -1, 0, 3), (1, 0, 0, 3), (1, 2, 1, 3), (1, 3, 1, 3), (1, -1, 0, 4), (1, 0, 0, 4), (1, 2, 1, 4), (1, 3, 1, 4), (2, -3, 0, 2), (2, -2, 0, 2), (2, -1, 0, 2), (2, 0, 0, 2), (2, 0, 1, 2), (2, 1, 1, 2), (2, 2, 1, 2), (2, 3, 1, 2), (2, 4, 2, 2), (2, 5, 2, 2), (2, 6, 2, 2), (2, 7, 2, 2), (2, -3, 0, 3), (2, -2, 0, 3), (2, -1, 0, 3), (2, 0, 0, 3), (2, 0, 1, 3), (2, 1, 1, 3), (2, 2, 1, 3), (2, 3, 1, 3), (2, 4, 2, 3), (2, 5, 2, 3), (2, 6, 2, 3), (2, 7, 2, 3), (2, -3, 0, 4), (2, -2, 0, 4), (2, -1, 0, 4), (2, 0, 0, 4), (2, 0, 1, 4), (2, 1, 1, 4), (2, 2, 1, 4), (2, 3, 1, 4), (2, 4, 2, 4), (2, 5, 2, 4), (2, 6, 2, 4), (2, 7, 2, 4), (3, -5, 0, 3), (3, -4, 0, 3), (3, -3, 0, 3), (3, -2, 0, 3), (3, -1, 0, 3), (3, 0, 0, 3), (3, -1, 1, 3), (3, 0, 1, 3), (3, 1, 1, 3), (3, 2, 1, 3), (3, 3, 1, 3), (3, 2, 2, 3), (3, 3, 2, 3), (3, 4, 2, 3), (3, 5, 2, 3), (3, 6, 2, 3), (3, 7, 2, 3), (3, 6, 3, 3), (3, 7, 3, 3), (3, 8, 3, 3), (3, 9, 3, 3), (3, 10, 3, 3), (3, -5, 0, 4), (3, -4, 0, 4), (3, -3, 0, 4), (3, -2, 0, 4), (3, -1, 0, 4), (3, 0, 0, 4), (3, -1, 1, 4), (3, 0, 1, 4), (3, 1, 1, 4), (3, 2, 1, 4), (3, 3, 1, 4), (3, 2, 2, 4), (3, 3, 2, 4), (3, 4, 2, 4), (3, 5, 2, 4), (3, 6, 2, 4), (3, 7, 2, 4), (3, 6, 3, 4), (3, 7, 3, 4), (3, 8, 3, 4), (3, 9, 3, 4), (3, 10, 3, 4), (4, -7, 0, 4), (4, -6, 0, 4), (4, -5, 0, 4), (4, -4, 0, 4), (4, -3, 0, 4), (4, -2, 0, 4), (4, -1, 0, 4), (4, 0, 0, 4), (4, -3, 1, 4), (4, -2, 1, 4), (4, -1, 1, 4), (4, 0, 1, 4), (4, 1, 1, 4), (4, 2, 1, 4), (4, 3, 1, 4), (4, 0, 2, 4), (4, 1, 2, 4), (4, 2, 2, 4), (4, 3, 2, 4), (4, 4, 2, 4), (4, 5, 2, 4), (4, 6, 2, 4), (4, 7, 2, 4), (4, 4, 3, 4), (4, 5, 3, 4), (4, 6, 3, 4), (4, 7, 3, 4), (4, 8, 3, 4), (4, 9, 3, 4), (4, 10, 3, 4), (4, 7, 4, 4), (4, 8, 4, 4), (4, 9, 4, 4), (4, 10, 4, 4), (4, 11, 4, 4), (4, 12, 4, 4), (4, 13, 4, 4), (4, 14, 4, 4)]
            sage: all(reduce_GL(f, -7)[0] == f for f in filter.iter_positive_forms_for_character())
            True
            sage: list(filter.iter_positive_forms_for_character(True))
            []
        
        ::
        
            sage: inds =  HermitianModularFormD2Indices_diagonal(-3)
            sage: filter = HermitianModularFormD2Filter_diagonal(3,-3)
            sage: iterated = list(filter.iter_positive_forms_for_character())
            sage: map(lambda s: inds.reduce(s)[0], iterated) == iterated
            True
        
        ::
        
            sage: filter = HermitianModularFormD2Filter_diagonal(infinity,-3)
            sage: list(filter.iter_positive_forms_for_character())
            Traceback (most recent call last):
            ...
            ArithmeticError: Cannot iterate over infinite filters.
            sage: filter = HermitianModularFormD2Filter_diagonal(4, -4, with_character = True, reduced=False)
            sage: list(filter.iter_positive_forms_for_character())
            Traceback (most recent call last):
            ...
            NotImplementedError: Iteration over nonreduced forms with character.
        """
        if self.is_infinite() :
            raise ArithmeticError( "Cannot iterate over infinite filters." )

        if not self.__with_character and for_character :
            raise StopIteration
        
        D = self.__D

        if self.__with_character :
            stepsize = 2
        else :
            stepsize = 1
        
        if self.__reduced :
            for a in xrange( 1 if for_character or not self.__with_character else 2,
                             self.__bound[0] if self.__has_multi_bound else self.__bound, stepsize ) :
                for c in xrange(a, self.__bound[1] if self.__has_multi_bound else self.__bound, stepsize) :
                    ## B1 ensures that we can take the square root of B2
                    B1 = isqrt(4 * a * c)
                    
                    if D == -4 :
                        for b2 in xrange(1 if for_character else 0, min(a, B1) + 1, stepsize) :
                            ## *_red : bounds given by general reduction of forms
                            ## *_redex : bounds given by sqrt(-3) reduction of forms
                            ## *_def : bounds given by positive definiteness of forms
                            b1min_red = max( 0, - ((2 * D * b2 - D * a) // 4) )
                            b1max_red = (-2 * D * b2 - D * a) // 4 + 1
                            b1min_redex = b2
                            b1max_redex = 2 * b2 + 1

                            B2 = D * (b2**2 - 4 * a * c)
                            b1min_def = (-D * b2 - (isqrt(B2 - 1) + 1 if B2 != 0 else 0)) // 2 + 1
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
                                yield (a, b1, b2, c) 
                    #! if D == -4
                    elif D == -3 :
                        ##TODO: Remove the for_character argument for_character that only
                        ##      applies to D = -4
                        for b2 in xrange(0, min(a, B1) + 1) :
                            ## *_red : bounds given by general reduction of forms
                            ## *_redex : bounds given by sqrt(-3) reduction of forms
                            ## *_def : bounds given by positive definiteness of forms
                            b1min_red = max( 0, - ((2 * D * b2 - D * a) // 4) )
                            b1max_red = (-2 * D * b2 - D * a) // 4 + 1
                            b1min_redex = b2
                            b1max_redex = (-D * b2) // 2 + 1

                            B2 = D * (b2**2 - 4 * a * c)
                            b1min_def = (-D * b2 - (isqrt(B2 - 1) + 1 if B2 != 0 else 0)) // 2 + 1
                            b1max_def = (-D * b2 + isqrt(B2 - 1)) // 2 + 1 \
                                        if B2 > 0 else \
                                        (-D * b2 ) // 2
                            
                            b1min = max(0, b1min_red, b1min_redex, b1min_def)
                            for b1 in xrange( b1min, min(b1max_red, b1max_def, b1max_redex) ) :
                                yield (a, b1, b2, c) 
                    #! elif D == -3
                    else :
                        for b2 in xrange(0, min(a, B1) + 1) :
                            ## *_red : bounds given by general reduction of forms
                            ## *_redex : bounds given by sqrt(-3) reduction of forms
                            ## *_def : bounds given by positive definiteness of forms
                            b1min_red = - ((2 * D * b2 - D * a) // 4)
                            b1max_red = (-2 * D * b2 - D * a) // 4 + 1
                            b1max_redex = (-D * b2) // 2 + 1

                            B2 = D * (b2**2 - 4 * a * c)
                            b1min_def = (-D * b2 - (isqrt(B2 - 1) + 1 if B2 != 0 else 0)) // 2 + 1
                            b1max_def = (-D * b2 + isqrt(B2 - 1)) // 2 + 1 \
                                        if B2 > 0 else \
                                        (-D * b2 ) // 2
                            
                            b1min = max(b1min_red, b1min_def)
                            for b1 in xrange( b1min, min(b1max_red, b1max_redex, b1max_def) ) :
                                yield (a, b1, b2, c)
                    #! else D == -4
                #! for c in xrange(a, self.__bound, 2)
            #! for a in xrange(0,self.__bound, 2)
        #! if self.__reduced
        else :
            if self.__with_character :
                raise NotImplementedError( "Iteration over nonreduced forms with character." )
            
            for a in xrange(1, self.__bound[0] if self.__has_multi_bound else self.__bound) :
                for c in xrange(1, self.__bound[1] if self.__has_multi_bound else self.__bound) :
                    ## B1 ensures that we can take the square root of B2
                    B1 = isqrt(4 * a * c)
                            
                    for b2 in xrange(-B1, B1 + 1) :
                        B2 = D * (b2**2 - 4 * a * c)
                        b1min_def = (-D * b2 - (isqrt(B2 - 1) + 1 if B2 != 0 else 0)) // 2 + 1
                        b1max_def = (-D * b2 + isqrt(B2 - 1)) // 2 + 1 \
                                    if B2 > 0 else \
                                    (-D * b2 ) // 2
                        
                        for b1 in xrange( b1min_def, b1max_def ) :
                            yield (a, b1, b2, c)
        #! if self.__reduced
        
        raise StopIteration
    
    def iter_forms_with_content_and_discriminant(self) :
        """
        Iterate over all semi-definite integral hermitian quadratic forms contained
        in the filter, possibly up to reduction. The content and discriminant of
        the form will be given, too.

        NOTES:
            You may not iterate over infinite filters.
            The iteration over none reduced forms is not implemented yet.

        OUTPUT:
            - A generator over 4-tuples of integers.

        EXAMPLES::
        
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3)
            sage: list(filter.iter_forms_with_content_and_discriminant())
            [((0, 0, 0, 0), 0, 0), ((0, 0, 0, 1), 1, 0), ((0, 0, 0, 2), 2, 0), ((0, 0, 0, 3), 3, 0), ((1, 0, 0, 1), 1, 3), ((1, 1, 1, 1), 1, 2), ((1, 0, 0, 2), 1, 6), ((1, 1, 1, 2), 1, 5), ((1, 0, 0, 3), 1, 9), ((1, 1, 1, 3), 1, 8), ((2, 0, 0, 2), 2, 12), ((2, 1, 1, 2), 1, 11), ((2, 2, 2, 2), 2, 8), ((2, 3, 2, 2), 1, 9), ((2, 0, 0, 3), 1, 18), ((2, 1, 1, 3), 1, 17), ((2, 2, 2, 3), 1, 14), ((2, 3, 2, 3), 1, 15), ((3, 0, 0, 3), 3, 27), ((3, 1, 1, 3), 1, 26), ((3, 2, 2, 3), 1, 23), ((3, 3, 2, 3), 1, 24), ((3, 3, 3, 3), 3, 18), ((3, 4, 3, 3), 1, 20)]

        TESTS::
        
            sage: filter = HermitianModularFormD2Filter_diagonal(infinity,-3)
            sage: filter.iter_forms_with_content_and_discriminant()
            Traceback (most recent call last):
            ...
            ArithmeticError: Cannot iterate over infinite filters.
        """
        return itertools.chain( self.iter_semidefinite_forms_for_character_with_content_and_discriminant(False),
                                self.iter_semidefinite_forms_for_character_with_content_and_discriminant(True),
                                self.iter_positive_forms_for_character_with_content_and_discriminant(False),
                                self.iter_positive_forms_for_character_with_content_and_discriminant(True) )
    
    def iter_semidefinite_forms_for_character_with_content_and_discriminant( self, for_character = False) :
        """
        Iterate over all indefinite integral hermitian quadratic forms
        contained in the filter. The content and discriminant of the
        form will be given, too.

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
        
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3)
            sage: list( filter.iter_semidefinite_forms_for_character_with_content_and_discriminant() )
            [((0, 0, 0, 0), 0, 0), ((0, 0, 0, 1), 1, 0), ((0, 0, 0, 2), 2, 0), ((0, 0, 0, 3), 3, 0)]
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4,with_character=True)
            sage: list( filter.iter_semidefinite_forms_for_character_with_content_and_discriminant() )
            [((0, 0, 0, 0), 0, 0), ((0, 0, 0, 2), 1, 0), ((0, 0, 0, 4), 2, 0), ((0, 0, 0, 6), 3, 0)]
            sage: filter = HermitianModularFormD2Filter_diagonal((3, 4), -4, with_character=True)
            sage: list( filter.iter_semidefinite_forms_for_character_with_content_and_discriminant() )
            [((0, 0, 0, 0), 0, 0), ((0, 0, 0, 2), 1, 0), ((0, 0, 0, 4), 2, 0), ((0, 0, 0, 6), 3, 0)]

        TESTS::
        
            sage: filter = HermitianModularFormD2Filter_diagonal(infinity,-3)
            sage: filter.iter_semidefinite_forms_for_character_with_content_and_discriminant()
            Traceback (most recent call last):
            ...
            ArithmeticError: Cannot iterate over infinite filters.
        """
        if self.is_infinite() :
            raise ArithmeticError( "Cannot iterate over infinite filters." )
        
        if self.__has_multi_bound :
            bound = self.__bound[1]
        else :
            bound = self.__bound
        
        if self.__with_character and not for_character :
            return (((0,0,0,c), c//2, 0) for c in xrange(0, bound, 2))
        elif not self.__with_character and not for_character :
            return (((0,0,0,c), c, 0) for c in xrange(0, bound))
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
            - A generator over 4-tuples of integers.

        EXAMPLES::

            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3)
            sage: list(filter.iter_positive_forms_for_character_with_content_and_discriminant())
            [((1, 0, 0, 1), 1, 3), ((1, 1, 1, 1), 1, 2), ((1, 0, 0, 2), 1, 6), ((1, 1, 1, 2), 1, 5), ((1, 0, 0, 3), 1, 9), ((1, 1, 1, 3), 1, 8), ((2, 0, 0, 2), 2, 12), ((2, 1, 1, 2), 1, 11), ((2, 2, 2, 2), 2, 8), ((2, 3, 2, 2), 1, 9), ((2, 0, 0, 3), 1, 18), ((2, 1, 1, 3), 1, 17), ((2, 2, 2, 3), 1, 14), ((2, 3, 2, 3), 1, 15), ((3, 0, 0, 3), 3, 27), ((3, 1, 1, 3), 1, 26), ((3, 2, 2, 3), 1, 23), ((3, 3, 2, 3), 1, 24), ((3, 3, 3, 3), 3, 18), ((3, 4, 3, 3), 1, 20)]
            sage: filter = HermitianModularFormD2Filter_diagonal((3, 4),-3)
            sage: list(filter.iter_positive_forms_for_character_with_content_and_discriminant())
            [((1, 0, 0, 1), 1, 3), ((1, 1, 1, 1), 1, 2), ((1, 0, 0, 2), 1, 6), ((1, 1, 1, 2), 1, 5), ((1, 0, 0, 3), 1, 9), ((1, 1, 1, 3), 1, 8), ((2, 0, 0, 2), 2, 12), ((2, 1, 1, 2), 1, 11), ((2, 2, 2, 2), 2, 8), ((2, 3, 2, 2), 1, 9), ((2, 0, 0, 3), 1, 18), ((2, 1, 1, 3), 1, 17), ((2, 2, 2, 3), 1, 14), ((2, 3, 2, 3), 1, 15)]

        TESTS::

            sage: filter = HermitianModularFormD2Filter_diagonal(3, -4)
            sage: map(lambda s: s[0], filter.iter_positive_forms_for_character_with_content_and_discriminant()) == list(filter.iter_positive_forms_for_character())
            True
            sage: filter = HermitianModularFormD2Filter_diagonal(3, -4, with_character = True)
            sage: map(lambda s: s[0], filter.iter_positive_forms_for_character_with_content_and_discriminant(True)) == list(filter.iter_positive_forms_for_character(True))
            True
            sage: map(lambda s: s[0], filter.iter_positive_forms_for_character_with_content_and_discriminant(False)) == list(filter.iter_positive_forms_for_character(False))
            True

        ::
        
            sage: filter = HermitianModularFormD2Filter_diagonal(3, -7)
            sage: map(lambda s: s[0], filter.iter_positive_forms_for_character_with_content_and_discriminant()) == list(filter.iter_positive_forms_for_character())
            True
            sage: filter = HermitianModularFormD2Filter_diagonal(3, -7)
            sage: list(filter.iter_positive_forms_for_character_with_content_and_discriminant(True))
            []
            
        ::
        
            sage: inds =  HermitianModularFormD2Indices_diagonal(-3)
            sage: filter = HermitianModularFormD2Filter_diagonal(3,-3)
            sage: iterated = map(lambda s: s[0], filter.iter_positive_forms_for_character_with_content_and_discriminant())
            sage: map(lambda s: inds.reduce(s)[0], iterated) == iterated
            True
            sage: list(filter.iter_positive_forms_for_character_with_content_and_discriminant(True))
            []

        ::
            
            sage: filter = HermitianModularFormD2Filter_diagonal(infinity,-3)
            sage: list(filter.iter_positive_forms_for_character_with_content_and_discriminant())
            Traceback (most recent call last):
            ...
            ArithmeticError: Cannot iterate over infinite filters.
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3,reduced=False)
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
            for a in xrange( 1 if for_character or not self.__with_character else 2,
                             self.__bound[0] if self.__has_multi_bound else self.__bound, stepsize) :
                for c in xrange(a, self.__bound[1] if self.__has_multi_bound else self.__bound, stepsize) :
                    ac_gcd = gcd(a,c)
                    ac_disc = -D * a * c
                    ## B1 ensures that we can take the square root of B2
                    B1 = isqrt(4 * a * c)
                    
                    if D == -4 :
                        for b2 in xrange(1 if for_character else 0, min(a, B1) + 1, stepsize) :
                            acb2_gcd = gcd(ac_gcd, b2)                            
                            
                            ## *_red : bounds given by general reduction of forms
                            ## *_redex : bounds given by sqrt(-3) reduction of forms
                            ## *_def : bounds given by positive definiteness of forms
                            b1min_red = max( 0, - ((2 * D * b2 - D * a) // 4) )
                            b1max_red = (-2 * D * b2 - D * a) // 4 + 1
                            b1min_redex = b2
                            b1max_redex = 2 * b2 + 1 

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
                    #! if D == -4
                    elif D == -3 :                   
                        for b2 in xrange(0, min(a, B1) + 1) :
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
                            for b1 in xrange( b1min, min(b1max_red, b1max_def, b1max_redex) ) :
                                yield ( (a, b1, b2, c), gcd(acb2_gcd, b1),
                                        ac_disc - DsqmD * b2**2 - b1**2 - D * b1 * b2 ) 
                    #! if D == -3
                    else :
                        for b2 in xrange(0, min(a, B1) + 1) :
                            acb2_gcd = gcd(ac_gcd, b2)                            
                            
                            ## *_red : bounds given by general reduction of forms
                            ## *_redex : bounds given by sqrt(-3) reduction of forms
                            ## *_def : bounds given by positive definiteness of forms
                            b1min_red = -((2 * D * b2 - D * a) // 4)
                            b1max_red = (-2 * D * b2 - D * a) // 4 + 1
                            b1max_redex = (-D * b2) // 2 + 1

                            B2 = D * (b2**2 - 4 * a * c)
                            b1min_def = (-D * b2 - isqrt(B2)) // 2
                            b1max_def = (-D * b2 + isqrt(B2 - 1)) // 2 + 1 \
                                        if B2 > 0 else \
                                        (-D * b2 ) // 2
                            
                            b1min = max(b1min_red, b1min_def)
                            for b1 in xrange( b1min, min(b1max_red, b1max_redex, b1max_def) ) :
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
        
    def __cmp__(self, other) :
        """
        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3)
            sage: filter2 = HermitianModularFormD2Filter_diagonal(4,-3)
            sage: filter == filter2
            True
            sage: filter2 = HermitianModularFormD2Filter_diagonal(4,-4)
            sage: filter == filter2
            False
            sage: filter2 = HermitianModularFormD2Filter_diagonal(5,-3)
            sage: filter == filter2
            False
            sage: filter2 = HermitianModularFormD2Filter_diagonal(4,-3,reduced=False)
            sage: filter == filter2
            False
            sage: filter2 = HermitianModularFormD2Filter_diagonal((4, 4),-3)
            sage: filter == filter2
            False
            sage: filter = HermitianModularFormD2Filter_diagonal((4, 4),-3)
            sage: filter == filter2
            True
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4,with_character=True,reduced=False)
            sage: filter2 = HermitianModularFormD2Filter_diagonal(4,-4,with_character=True,reduced=False)
            sage: filter == filter2
            True
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4)
            sage: filter2 = HermitianModularFormD2Filter_diagonal(4,-4,with_character=True)
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
            c = cmp(self.__bound, other.__bound)
            
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
        return self.__D + 19 * hash(self.__with_character) + 31 * hash(self.__bound)
                   
    def _repr_(self) :
        """
        OUTPUT:
            - A string

        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3);filter
            Reduced diagonal filter for discriminant -3 with bound 4
            sage: filter = HermitianModularFormD2Filter_diagonal((2,4),-3);filter
            Reduced diagonal filter for discriminant -3 with bound (2, 4)
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4,with_character=True,reduced=False);filter
            Diagonal filter for discriminant -4 respecting characters with bound 8
        """
        return "%siagonal filter for discriminant %s%s with bound %s" % \
               ( "Reduced d" if self.__reduced else "D", self.__D, 
                 " respecting characters" if self.__with_character else "",
                 self.__bound )
    
    def _latex_(self) :
        r"""
        OUTPUT:
            - A string

        TESTS::
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-3)
            sage: filter._latex_()
            '\text{Reduced diagonal filter for discriminant $-3$ with bound $4$}'
            sage: filter = HermitianModularFormD2Filter_diagonal(4,-4,with_character=True,reduced=False)
            sage: filter._latex_()
            '\text{Diagonal filter for discriminant $-4$ respecting characters with bound $8$}'
        """
        return "\text{%siagonal filter for discriminant $%s$%s with bound $%s$}" % \
               ( "Reduced d" if self.__reduced else "D", latex(self.__D), 
                 " respecting characters" if self.__with_character else "",
                 self.__bound )

#===============================================================================
# HermitianModularFormD2Indices_diagonal
#===============================================================================

class HermitianModularFormD2Indices_diagonal( SageObject ) :
    """
    This class implements the monoid of all semi positive definite hermitian
    binary quadratic forms over the integers `o_{\Q(\sqrt{D})}` of the imaginary
    quadratic field with discriminant `D`.  The associated filters are
    given in :class:~`.HermitianModularFormD2Filter_diagonal` and restrict
    the diagonal entries of such a form in matrix notation.
    
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
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds = HermitianModularFormD2Indices_diagonal(-3, True)
            Traceback (most recent call last):
            ...
            ValueError: Characters are admissable only if 4 | D.
            sage: inds = HermitianModularFormD2Indices_diagonal(-3)
            sage: inds = HermitianModularFormD2Indices_diagonal(-4, True)
            sage: inds = HermitianModularFormD2Indices_diagonal(-7, False, False)
            sage: inds = HermitianModularFormD2Indices_diagonal(-8, True, False)
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
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds = HermitianModularFormD2Indices_diagonal(-3)
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
            One the two generators with non-vanishing entries on the diagonal are returned.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds = HermitianModularFormD2Indices_diagonal(-3)
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
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds = HermitianModularFormD2Indices_diagonal(-3)
            sage: inds.gens()
            [(1, 0, 0, 0), (0, 0, 0, 1)]
        """
        return [self.gen(i) for i in xrange(self.ngens())]

    def is_commutative(self) :
        """
        OUTPUT:
            A boolean.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds = HermitianModularFormD2Indices_diagonal(-3)
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
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds = HermitianModularFormD2Indices_diagonal(-3)
            sage: inds.has_reduced_filters()
            True
            sage: inds = HermitianModularFormD2Indices_diagonal(-4, reduced = False)
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
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds = HermitianModularFormD2Indices_diagonal(-3)
            sage: inds.has_filters_with_character()
            False
            sage: inds = HermitianModularFormD2Indices_diagonal(-4, True)
            sage: inds.has_filters_with_character()
            True
        """
        return self.__with_character
    
    def monoid(self) :
        """
        If ``self`` respects the action of `\mathrm{GL}_{2}(o_{\Q(\sqrt{D})}` return the underlying
        monoid without this action. Otherwise return a copy of ``self``.
        
        OUTPUT:
            An instance of :class:`~.HermitianModularFormD2Indices_diagonal`.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds_a = HermitianModularFormD2Indices_diagonal(-3)
            sage: inds_woa = HermitianModularFormD2Indices_diagonal(-3, reduced = False)
            sage: inds_a.monoid() == inds_woa
            True
            sage: inds_woa.monoid() == inds_woa
            True
            sage: inds_woa.monoid() is inds_woa
            False
        """
        
        return HermitianModularFormD2Indices_diagonal( self.__D,
                        with_character = self.__with_character, reduced = False ) 

    def group(self) :
        """
        If ``self`` respects the action of `\mathrm{GL}_{2}(o_{\Q(\sqrt{D})}`, return this group.
        
        OUTPUT:
            A string.
        
        NOTE:
            The return value may change later to the actual group.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds_a = HermitianModularFormD2Indices_diagonal(-3)
            sage: inds_woa = HermitianModularFormD2Indices_diagonal(-3, reduced = False)
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
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds_a = HermitianModularFormD2Indices_diagonal(-3)
            sage: inds_woa = HermitianModularFormD2Indices_diagonal(-3, reduced = False)
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
    
    def filter(self, bound) :
        """
        Return a filter associated to this monoid of hermitian quadratic forms with given bound.
        
        INPUT:
            ``bound``    -- An integer or an instance of :class:`~.HermitianModularFormD2Filter_diagonal`;
                            A bound on the diagonal of the quadratic forms.
        
        OUTPUT:
            An instance of :class:`~.HermitianModularFormD2Filter_diagonal`.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal
            sage: inds = HermitianModularFormD2Indices_diagonal(-3)
            sage: inds.filter(2) == HermitianModularFormD2Filter_diagonal(2, -3)
            True
            sage: inds = HermitianModularFormD2Indices_diagonal(-7, reduced = False)
            sage: inds.filter(2) == HermitianModularFormD2Filter_diagonal(2, -7, reduced = False)
            True
            sage: inds = HermitianModularFormD2Indices_diagonal(-4, True) 
            sage: inds.filter(2) == HermitianModularFormD2Filter_diagonal(2, -4, True)
            True
            sage: inds = HermitianModularFormD2Indices_diagonal(-8, True, reduced = False)
            sage: inds.filter(2) == HermitianModularFormD2Filter_diagonal(2, -8, True, reduced = False)
            True
            sage: filter = HermitianModularFormD2Filter_diagonal(2, -8, True, reduced = False)
            sage: inds.filter(filter) == filter
            True
        """
        return HermitianModularFormD2Filter_diagonal( bound, self.__D,
                 with_character = self.__with_character, reduced = self.__reduced )
        
    def filter_all(self) :
        """
        Return the filter associated to this monoid of quadratic forms which contains all
        elements.
        
        OUTPUT:
            An instance of :class:`~HermitianModularFormD2Filter_diagonal`.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds = HermitianModularFormD2Indices_diagonal(-3)
            sage: inds.filter_all().is_all()
            True
            sage: inds = HermitianModularFormD2Indices_diagonal(-8, True, reduced = False)
            sage: fil = inds.filter_all()
            sage: fil.is_with_character()
            True
            sage: fil.is_reduced()
            False
        """
        return HermitianModularFormD2Filter_diagonal( infinity, self.__D,
                 with_character = self.__with_character, reduced = self.__reduced )
    
    def minimal_composition_filter(self, ls, rs) :
        """
        Given two lists `ls` and `rs` of hermitian quadratic forms return a filter that contains
        all the sums `l + r` of elements `l \in ls,\, r \in rs`.
        
        INPUT:
            `ls`  -- A list of 4-tuples of integers. Each element represents a quadratic form.
            `rs`  -- A list of 4-tuples of integers. Each element represents a quadratic form.
    
        OUTPUT:
            An instance of :class:`~HermitianModularFormD2Filter_diagonal`.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds = HermitianModularFormD2Indices_diagonal(-3)
            sage: inds.minimal_composition_filter([], []).index()
            0
            sage: inds = HermitianModularFormD2Indices_diagonal(-4, True, False)
            sage: fil = inds.minimal_composition_filter([], [])
            sage: fil.is_with_character()
            True
            sage: fil.is_reduced()
            False
            sage: inds.minimal_composition_filter([], [(1,0,0,1)]).index()
            0
            sage: inds.minimal_composition_filter([(1,0,0,1)], []).index()
            0
            sage: inds.minimal_composition_filter([(1,0,0,1)], [(1,0,0,1)]).index()
            4
            sage: inds.minimal_composition_filter([(1,1,0,1)], [(1,0,0,d) for d in range(6)]).index()
            8
            sage: inds.minimal_composition_filter([(d,1,0,1) for d in range(10)], [(1,0,0,d) for d in range(6)]).index()
            12
            sage: inds = HermitianModularFormD2Indices_diagonal(-4)
            sage: inds.minimal_composition_filter([(1,0,0,1)], [(1,0,0,1)]).index()
            3
        """
        if len(ls) == 0 or len(rs) == 0 :
            return HermitianModularFormD2Filter_diagonal( 0, self.__D,
                                   with_character = self.__with_character,
                                   reduced = self.__reduced )
        
        bound = max( max(0, max(a for (a,_,_,_) in ls)) + max(0, max(a for (a,_,_,_) in rs)),
                            max(0, max(c for (_,_,_,c) in ls)) + max(0, max(c for (_,_,_,c) in rs)) ) + 1
        if self.__with_character :
            bound = -(-bound // 2)
            
        return HermitianModularFormD2Filter_diagonal( bound, self.__D,
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
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds = HermitianModularFormD2Indices_diagonal(-3)
            sage: inds._reduction_function() == inds.reduce
            True
            sage: inds = HermitianModularFormD2Indices_diagonal(-4, reduced = False)
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
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds = HermitianModularFormD2Indices_diagonal(-3)
            sage: inds.reduce((1,0,0,1))
            ((1, 0, 0, 1), (1, 0, 1))
            sage: inds = HermitianModularFormD2Indices_diagonal(-4, True)
            sage: inds.reduce((2,0,0,2))
            Traceback (most recent call last):
            ...
            NotImplementedError
        """
        if not self.__with_character :
            return reduce_GL(s, self.__D)
  
        raise NotImplementedError

    def _decompositions_with_stepsize(self, s, stepsize, offset) :
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
        
        OUTPUT:
            A generator interating over pairs of 4-tuples of integers.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds = HermitianModularFormD2Indices_diagonal(-3)
            sage: len(list(inds._decompositions_with_stepsize((2,0,0,2), 1, 0)))
            21
            sage: ((0,0,0,0), (4,0,0,4)) in inds._decompositions_with_stepsize((4,0,0,4), 2, 1)
            False
            sage: list(inds._decompositions_with_stepsize((2,0,0,2), 2, 0))                    
            [((0, 0, 0, 0), (2, 0, 0, 2)), ((0, 0, 0, 2), (2, 0, 0, 0)), ((2, 0, 0, 0), (0, 0, 0, 2)), ((2, 0, 0, 2), (0, 0, 0, 0))]
        """
        (a, b1, b2, c) = s

        D = self.__D

        for a1 in xrange(offset, a + 1, stepsize) :
            a2 = a - a1
            for c1 in xrange(offset, c + 1, stepsize) :
                c2 = c - c1
                
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
        
        raise StopIteration
    
    def decompositions(self, s, signature = 0) :
        """
        Decompose a quadratic form `s = s_1 + s_2` in all possible ways. If ``signature``
        is non-zero and this monoid is for characters, the iteration is restricted to all
        even `s_1` if ``signature`` is `1` and to all odd `s_1` if it is `-1`. 
        
        INPUT:
            `s`           -- A 4-tuple of integers; Represents a quadratic form.
            ``signature`` -- An integer `-1`, `0` or `1`; Determines the oddness of `s_1`.

        OUTPUT:
            A generator interating over pairs of 4-tuples of integers.
            
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds_a = HermitianModularFormD2Indices_diagonal(-3)
            sage: inds_woa = HermitianModularFormD2Indices_diagonal(-3, reduced = False)
            sage: len(list(inds_a.decompositions((2,1,0,2))))
            14
            sage: list(inds_a.decompositions((2,1,0,2))) == list(inds_woa.decompositions((2,1,0,2)))
            True
            sage: inds = HermitianModularFormD2Indices_diagonal(-4, True)
            sage: list(inds.decompositions((3,0,0,3), 1)) + list(inds.decompositions((3,0,0,3), -1)) == list(inds.decompositions((3,0,0,3)))
            True
            sage: inds.decompositions((3,0,0,3), 2)
            Traceback (most recent call last):
            ...
            ValueError: Signature must be -1, 0 or 1.
        """
        if not self.__with_character :
            return self._decompositions_with_stepsize(s, 1, 0)
        else :
            if signature == 0 :
                return itertools.chain(self._decompositions_with_stepsize(s, 2, 0),
                                       self._decompositions_with_stepsize(s, 2, 1))
            elif signature == 1 :
                return itertools.chain(self._decompositions_with_stepsize(s, 2, 0))
            elif signature == -1 :
                return itertools.chain(self._decompositions_with_stepsize(s, 2, 1))
            else :
                raise ValueError( "Signature must be -1, 0 or 1." )
        
        raise StopIteration
    
    def zero_element(self) :
        """
        Return the zero element of this monoid.
        
        OUTPUT:
            A 4-tuple of integers.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds = HermitianModularFormD2Indices_diagonal(-3)
            sage: inds.zero_element()
            (0, 0, 0, 0)
        """
        return (0,0,0,0)

    def __contains__(self, x) :
        """
        INPUT:
            An aribitrary object.
            
        OUTPUT:
            A boolean.
            
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds = HermitianModularFormD2Indices_diagonal(-3)
            sage: (2, 2, 3, 4) in inds
            True
            sage: (1, 2, 3, 4) in inds
            False
            sage: (1, 3, 4) in inds
            False
            sage: 'a' in inds
            False
            sage: (1,2,3,'a') in inds
            False
        """
        D = self.__D
        return isinstance(x, tuple) and len(x) == 4 and all(isinstance(e, (int, Integer)) for e in x) and \
               x[0] >= 0 and x[3] >= 0 and - D * x[0] * x[3] - x[1]**2 - D * x[1] * x[2] - (D**2 - D) // 4 * x[2]**2 >= 0

    def __cmp__(self, other) :
        """
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: inds = HermitianModularFormD2Indices_diagonal(-3)
            sage: inds == 2
            False 
            sage: inds == HermitianModularFormD2Indices_diagonal(-4)
            False
            sage: inds == HermitianModularFormD2Indices_diagonal(-4, True)
            False
            sage: inds == HermitianModularFormD2Indices_diagonal(-3, reduced = False)
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
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: hash( HermitianModularFormD2Indices_diagonal(-3) )
            16
            sage: hash( HermitianModularFormD2Indices_diagonal(-4, True) )
            28
            sage: hash( HermitianModularFormD2Indices_diagonal(-7, reduced = False) )
            -7
            sage: hash( HermitianModularFormD2Indices_diagonal(-8, True, False) )
            5
        """
        return hash(self.__D) + 13 * hash(self.__with_character) + 19 * hash(self.__reduced)
    
    def _repr_(self) :
        """
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: repr( HermitianModularFormD2Indices_diagonal(-3) )
            'Reduced quadratic forms over o_QQ(\\sqrt -3)'
            sage: repr( HermitianModularFormD2Indices_diagonal(-4, True) )
            'Reduced quadratic forms over o_QQ(\\sqrt -4) for character forms'
            sage: repr( HermitianModularFormD2Indices_diagonal(-7, reduced = False) )
            'Non-reduced quadratic forms over o_QQ(\\sqrt -7)'
            sage: repr( HermitianModularFormD2Indices_diagonal(-8, True, False) )
            'Non-reduced quadratic forms over o_QQ(\\sqrt -8) for character forms'
        """
        return "%seduced quadratic forms over o_QQ(\sqrt %s)%s" % \
          ( "R" if self.__reduced else "Non-r", self.__D,
            " for character forms" if self.__with_character else "" )
            
    def _latex_(self) :
        r"""
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
            sage: latex( HermitianModularFormD2Indices_diagonal(-3) )
            \text{Reduced quadratic forms over $\mathfrak{o}_{\mathbb{Q}(\sqrt{-3})}$}
            sage: latex( HermitianModularFormD2Indices_diagonal(-4, True) )
            \text{Reduced quadratic forms over $\mathfrak{o}_{\mathbb{Q}(\sqrt{-4})}$ for character forms}
            sage: latex( HermitianModularFormD2Indices_diagonal(-7, reduced = False) )
            \text{Non-reduced quadratic forms over $\mathfrak{o}_{\mathbb{Q}(\sqrt{-7})}$}
            sage: latex( HermitianModularFormD2Indices_diagonal(-8, True, False) )
            \text{Non-reduced quadratic forms over $\mathfrak{o}_{\mathbb{Q}(\sqrt{-8})}$ for character forms}
        """
        return "\\text{%seduced quadratic forms over $\mathfrak{o}_{\mathbb{Q}(\sqrt{%s})}$%s}" % \
          ( "R" if self.__reduced else "Non-r", latex(self.__D),
            " for character forms" if self.__with_character else "" )

#===============================================================================
# HermitianModularFormD2FourierExpansionCharacterMonoid
#===============================================================================
## TODO: Add shortcuts for characters that may occur, so that the user
##       does not confuse the indices.

_character_eval_function_cache = dict()

def HermitianModularFormD2FourierExpansionCharacterMonoid( D, K ) :
    """
    Return the monoid of all possible characters for hermitian modular
    forms associated to the full modular over `\Q(\sqrt{D})`.
    The first component of the underlying monoid distinguishes whether
    there the Fourier expansion is symmetric or anti-symmetric, the
    second reflects the multiple of `\det^{|\mathfrak{o}_{\Q(\sqrt{D})}|/2}`
    that occures as a character, and if `D = -4` the third is the exceptional
    character `\nu`.   
    
    INPUT:
        - `K` -- A ring; The codomain of all characters.
        - `D` -- A negative integer; The fundamental discriminant of an
                 imaginary quadratic field.
    
    OUTPUT:
        A monoid of characters.
    
    TESTS:
        sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2FourierExpansionCharacterMonoid
        sage: HermitianModularFormD2FourierExpansionCharacterMonoid(-4, ZZ)
        Character monoid over Multiplicative Abelian Group isomorphic to C2 x C2 x C2
        sage: HermitianModularFormD2FourierExpansionCharacterMonoid(-3, ZZ)
        Character monoid over Multiplicative Abelian Group isomorphic to C2 x C2
        sage: M = HermitianModularFormD2FourierExpansionCharacterMonoid(-7, QQ)
        sage: M([1,0]) * M([0,1])
        f0*f1
    """
    try :
        (C, eval) = _character_eval_function_cache[D]
    except KeyError :
        if D == -4 :
            C = AbelianGroup([2,2,2])
            eval = lambda (trans, det, nu), c:   (1 if c._monoid_element().list()[0] == 0 or trans == 1 else -1) \
                                               * (1 if c._monoid_element().list()[1] == 0 or det == 0 or det == 2 else -1 ) \
                                               * (1 if c._monoid_element().list()[2] == 0 or nu == 1 else -1)
        elif D == -3 :
            C = AbelianGroup([2,2])
            eval = lambda (trans, det, nu), c:   (1 if c._monoid_element().list()[0] == 0 or trans == 1 else -1) \
                                               * (1 if c._monoid_element().list()[1] == 0 or det == 0 or det == 2 or det == 4 else -1 )
        else :
            C = AbelianGroup([2,2])
            
            eval = lambda (trans, det, nu), c:   (1 if c._monoid_element().list()[0] == 0 or trans == 1 else -1) \
                                               * (1 if c._monoid_element().list()[1] == 0 or det == 0 else -1 )
        
        _character_eval_function_cache[D] = (C, eval)
    
    return CharacterMonoid_class("GL(2,o_QQ(sqrt %s))" % D, C, K, eval)

def HermitianModularFormD2FourierExpansionTrivialCharacter(D, K, weight_parity = 0) :
    """
    Return the trivial character attached to Fourier expansions
    of hermitian modular forms.
    
    INPUT:

        - `K`               -- A ring; The codomain of all characters.
        - `D`               -- A negative integer; The fundamental discriminant of an
                               imaginary quadratic field.
        - ``weight_parity`` -- An integer (default: `0`); The parity of the
                               associated weight
    
    OUTPUT:

        An element of a monoid of characters.
    
    TESTS:
        sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2FourierExpansionCharacterMonoid
        sage: ch = HermitianModularFormD2FourierExpansionTrivialCharacter(-4, ZZ)
        sage: map(ch, [(-1,0,1), (1,1,1), (1,1,-1)])
        [1, 1, 1]
        sage: ch = HermitianModularFormD2FourierExpansionTrivialCharacter(-3, ZZ)
        sage: map(ch, [(-1,0,0), (1, 1,0)])
        [1, 1]
        sage: ch = HermitianModularFormD2FourierExpansionTrivialCharacter(-7, ZZ)
        sage: map(ch, [(-1,0,0), (1, 1,0)])
        [1, 1]
        sage: ch = HermitianModularFormD2FourierExpansionTrivialCharacter(-4, ZZ, 1)
        sage: map(ch, [(-1,0,1), (1,1,1), (1,1,-1)])
        [1, -1, -1]
        sage: ch = HermitianModularFormD2FourierExpansionTrivialCharacter(-3, ZZ, 1)
        sage: map(ch, [(-1,0,0), (1, 1,0)])
        [1, -1]
    """
    chmonoid = HermitianModularFormD2FourierExpansionCharacterMonoid(D, K)
    monoid = chmonoid.monoid()
    
    if weight_parity % 2 == 0 :
        return chmonoid.one_element()
    elif D == -4 :
        return CharacterMonoidElement_class(chmonoid, monoid([0,1,0]))
    else :
        return CharacterMonoidElement_class(chmonoid, monoid([0,1]))

def HermitianModularFormD2FourierExpansionTransposeCharacter(D, K, weight_parity = 0) :
    """
    Return the character attached to Fourier expansions
    of hermitian modular forms that attains `-1` for transpositions
    and `1` otherwise.
    
    INPUT:

        - `K`               -- A ring; The codomain of all characters.
        - `D`               -- A negative integer; The fundamental discriminant of an
                               imaginary quadratic field.
        - ``weight_parity`` -- An integer (default: `0`); The parity of the
                               associated weight
    
    OUTPUT:
        An element of a monoid of characters.
    
    TESTS:
        sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2FourierExpansionCharacterMonoid
        sage: ch = HermitianModularFormD2FourierExpansionTransposeCharacter(-4, ZZ)
        sage: map(ch, [(-1,0,1), (1,1,1), (1,1,-1)])
        [-1, 1, 1]
        sage: ch = HermitianModularFormD2FourierExpansionTransposeCharacter(-3, ZZ)
        sage: map(ch, [(-1,0,0), (1, 1,0)])
        [-1, 1]
        sage: ch = HermitianModularFormD2FourierExpansionTransposeCharacter(-7, ZZ)
        sage: map(ch, [(-1,0,0), (1, 1,0)])
        [-1, 1]
        sage: ch = HermitianModularFormD2FourierExpansionTransposeCharacter(-4, ZZ)
        sage: map(ch, [(-1,0,1), (1,1,1), (1,1,-1)])
        [-1, 1, 1]
        sage: ch = HermitianModularFormD2FourierExpansionTransposeCharacter(-3, ZZ)
        sage: map(ch, [(-1,0,0), (1, 1,0)])
        [-1, 1]
    """
    chmonoid = HermitianModularFormD2FourierExpansionCharacterMonoid(D, K)
    monoid = chmonoid.monoid()
    
    if weight_parity % 2 == 0 :
        if D == -4 :
            return CharacterMonoidElement_class(chmonoid, monoid([1,0,0]))
        else :
            return CharacterMonoidElement_class(chmonoid, monoid([1,0]))
    else :
        if D == -4 :
            return CharacterMonoidElement_class(chmonoid, monoid([1,1,0]))
        else :
            return CharacterMonoidElement_class(chmonoid, monoid([1,1]))

#===============================================================================
# HermitianModularFormD2FourierExpansionRing
#===============================================================================

def HermitianModularFormD2FourierExpansionRing(K, D, with_nu_character = False) :
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
        sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Indices_diagonal
        sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2FourierExpansionRing
        sage: fe_ring = HermitianModularFormD2FourierExpansionRing(ZZ, -3)
        sage: fe_ring.action() == HermitianModularFormD2Indices_diagonal(-3)
        True
        sage: fe_ring.base_ring()
        Integer Ring
        sage: fe_ring.coefficient_domain()
        Integer Ring
        sage: fe_ring.characters()
        Character monoid over Multiplicative Abelian Group isomorphic to C2 x C2
        sage: fe_ring.representation()
        Trivial representation of GL(2,o_QQ(sqrt -3)) on Integer Ring
        sage: fe_ring = HermitianModularFormD2FourierExpansionRing(QQ, -4, True)
        sage: fe_ring.action() == HermitianModularFormD2Indices_diagonal(-4, True)
        True
    """
    R = EquivariantMonoidPowerSeriesRing(
         HermitianModularFormD2Indices_diagonal(D, with_nu_character),
         HermitianModularFormD2FourierExpansionCharacterMonoid(D, ZZ),
         TrivialRepresentation("GL(2,o_QQ(sqrt %s))" % (D,), K))
        
    return R
