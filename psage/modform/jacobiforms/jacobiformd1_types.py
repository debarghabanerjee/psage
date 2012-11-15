r"""
Types of Jacobi forms of fixed index and weight.

AUTHOR :
    - Martin Raum (2012 - 09 - 11) Initial version based on code for
                                   Jacobi forms of scalar index.
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

from operator import xor
from psage.modform.fourier_expansion_framework.gradedexpansions.gradedexpansion_grading import TrivialGrading
from psage.modform.fourier_expansion_framework.modularforms.modularform_ambient import ModularFormsModule_generic
from psage.modform.fourier_expansion_framework.modularforms.modularform_types import ModularFormType_abstract
from psage.modform.jacobiforms.jacobiformd1_dimensionformula import dimension__jacobi
from psage.modform.jacobiforms.jacobiformd1_element import JacobiFormD1_class
from psage.modform.jacobiforms.jacobiformd1_fegenerators import jacobi_form_d1_by_restriction
from psage.modform.jacobiforms.jacobiformd1_fourierexpansion import JacobiFormD1FourierExpansionModule, \
                                                                    JacobiFormD1Filter
from psage.modform.jacobiforms.jacobiformd1_module import JacobiFormD1Module
from sage.categories.number_fields import NumberFields
from sage.matrix.constructor import diagonal_matrix, matrix, zero_matrix,\
    identity_matrix
from sage.misc.cachefunc import cached_method
from sage.misc.mrange import mrange
from sage.modular.modform.constructor import ModularForms
from sage.rings.all import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.number_field.number_field import CyclotomicField
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.rational_field import QQ
from sage.structure.sequence import Sequence


#===============================================================================
# JacobiFormsD1
#===============================================================================

_jacobiforms_cache = dict()

def JacobiFormsD1(A, type, precision, *args, **kwds) :
    global _jacobiforms_cache
    
    if isinstance(precision, (int, Integer)) :
        precision = JacobiFormD1Filter(precision, type.index())
    
    k = (A, type, precision)
    
    try :
        return _jacobiforms_cache[k]
    except KeyError :
        if isinstance(type, JacobiFormD1_Gamma) :
            M = JacobiFormD1Module(A, type, precision)
        else :
            raise TypeError( "{0} must be a Jacobi form type,".format(type) )
        
        _jacobiforms_cache[k] = M
        return M
    
#===============================================================================
# JacobiFormD1_Gamma
#===============================================================================

class JacobiFormD1_Gamma ( ModularFormType_abstract ) :
    r"""
    Type of Jacobi forms of degree `1` and arbitrary rank index associated with
    the full modular group.
    
    TESTS::
    
        sage: from psage.modform.jacobiforms import *
        sage: from psage.modform.jacobiforms.jacobiformd1_fourierexpansion import JacobiFormD1Filter
        sage: L = QuadraticForm(matrix(2, [2,1,1,2])) 
        sage: JR = JacobiFormsD1(QQ, JacobiFormD1_Gamma(10, L), JacobiFormD1Filter(5, L))
        sage: JR.gens()
        (Graded expansion phiRes_0, Graded expansion phiRes_1)
        sage: JR.0 + 2 * JR.1
        Graded expansion phiRes_0 + 2*phiRes_1
    """
    def __init__(self, weight, index) :
        self.__weight = weight
        self.__index = index
    
    def weight(self) :
        return self.__weight

    def index(self) :
        return self.__index
    
    def _ambient_construction_function(self) :
        return JacobiFormsD1

    def _ambient_element_class(self) :
        return JacobiFormD1_class
    
    def group(self) :
        return "\Gamma^J_{0}".format(self.__index.matrix().nrows())
    
    @cached_method
    def _rank(self, K) :
        return dimension__jacobi(self.__weight, self.__index)
    
    @cached_method
    def generators(self, K, precision) :
        if K is QQ or K in NumberFields() :
            return Sequence( [ jacobi_form_d1_by_restriction(precision, self.__weight, i)
                               for i in xrange(self._rank(K)) ],
                             universe = JacobiFormD1FourierExpansionModule(QQ, self.__weight, self.__index) )
        
        raise NotImplementedError
    
    def grading(self, K) :
        if K is QQ or K in NumberFields() :
            return TrivialGrading( self._rank(K), None )
        
        raise NotImplementedError

    def _generator_names(self, K) :
        if K is QQ or K in NumberFields() :
            return [ "phiRes_%s" % (i,) for i in xrange(self._rank(K)) ]
        
        raise NotImplementedError

    def _generator_by_name(self, K, name) :
        if K is QQ or K in NumberFields() :
            R = self.generator_relations(K).ring()
            try :
                return R.gen(self._generator_names(K).index(name))
            except ValueError :
                raise ValueError( "Name {0} does not exist for {1}".format(name, K) )
        
        raise NotImplementedError
    
    @cached_method
    def generator_relations(self, K) :
        r"""
        An ideal I in a polynomial ring R, such that the associated module
        is (R / I)_1. 
        """
        if K is QQ or K in NumberFields() :
            R = PolynomialRing(K, self._generator_names(K))
            return R.ideal(0)
            
        raise NotImplementedError
    
    def graded_submodules_are_free(self) :
        return True

    def __cmp__(self, other) :
        c = cmp(type(self), type(other))
        
        if c == 0 :
            c = cmp(self.__index, other.__index)
        if c == 0 :
            c = cmp(self.__weight, other.__weight)
            
        return c

    def __hash__(self) :
        mat = self.__index.matrix()
        mat.set_immutable()
        return reduce(xor, map(hash, [type(self), self.__weight, mat]))
