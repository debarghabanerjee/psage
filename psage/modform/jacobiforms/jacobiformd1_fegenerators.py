r"""
Using restriction to scalar indices, we compute Jacobi forms of arbitrary index.  
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

from psage.modform.fourier_expansion_framework.monoidpowerseries.monoidpowerseries_lazyelement import \
                                        EquivariantMonoidPowerSeries_lazy
from psage.modform.jacobiforms.jacobiformd1_dimensionformula import dimension__jacobi
from psage.modform.jacobiforms.jacobiformd1_fourierexpansion import JacobiFormD1FourierExpansionModule, \
                                                                    JacobiFormD1Filter
from psage.modform.jacobiforms.jacobiformd1nn_fourierexpansion import JacobiFormD1NNIndices, JacobiFormD1NNFilter, JacobiFormD1WeightCharacter
from psage.modform.jacobiforms.jacobiformd1nn_types import JacobiFormsD1NN, JacobiFormD1NN_Gamma
from sage.combinat.partition import number_of_partitions
from sage.libs.flint.fmpz_poly import Fmpz_poly  
from sage.matrix.constructor import matrix, zero_matrix
from sage.misc.cachefunc import cached_function
from sage.misc.cython import cython_lambda
from sage.misc.functional import isqrt
from sage.misc.flatten import flatten
from sage.misc.misc import prod
from sage.modular.modform.constructor import ModularForms
from sage.modular.modform.element import ModularFormElement
from sage.modules.all import FreeModule, vector, span
from sage.rings.all import GF
from sage.rings.arith import binomial, factorial
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ
from sage.quadratic_forms.all import QuadraticForm
from sage.sets.all import Set
from sage.structure.sage_object import SageObject
import operator
from copy import copy

#===============================================================================
# jacobi_form_d1_by_restriction
#===============================================================================

def jacobi_form_d1_by_restriction(precision, k, i) :
    r"""
    Lazy Fourier expansions of a basis of Jacobi forms (of arbitrary
    rank index).
    
    INPUT:
    
    - ``precision`` -- A filter for Fourier expansions of Jacobi forms. 
    
    - `k` -- An integer; The weight of the Jacobi forms.
    
    - `i` -- A nonnegative integer less the dimension of the considered
             space of Jacobi forms.
    
    OUTPUT:
    
    - A lazy element of the corresponding module of Fourier expansions (over `\QQ`).
    
    TESTS::
    
        sage: from psage.modform.jacobiforms.jacobiformd1_fegenerators import *
        sage: from psage.modform.jacobiforms.jacobiformd1_fourierexpansion import *
        sage: indices = JacobiFormD1Indices(QuadraticForm(matrix(2, [2,1,1,2])))
        sage: h = jacobi_form_d1_by_restriction(indices.filter(5), 12, 1)
    """
    expansion_ring = JacobiFormD1FourierExpansionModule(QQ, k, precision.jacobi_index())
    coefficients_factory = JacobiFormD1DelayedFactory__restriction( precision, k, i )
    
    return EquivariantMonoidPowerSeries_lazy(expansion_ring, precision, coefficients_factory.getcoeff)

#===============================================================================
# JacobiFormD1DelayedFactory__restriction
#===============================================================================

class JacobiFormD1DelayedFactory__restriction :
    def __init__(self, precision, k, i) :
        r"""
        INPUT:
    
        - ``precision`` -- A filter for Fourier expansions of Jacobi forms. 
        
        - `k` -- An integer; The weight of the Jacobi forms.
        
        - `i` -- A nonnegative integer less the dimension of the considered
                 space of Jacobi forms.
        """
        self.__precision = precision
        self.__weight = k
        self.__i = i
        
        self.__ch = JacobiFormD1WeightCharacter(k, precision.jacobi_index().matrix().nrows())
        
    def getcoeff(self, key, **kwds) :
        (ch, k) = key
        
        if ch != self.__ch :
            return 0
        else :
            return _coefficient_by_restriction( self.__precision, self.__weight )[self.__i][(ch,k)]                                    

#===============================================================================
# _find_complete_set_of_restriction_vectors
#===============================================================================

def _find_complete_set_of_restriction_vectors(L, R, additional_s = 0, reduction_function = None) :
    r"""
    Given a set R of elements in L^# (e.g. representatives for ( L^# / L ) / \pm 1)
    find a complete set of restriction vectors. (See [GKR])
    
    INPUT:
    
    - `L` -- A quadratic form.
    
    - `R` -- A list of tuples or vectors in L \otimes QQ (with
             given coordinates).

    - ``additional_s`` -- A non-negative integer; Number of additional
                          elements of `L` that should be returned.
    
    - ``reduction_function`` -- A function that takes a tuple representing an element in `L^\#`
                                and returs a pair of a reduced element in `L^\#` and a sign.
    
    OUTPUT:
    
    - A set S of pairs, the first of which is a vector corresponding to
      an element in L, and the second of which is an integer.
    
    TESTS::
    
        sage: from psage.modform.jacobiforms.jacobiformd1_fegenerators import _find_complete_set_of_restriction_vectors
        sage: from psage.modform.jacobiforms.jacobiformd1_fourierexpansion import *
        sage: indices = JacobiFormD1Indices(QuadraticForm(matrix(2, [2,1,1,2])))
        sage: _find_complete_set_of_restriction_vectors(indices.jacobi_index(), indices._r_representatives)
        [((1, 0), 0), ((1, 0), 1), ((-2, 1), 1)]
        sage: _find_complete_set_of_restriction_vectors(indices.jacobi_index(), indices._r_representatives, reduction_function = indices.reduce_r)
        [((1, 0), 0), ((1, 0), 1), ((-2, 1), 1)]
        
    ::
     
        sage: from psage.modform.jacobiforms.jacobiformd1_fegenerators import _local_restriction_matrix
        sage: indices = JacobiFormD1Indices(QuadraticForm(matrix(4, [2,0,0,1, 0,2,0,1, 0,0,2,1, 1,1,1,2])))
        sage: S = _find_complete_set_of_restriction_vectors(indices.jacobi_index(), indices._r_representatives)
        sage: _local_restriction_matrix(indices._r_representatives, S).rank()
        4
    """
    R = [map(vector, rs) for rs in R]
    
    length_inc = 5
    max_length = 5
    cur_length = 1
    short_vectors = L.short_vector_list_up_to_length(max_length, True)
    
    S = list()
    restriction_space = FreeModule(QQ, len(R)).span([])
    
    
    while (len(S) < len(R) + additional_s ) :
        while len(short_vectors[cur_length]) == 0 :
            cur_length += 1
            if max_length >= cur_length :
                max_length += length_inc
                short_vectors = L.short_vector_list_up_to_length(max_length, True)
        
        s = vector( short_vectors[cur_length].pop() )
        
        rcands = Set([ s.dot_product(r) for rs in R for r in rs ])
        
        for r in rcands :
            v = _eval_restriction_vector(R, s, r, reduction_function)
            if len(S) - restriction_space.rank() < additional_s \
              or v not in restriction_space :
                S.append((s, r))
                restriction_space = restriction_space + FreeModule(QQ, len(R)).span([v])
                
                if len(S) == len(R) + additional_s :
                    break
    
    return S
    
#===============================================================================
# _eval_restriction_vector
#===============================================================================

def _eval_restriction_vector(R, s, r, reduction_function) :
    r"""
    For each list rs in R compute the multiplicity of s r' = r, r' in rs.
    
    INPUT:
    
    - `R` -- A list of lists of vectors.
    
    - `s` -- A vector of the same length.
    
    - `r` -- An integer.
    
    - ``reduction_function`` -- A function that takes a tuple representing an element in `L^\#`
                                and returs a pair of a reduced element in `L^\#` and a sign.
    
    OUTPUT:
    
    - A vector with integer entries that correspond to the elements
      of R in the given order.
    """
    if reduction_function is None :
        return vector([ len([ rp for rp in rs if s.dot_product(rp) == r ])
                        for rs in R ])
    else :
        return vector([ sum( reduction_function(rp)[1] for rp in rs if s.dot_product(rp) == r )
                        for rs in R ])

#===============================================================================
# _local_restriction_matrix
#===============================================================================

def _local_restriction_matrix(R, S, reduction_function = None) :
    r"""
    Return a matrix whose rows correspond to the evaluations of the restriction
    vectors (s, r) in S.
    
    INPUT:
    
    - `R` -- A list of tuples or vectors in L \otimes QQ (with given coordinates).
    
    - `S` -- A list of pairs `(s, r)`, where `s` is a vector,
             and `r` is an integer.

    - ``reduction_function`` -- A function that takes a tuple representing an element in `L^\#`
                                and returs a pair of a reduced element in `L^\#` and a sign.
             
    OUTPUT:
    
    - A matrix with integer entries.
    
    TESTS::

        sage: from psage.modform.jacobiforms.jacobiformd1_fourierexpansion import *
        sage: from psage.modform.jacobiforms.jacobiformd1_fegenerators import _find_complete_set_of_restriction_vectors
        sage: from psage.modform.jacobiforms.jacobiformd1_fegenerators import _local_restriction_matrix                
        sage: indices = JacobiFormD1Indices(QuadraticForm(matrix(2, [2,1,1,2])))
        sage: R = indices._r_representatives
        sage: S = _find_complete_set_of_restriction_vectors(indices.jacobi_index(), R, 4)        
        sage: _local_restriction_matrix(R, S)
        [1 1 1]
        [0 1 1]
        [0 1 1]
        [1 1 1]
        [0 1 1]
        [0 1 1]
        [0 2 0]
    """
    R = [map(vector, rs) for rs in R]
    
    return matrix([ _eval_restriction_vector(R, vector(s), r, reduction_function) for (s, r) in S ])

#===============================================================================
# _global_restriction_matrix
#===============================================================================

def _global_restriction_matrix(precision, S, weight_parity, find_relations = False) :
    r"""
    A matrix that maps the Fourier expansion of a Jacobi form of given precision
    to their restrictions with respect to the elements of S.
    
    INPUT:
    
    - ``precision`` -- An instance of JacobiFormD1Filter.
    
    - `S` -- A list of vectors.
    
    - ``weight_parity`` -- The parity of the weight of the considered Jacobi forms.
    
    - ``find_relation`` -- A boolean. If ``True``, then the restrictions to
                           nonreduced indices will also be computed.
                           
    TESTS::
    
        sage: from psage.modform.jacobiforms.jacobiformd1_fourierexpansion import *
        sage: from psage.modform.jacobiforms.jacobiformd1_fegenerators import _global_restriction_matrix
        sage: precision = JacobiFormD1Filter(5, QuadraticForm(matrix(2, [2,1,1,2])))
        sage: (global_restriction_matrix, row_groups, row_labels, column_labels) = _global_restriction_matrix(precision, [vector((1,0))], 12)
        sage: global_restriction_matrix
        [1 0 0 0 0 0 0 0 0]
        [0 1 2 0 0 0 0 0 0]
        [2 0 2 0 0 0 0 0 0]
        [0 0 2 1 2 0 0 0 0]
        [0 2 0 0 2 0 0 0 0]
        [2 0 0 0 2 1 2 0 0]
        [0 0 2 2 0 0 2 0 0]
        [0 2 0 0 0 0 2 1 2]
        [0 0 0 0 2 2 0 0 2]
        sage: (row_groups, row_labels, column_labels)
        ([((1, 0), 1, 0, 9)], {1: {(0, 0): 0, (3, 0): 5, (3, 1): 6, (2, 1): 4, (2, 0): 3, (1, 0): 1, (4, 1): 8, (1, 1): 2, (4, 0): 7}}, [(0, (0, 0)), (1, (0, 0)), (1, (1, 1)), (2, (0, 0)), (2, (1, 1)), (3, (0, 0)), (3, (1, 1)), (4, (0, 0)), (4, (1, 1))])
    """
    L = precision.jacobi_index()
    weight_parity = weight_parity % 2

    jacobi_indices = [ L(s) for s in S ]
    index_filters = dict( (m, list(JacobiFormD1NNFilter(precision.index(), m, reduced = not find_relations)))
                          for m in Set(jacobi_indices) )
    
    column_labels = list(precision)
    reductions = dict( (l, list()) for l in column_labels )
    for l in precision.monoid_filter() :
        (lred, sign) = precision.monoid().reduce(l)
        reductions[lred].append((l, sign)) 

    row_groups = [ len(index_filters[m]) for m in jacobi_indices ]
    row_groups = [ (s, m, sum(row_groups[:i]), row_groups[i]) for ((i, s), m) in zip(enumerate(S), jacobi_indices) ]
    row_labels = dict( (m, dict( (l, i) for (i, l) in enumerate(index_filters[m]) ))
                       for m in Set(jacobi_indices) )
    dot_products = [ cython_lambda( ' , '.join([ 'int x{0}'.format(i) for i in range(len(s)) ]),
                                    ' + '.join([ '{0} * x{1}'.format(s[i], i) for i in range(len(s)) ]) )
                     for (s, _, _, _) in row_groups ]
    
    restriction_matrix = zero_matrix(ZZ, row_groups[-1][2] + row_groups[-1][3], len(column_labels))
    
    for (cind, l) in enumerate(column_labels) :
        for ((n, r), sign) in reductions[l] :
            for ((s, m, start, length), dot_product) in zip(row_groups, dot_products) :
                row_labels_dict = row_labels[m]
                try :
                    restriction_matrix[start + row_labels_dict[(n, dot_product(*r))], cind] \
                      += 1 if weight_parity == 0 else sign
                except KeyError :
                    pass

    return (restriction_matrix, row_groups, row_labels, column_labels)

#===============================================================================
# _global_relation_matrix
#===============================================================================

def _global_relation_matrix(precision, S, weight_parity) :
    r"""
    Deduce restrictions on the coefficients of a Jacobi form based on
    the specialization to Jacobi form of scalar index.
    
    INPUT:
    
    - ``precision`` -- An instance of JacobiFormD1Filter.
    
    - `S` -- A list of vectors.
    
    - ``weight_parity`` -- The parity of the weight of the considered Jacobi forms.
    """
    L = precision.jacobi_index()
    weight_parity = weight_parity % 2
    
    (mat, row_groups, row_labels, column_labels) = _global_restriction_matrix(precision, S, weight_parity, find_relations = True)
    
    reduced_index_indices = dict( (m, JacobiFormD1NNIndices(m)) for m in Set(m for (_,m,_,_) in row_groups) )
        
    relations = list()
    for (s, m, start, length) in row_groups :
        row_labels_dict = row_labels[m]
        for (l, i) in row_labels_dict.iteritems() :
            (lred, sign) = reduced_index_indices[m].reduce(l)
            if lred == l :
                continue
            
            relations.append(mat.row(start + row_labels_dict[lred]) - (1 if weight_parity == 0 else sign) * mat.row(start + i))

    return (matrix(len(relations), len(column_labels), relations), column_labels)
    
#===============================================================================
# _coefficient_by_restriction
#===============================================================================

_coefficient_by_restriction__cache = dict()
def _coefficient_by_restriction( precision, k, relation_precision = None ) :
    r"""
    Compute the Fourier expansions of Jacobi forms (over `\QQ`) of weight `k` and 
    index `L` (an even symmetric matrix) up to given precision.
    
    ALGORITHM:
    
    See [GKR12]. The algorithm will be applied for precision ``relation_precision``.
    The remaining Fourier coefficients will be determined using fewer restrictions.
    
    NOTE:
    
    In case ``relation_precision`` is not ``None``, we assume that it is of the form
    `\{ (n, r) : n < B \}`, where `B` is the index of the filter.
    
    INPUT:
    
    - ``precision`` -- A filter for Jacobi forms of arbitrary index.
    
    - `k` -- An integer.
    
    - ``relation_precision`` -- A filter for Jacobi forms or ``None`` (default: ``None``).
    
    OUTPUT:
    
    - A list of elements of the corresponding Fourier expansion module.
    
    TESTS:
    
    See ``_test__coefficient_by_restriction`` for further tests.
    
    ::
    
        sage: from psage.modform.jacobiforms.jacobiformd1_fourierexpansion import *
        sage: from psage.modform.jacobiforms.jacobiformd1_fegenerators import _coefficient_by_restriction
        sage: indices = JacobiFormD1Indices(QuadraticForm(matrix(2, [2,1,1,2])))
        sage: precision = indices.filter(20)
        sage: relation_precision = indices.filter(10)
        sage: _coefficient_by_restriction(precision, 10) == _coefficient_by_restriction(precision, 10, relation_precision) 
        True
    """
    L = precision.jacobi_index()
    Lmat = L.matrix()
    Lmat.set_immutable()
    
    global _coefficient_by_restriction__cache
    try :
        expansions = _coefficient_by_restriction__cache[(k, Lmat)]

        if len(expansions) == 0 \
          or expansions[0].precision() >= precision :
            ## TODO: This should be checked in the framework
            return [ f.truncate(precision) for f in expansions ]
    except KeyError :
        pass
     
    
    dim = dimension__jacobi(k, Lmat)
    if dim == 0 :
        return []

    R = precision.monoid()._r_representatives
    S_extended = _find_complete_set_of_restriction_vectors(L, R, reduction_function = precision.monoid().reduce_r)
    S = list()
    for (s, _) in S_extended :
        if s not in S :
            S.append(s) 
    max_S_length = max([L(s) for s in S])

    if relation_precision is None :
        relation_precision = precision

    (global_restriction_matrix__big, row_groups, row_labels, column_labels) = _global_restriction_matrix(precision, S, k)
    (global_relation_matrix, column_labels_relations) = _global_relation_matrix(relation_precision, flatten(L.short_vector_list_up_to_length(max_S_length + 1, True)[1:]), k)
    global_restriction_matrix__big.change_ring(QQ)
    global_relation_matrix.change_ring(QQ)

    if relation_precision == precision :
        assert column_labels == column_labels_relations
    if column_labels != column_labels_relations :
        row_groups__small = [ len(filter( lambda (n,r): n < relation_precision.index(), row_labels[m].keys())) for (_, m, _, _) in row_groups ]
        row_groups__small = [ (sum(row_groups__small[:i]), row_groups__small[i]) for i in range(len(row_groups)) ]
        
        row_labels__small = dict()
        for (m, row_labels_dict) in row_labels.iteritems() :
            row_labels_dict__small = dict()
            
            label_nmb = 0
            for (l, i) in row_labels_dict.iteritems() :
                if l[0] < relation_precision.index() :
                    row_labels_dict__small[l] = (label_nmb, i)
                    label_nmb += 1
            
            row_labels__small[m] = row_labels_dict__small
        
        row_indices__small = list()
        for ((s, m, start, _), (start_small, length_small)) in zip(row_groups, row_groups__small) :
            row_labels_dict = row_labels__small[L(s)]
            row_indices__sub = length_small * [None]
            for (l,(i, i_pre)) in row_labels_dict.iteritems() :
                row_indices__sub[i] = start + i_pre
            row_indices__small += row_indices__sub
        
        global_restriction_matrix = global_restriction_matrix__big.matrix_from_rows_and_columns(
                                                row_indices__small,
                                                [column_labels.index(l) for l in column_labels_relations] )
    else :
        global_restriction_matrix = global_restriction_matrix__big
    
    
    jacobi_indices = [ m for (_, m, _, _) in row_groups ]
    index_filters = dict( (m, JacobiFormD1NNFilter(precision.index(), m)) for m in Set(jacobi_indices) )
    jacobi_forms = dict( (m, JacobiFormsD1NN(QQ, JacobiFormD1NN_Gamma(k, m), prec) )
                         for (m, prec) in index_filters.iteritems() )
    
    forms = list()
    ch1 = JacobiFormD1WeightCharacter(k)
    for (s, m, start, length) in row_groups :
        row_labels_dict = row_labels[m]
        for f in jacobi_forms[m].graded_submodule(None).basis() :
            f = f.fourier_expansion()
            v = vector(ZZ, len(row_labels_dict))
            for (l, i) in row_labels_dict.iteritems() :
                v[i] = f[(ch1, l)]
    
            forms.append(vector(   start*[0] + v.list()
                                 + (row_groups[-1][2] + row_groups[-1][3] - start - length)*[0] ))
    
    
    if relation_precision == precision :
        restriction_expansion = span(forms)
    else :
        restriction_expansion_matrix__big = matrix(forms).transpose()
        restriction_expansion_matrix = restriction_expansion_matrix__big.matrix_from_rows(row_indices__small)
        
        restriction_expansion = restriction_expansion_matrix.column_module() 
    
    restriction_expansions = global_restriction_matrix.column_module().intersection(restriction_expansion)
    restriction_pullback =   global_restriction_matrix.solve_right(restriction_expansions.basis_matrix().transpose()).column_space() \
                           + global_restriction_matrix.right_kernel().change_ring(QQ)
    jacobi_expansions = restriction_pullback.intersection(global_relation_matrix.right_kernel())
    

    if jacobi_expansions.dimension() < dim :
        raise RuntimeError( "There is a bug in the implementation of the restriction method. Dimensions: {0}, {1}".format(jacobi_expansions.dimension(), dim) )
    if jacobi_expansions.dimension() > dim :
        raise ValueError( "Could not construct enough restrictions to determine Fourier expansion uniquely" )
    
    if relation_precision != precision :
        ## In this case, we reconstruct the whole Fourier expansion from partial restrictions
        restriction_coordinates = restriction_expansion_matrix.solve_right( global_restriction_matrix * jacobi_expansions.basis_matrix().transpose() )
        jacobi_expansions__big = global_restriction_matrix__big.solve_right( restriction_expansion_matrix__big * restriction_coordinates )
        
        jacobi_expansions = jacobi_expansions__big.column_module()  
    
    fourier_expansion_module = JacobiFormD1FourierExpansionModule(QQ, k, L)    
    chL = JacobiFormD1WeightCharacter(k, Lmat.nrows())
    
    expansions = list()
    for v in jacobi_expansions.change_ring(QQ).basis() :
        expansions.append( fourier_expansion_module( {chL: dict( (l, c) for (l, c) in zip(column_labels, v) )} ))
        for e in expansions :
            e._set_precision(precision)

    
    _coefficient_by_restriction__cache[(k, Lmat)] = expansions
    
    return expansions

def _test__coefficient_by_restriction(precision, k, relation_precision = None, additional_lengths = 1 ) :
    r"""
    TESTS::
    
        sage: from psage.modform.jacobiforms.jacobiformd1_fourierexpansion import *
        sage: from psage.modform.jacobiforms.jacobiformd1_fegenerators import _test__coefficient_by_restriction

    ::

        sage: indices = JacobiFormD1Indices(QuadraticForm(matrix(2, [2,1,1,2])))
        sage: precision = indices.filter(10)
        sage: _test__coefficient_by_restriction(precision, 10, additional_lengths = 10)
        
    ::

        sage: indices = JacobiFormD1Indices(QuadraticForm(matrix(2, [4,1,1,2])))
        sage: precision = JacobiFormD1Filter(5, indices.jacobi_index())
        sage: _test__coefficient_by_restriction(precision, 40, additional_lengths = 4) # long test
        
    We use different precisions for relations and restrictions::

        sage: indices = JacobiFormD1Indices(QuadraticForm(matrix(2, [2,1,1,2])))
        sage: precision = indices.filter(20)
        sage: relation_precision = indices.filter(2)
        sage: _test__coefficient_by_restriction(precision, 10, relation_precision, additional_lengths = 4)
    """
    from sage.misc.misc import verbose
    
    L = precision.jacobi_index()
    
    if relation_precision is not None and not relation_precision <= precision :
        raise ValueError( "Relation precision must be less than or equal to precision." )

    expansions = _coefficient_by_restriction(precision, k, relation_precision)
    verbose( "Start testing restrictions of {2} Jacobi forms of weight {0} and index {1}".format(k, L, len(expansions)) )
    
    ch1 = JacobiFormD1WeightCharacter(k)
    chL = JacobiFormD1WeightCharacter(k, L.matrix().nrows())
    
    R = precision.monoid()._r_representatives
    S_extended = _find_complete_set_of_restriction_vectors(L, R)

    S = list()
    for (s, _) in S_extended :
        if s not in S :
            S.append(s) 
    max_S_length = max([L(s) for s in S])

    S = L.short_vector_list_up_to_length(max_S_length + 1 + additional_lengths, True)[1:]
    Sold = flatten(S[:max_S_length + 1], max_level = 1)
    Snew = flatten(S[max_S_length + 1:], max_level = 1)
    S = flatten(S, max_level = 1)
    verbose( "Will use the following restriction vectors: {0}".format(S) )
    
    jacobi_forms_dict = dict()
    non_zero_expansions = list()
    for s in S :
        m = L(s)
        verbose( "Restriction to index {0} via {1}".format(m, s) )
        
        try :
            jacobi_forms = jacobi_forms_dict[m]
        except KeyError : 
            jacobi_forms = JacobiFormsD1NN(QQ, JacobiFormD1NN_Gamma(k, m), JacobiFormD1NNFilter(precision.index(), m))
            jacobi_forms_dict[m] = jacobi_forms
        jacobi_forms_module = span([ vector( b[(ch1, k)] for k in jacobi_forms.fourier_expansion_precision() )
                                     for b in map(lambda b: b.fourier_expansion(), jacobi_forms.graded_submodule(None).basis()) ])
        
        fourier_expansion_module = jacobi_forms.fourier_expansion_ambient()
        
        for (i, expansion) in enumerate(expansions) :
            verbose( "Testing restriction of {0}-th form".format(i) )
            restricted_expansion_dict = dict()
            for (n,r) in precision.monoid_filter() :
                rres = s.dot_product(vector(r))
                try :
                    restricted_expansion_dict[(n,rres)] += expansion[(chL,(n,r))]
                except KeyError :
                    restricted_expansion_dict[(n,rres)] = expansion[(chL,(n,r))]
            
            restricted_expansion = vector( restricted_expansion_dict.get(k, 0) for k in jacobi_forms.fourier_expansion_precision() )
            if restricted_expansion not in jacobi_forms_module :
                raise RuntimeError( "{0}-th restricted via {1} is not a Jacobi form".format(i, s) )
            
            if restricted_expansion != 0 :
                non_zero_expansions.append(i)
    
    assert Set(non_zero_expansions) == Set(range(len(expansions)))
 
