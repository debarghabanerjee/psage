#clang C++

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

from cython.operator cimport dereference as deref, preincrement as inc
from operator import neg
from __builtin__ import map as py_map


include "interrupt.pxi"

include "enumerate_short_vectors.pxd"

cpdef object enumerate_short_vectors__python( object lattice, lower_bound, upper_bound, up_to_sign = False ) :
    r"""
    Enumerate vectors of minimal norm ``lower_bound`` and maximal norm ``upper_bound``, either up to sign or not.

    INPUT:
    
    - ``lattice`` - a list of list of integers which corresponds to the Gram matrix of a binary quadratic form.

    - ``lower_bound`` - a positive integer.

    - ``upper_bound`` - a positive integer.

    - ``up_to_sign`` - a boolean (default: ``False``).

    OUPUT:

    - A dictionary mapping integers to a list of tuples.  Each tuple corresponds to a vector.

    """
    dim = len(lattice)
    for row in lattice :
        if len(lattice) != dim :
            raise ValueError( "lattice must be a list of list, representing a square matrix" )


    cdef vector[vector[int]] lattice__cpp
    cdef vector[int] row__cpp

    for row in lattice :
        row__cpp = vector[int]()
        for e in row :
            row__cpp.push_back( e )
        lattice__cpp.push_back( row__cpp )

    cdef object svs = dict()

    if upper_bound < lower_bound :
        return dict()
    if upper_bound == 0 :
        svs[0] = [ tuple( [0]*dim ) ]
        return svs

    if lower_bound <= 0 :
        lower_bound = 2
        add_zero_vector = True
    else :
        add_zero_vector = False

    cdef map[uint, vector[vector[int]]] result

    sig_on()
    enumerate_short_vectors( lattice__cpp, lower_bound, upper_bound, result )
    sig_off()

    cdef object res_v
    cdef size_t ind, v_ind

    for ind in range(lower_bound, upper_bound + 1, 2) :
        svs[ind] = list()

    cdef map[uint, vector[vector[int]]].iterator result_it, result_itend
    cdef vector[vector[int]] vecs
    cdef vector[vector[int]].iterator vecs_it, vecs_itend
    cdef vector[int] vec
    cdef vector[int].iterator vec_it, vec_itend
    cdef object vecs_py
    cdef object vec_py

    result_it = result.begin()
    result_itend = result.end()
    while ( result_it != result_itend ) :
        vecs_py = list()
        vecs = deref(result_it).second

        vecs_it = vecs.begin()
        vecs_itend = vecs.end()
        while ( vecs_it != vecs_itend ) :
            vec = deref( vecs_it )
            vec_py = list()
            vec_it = vec.begin()
            vec_itend = vec.end()
            while ( vec_it != vec_itend ) :
                vec_py.append( deref(vec_it) )
                inc(vec_it)

            vecs_py.append( tuple(vec_py) )
            inc(vecs_it)

        svs[ deref(result_it).first ] = vecs_py
        inc(result_it)

    if not up_to_sign :
        for length in svs.keys() :
            svs[length] = svs[length] + [ tuple(py_map(neg, sv)) for sv in svs[length] ]
    
    if add_zero_vector :
        svs[0] = [ tuple( [0]*dim ) ]

    return svs
