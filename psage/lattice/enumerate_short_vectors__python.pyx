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

from cython.operator cimport dereference as deref

include "interrupt.pxi"

include "enumerate_short_vectors.pxd"

cpdef object enumerate_short_vectors__python( object lattice, lower_bound, upper_bound ) :
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


    cdef vector[pair[vector[int], uint]] result

    enumerate_short_vectors( lattice_cpp, lower_bound, upper_bound, result )

    object svs
    object res_v
    cdef vector[int]::iterator v_it
    cdef vector[int]::const_iterator v_it_end

    cdef result::iterator it = result.begin()
    cdef result::const_iterator it_end = result.cend()
    while it != it_end :
        res_v = list()

        v_it = it.first().begin()
        v_it_end = it.first().cend()
        while v_it != v_it_end :
            res_v.append( deref(v_it) )
            v_it += 1
        
        svs[ it.second() ].append(tuple(res_v))

        it += 1
    
    return svs
