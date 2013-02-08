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

    print "enumerate"
    sig_on()
    enumerate_short_vectors( lattice__cpp, lower_bound, upper_bound, result )
    sig_off()
    print "done"

    cdef object svs = dict()
    cdef object res_v
    cdef size_t ind, v_ind

    for ind in range(result.size()) :
        res_v = list()

        for v_ind in range( dim ) :
            res_v.append( result[ind].first[v_ind] )
        
        svs[ result[ind].second ].append(tuple(res_v))

    
    return svs
