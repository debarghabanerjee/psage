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

from libcpp.pair cimport pair
from libcpp.vector cimport vector

cdef extern from * :
    ctypedef int uint "unsigned int"
    ctypedef vector[vector[int]] const_vector_vector_int "const std::vector<std::vector<int>>"

cdef extern from 'enumeate_short_vectors.h' :
   enumerate_short_vectors( const_vector_vector_int&, uint, uint, vector[pair[vector[int], uint]]& )
