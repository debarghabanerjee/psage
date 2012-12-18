################################################################################
#       Copyright (C) 2012 Martin Raum <martin@raum-brothers.eu>
#
#  Distributed under the terms of the GNU General Public License (GPL)
#
#    This code is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    General Public License for more details.
#
#  The full text of the GPL is available at:
#
#                  http://www.gnu.org/licenses/
################################################################################

from libc.stdint cimport int64_t, uint64_t

cdef extern from 'short_vector_file.h' :
     cppclass ShortVectorFile :
          ShortVectorFile(object, object, unsigned int64)
          ShortVectorFile(object)
          object write_vectors( uint64_t, object )
          uint64_t maximal_vector_length()
          void increase_maximal_vector_length( uint64_t )
