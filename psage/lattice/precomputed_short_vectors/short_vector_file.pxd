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

cdef extern from 'short_vector_file.h' :
     cppclass ShortVectorFile :
          ShortVectorFile(object, object, unsigned int)
          ShortVectorFile(object)
          
          void flush()

          object stored_vectors_py()
          object read_vectors_py( unsigned int )
          object write_vectors_py( unsigned int, object )

          unsigned int maximal_vector_length()
          void increase_maximal_vector_length( unsigned int )