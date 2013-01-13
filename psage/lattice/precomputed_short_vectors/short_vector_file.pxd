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

cdef extern from 'short_vector_file.h' :
     cppclass ShortVectorFile :
          ShortVectorFile(object, object, unsigned int)
          ShortVectorFile(object)
          
          object stored_vectors_py()
          object read_vectors_py( unsigned int )
          object write_vectors_py( unsigned int, object )

          unsigned int maximal_vector_length()
          void increase_maximal_vector_length( unsigned int )
