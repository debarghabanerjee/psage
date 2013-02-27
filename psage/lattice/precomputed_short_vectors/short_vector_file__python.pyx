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

from sage.rings.all import ZZ

include "interrupt.pxi"

include "short_vector_file_py.pxd"

cdef class ShortVectorFile__python :
    cdef ShortVectorFilePy *this_ptr

    def __cinit__( self, output_file_name, lattice = None, maximal_vector_length = None ) :
        if lattice is None :
            f = file(output_file_name, 'r')
            f.close()
            
            sig_on()
            self.this_ptr = new ShortVectorFilePy(output_file_name)
            sig_off()
        else :
            lattice = self._check_lattice( lattice )

            f = file(output_file_name, 'a')
            f.close()
            
            sig_on()
            self.this_ptr = new ShortVectorFilePy(output_file_name, lattice, maximal_vector_length)
            sig_off()
            
    def __dealloc__( self ) :
        del self.this_ptr
    
    def _check_lattice( self, lattice ) :
        for i in range(len(lattice)) :
            for j in range(len(lattice)) :
                if lattice[i][j] not in ZZ or (i == j and lattice[i][j] % 2 != 0) :
                    raise ValueError( "lattice must be even" )

        return [[int(e) for e in row] for row in lattice]

    def flush( self ) :
        sig_on()
        self.this_ptr.flush()
        sig_off()
    
    def stored_vectors( self ) :
        sig_on()
        return_val = self.this_ptr.stored_vectors_py()
        sig_off()
        
        return return_val
    
    def read_vectors( self, length ) :
        if length % 2 != 0 :
            return []
        
        sig_on()
        return_val = self.this_ptr.read_vectors_py( int(length) )
        sig_off()

        if return_val is None :
            raise ValueError( "vectors of length {0} are not stored".format(length) )
                
        return return_val
    
    def write_vectors( self, length, vectors ) :
        if length % 2 == 1 :
            raise ValueError( "Only vectors of even length can be stored" )
        if length <= 0 :
            raise ValueError( "Only vectors of positive length can be stored" )
        
        vectors = [ tuple(map(int, v)) for v in vectors ]
        
        sig_on()
        written = self.this_ptr.write_vectors_py( int(length), vectors )
        sig_off()

        if not written :
            raise ValueError( "Could not write list {0} of vectors of length {1}" \
                              .format(vectors, length) )

    def maximal_vector_length( self ) :
        sig_on()
        return_val = self.this_ptr.maximal_vector_length()
        sig_off()
        
        return int(return_val)
    
    def increase_maximal_vector_length( self, maximal_vector_length ) :
        if maximal_vector_length % 2 == 1 :
            maximal_vector_length = maximal_vector_length - 1
        sig_on()
        self.this_ptr.increase_maximal_vector_length( int(maximal_vector_length) )
        sig_off()
