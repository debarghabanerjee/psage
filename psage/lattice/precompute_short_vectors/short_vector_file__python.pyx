#clang C++

include "interrupt.pxi"

include "short_vector_file.pxd"

cdef class ShortVectorFile__python :
    cdef ShortVectorFile *this_ptr
    
    def __cinit__( self, output_file_name, lattice = None, maximal_vector_length = None ) :
        if lattice is None :
            sig_on()
            self.this_ptr = new ShortVectorFile(output_file_name)
            sig_off()
        else :
            sig_on()
            f = file(output_file_name, 'a')
            f.close()
            self.this_ptr = new ShortVectorFile(output_file_name, lattice, maximal_vector_length)
            sig_off()
            
    def __deallocpp__( self ) :
        print "called pre"
        del self.this_ptr
    
    def stored_vectors( self ) :
        sig_on()
        return_val = self.this_ptr.stored_vectors()
        sig_off()
        
        return return_val
    
    def read_vectors( self, length ) :
        sig_on()
        return_val = self.this_ptr.read_vectors( int(length) )
        sig_off()
        
        return return_val
    
    def write_vectors( self, length, vectors ) :
        vectors = [ tuple(map(int, v)) for v in vectors ]
        
        sig_on()
        written = self.this_ptr.write_vectors( int(length), vectors )
        sig_off()

        if not written :
            raise ValueError( "Could not write list {0} of vectors of length {1}" \
                              .format(vectors, length) )

    def maximal_vector_length( self ) :
        sig_on()
        return_val = self.this_ptr.maximal_vector_length()
        sig_off()
        
        return return_val
    
    def increase_maximal_vector_length( self, maximal_vector_length ) :
        sig_on()
        self.this_ptr.increase_maximal_vector_length( int(maximal_vector_length) )
        sig_off()
