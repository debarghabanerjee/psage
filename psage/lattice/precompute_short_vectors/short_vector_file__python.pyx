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
            
    def __del__( self ) :
        print "called pre2"
        del self.this_ptr
    
    def __deallocpp__( self ) :
        print "called pre"
        del self.this_ptr

    def increase_maximal_vector_length( self, maximal_vector_length ) :
        sig_on()
        self.this_ptr.increase_maximal_vector_length( maximal_vector_length )
        sig_off()
        
    def write_vectors( self, length, vectors ) :
        sig_on()
        self.this_ptr.write_vectors( length, vectors )
        sig_off()
