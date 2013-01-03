from sage.interfaces.magma import magma
from sage.rings.all import Integer
from short_vector_file__python import ShortVectorFile__python
import os

def precompute_short_vectors__magma( lattice, lengths, output_file_name, maximal_length = None ) :
    r"""
    INPUT:

    - ``lattice`` -- A matrix with integral entries.
    """
    if maximal_length is None :
        maximal_length = max(lengths)

    lattice =  map(list, lattice.rows())

    if output_file_name in os.listdir('.') :
        lattice_file = ShortVectorFile__python( output_file_name )
        if lattice_file.maximal_vector_length() < maximal_length :
            lattice_file.increase_maximal_vector_length( maximal_length )
    else :
        lattice_file = ShortVectorFile__python( output_file_name, lattice, maximal_length ) 

    magma.eval( "L := LatticeWithGram(Matrix({0}));".format(lattice) )
    
    for m in lengths :
        svs = magma.eval( "ShortVectors(L, {0}, {0});".format(m) ).split('\n')[1:-1]
        svs = [sv.split(',')[0].lstrip()[2:-1].split(" ") for sv in svs]
        svs = [filter( lambda s: s != "", sv ) for sv in svs]
        svs = [map(lambda e: Integer(e), sv) for sv in svs]

        lattice_file.write_vectors( m, svs )
