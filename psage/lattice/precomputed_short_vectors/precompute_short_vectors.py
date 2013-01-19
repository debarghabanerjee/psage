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

import os
from sage.interfaces.magma import magma
from sage.rings.all import Integer
from sage.quadratic_forms.all import QuadraticForm
from psage.lattice.precomputed_short_vectors.short_vector_file__python import ShortVectorFile__python

################################################################################
### precompute_short_vectors__magma
################################################################################

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

    lattice_file.flush()

################################################################################
### precompute_short_vectors__sage
################################################################################

def precompute_short_vectors__sage( lattice, lengths, output_file_name, maximal_length = None ) :
    r"""
    INPUT:

    - ``lattice`` -- A matrix with integral entries.

    EXAMPLE::

        sage: from psage.lattice.precomputed_short_vectors import *
        sage: file_name = tmp_filename()
        sage: L = matrix(ZZ, [[2, 1], [1, 2]])
        sage: precompute_short_vectors__sage( L, range(2, 11, 2), file_name, maximal_length = 100 )
        sage: svf = ShortVectorFile__python( file_name )
        sage: svf.stored_vectors()
        [(2, 3), (4, 0), (6, 3), (8, 3), (10, 0)]
        sage: svf.read_vectors(8)
        [(0, 2), (-2, 2), (2, 0)]
    """
    if maximal_length is None :
        maximal_length = max(lengths)

    lattice_list =  map(list, lattice.rows())

    if output_file_name in os.listdir('.') :
        lattice_file = ShortVectorFile__python( output_file_name )
        if lattice_file.maximal_vector_length() < maximal_length :
            lattice_file.increase_maximal_vector_length( maximal_length )
    else :
        lattice_file = ShortVectorFile__python( output_file_name, lattice_list, maximal_length ) 

    qf = QuadraticForm(lattice)
    
    for (m, svs) in enumerate( qf.short_vector_list_up_to_length( max(lengths) / 2 + 1, True ) ) :
        if 2 * m in lengths :
            lattice_file.write_vectors( 2 * m, svs )
    
    lattice_file.flush()
