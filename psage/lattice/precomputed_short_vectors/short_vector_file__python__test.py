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

import unittest

from psage.lattice.precomputed_short_vectors.short_vector_file__python import ShortVectorFile__python

from sage.misc.temporary_file import tmp_filename

class TestShortVectorFile(unittest.TestCase) :
    
    def setUp( self ) :
        pass

    def test__init( self ) :
        file_name = tmp_filename()
        svf = ShortVectorFile__python( file_name, [[2, 1], [1, 2]], 20 )
        del svf

        svf = ShortVectorFile__python( file_name )

    def test_write_vectors( self ) :
        file_name = tmp_filename()
        svf = ShortVectorFile__python( file_name, [[2, 1], [1, 2]], 20 )

        svf.write_vectors(10, [(1,0), (5, 7)])
        svf.write_vectors(2, [(-1,0), (5, -7)])
        svf.write_vectors(20, 100*[(1,0)])

        self.assertRaises( ValueError,
                           svf.write_vectors,
                           5, [(1,0)] )

        del svf

        file_name = tmp_filename()

        svf = ShortVectorFile__python( file_name, [[2, 1], [1, 4]], 10000 )
        for length in range(2, 10001, 2) :
            svf.write_vectors( length, 5 * [(1, -1)] )
        del svf

    def test_read_vectors( self ) :
        file_name = tmp_filename()
        svf = ShortVectorFile__python( file_name, [[2, 1], [1, 2]], 20 )

        svf.write_vectors(10, [(1,0), (5, 7)])
        self.assertEqual( svf.read_vectors(10),
                          [(1,0), (5, 7)] )
        self.assertRaises( ValueError,
                           svf.read_vectors,
                           6 )
        
        del svf
        svf = ShortVectorFile__python( file_name )
        self.assertEqual( svf.read_vectors(10),
                          [(1,0), (5, 7)] )

    def test_stored_vectors( self ) :
        file_name = tmp_filename()
        svf = ShortVectorFile__python( file_name, [[2, 1], [1, 2]], 20 )

        svf.write_vectors(10, [(1,0), (5, 7)])
        self.assertEqual( svf.stored_vectors(),
                          [(10, 2)] )

        del svf
        svf = ShortVectorFile__python( file_name )
        self.assertEqual( svf.stored_vectors(),
                          [(10, 2)] )
        del svf
        
    def test_maximal_vector_length( self ) :
        file_name = tmp_filename()
        svf = ShortVectorFile__python( file_name, [[2, 1], [1, 2]], 20 )

        self.assertEqual( svf.maximal_vector_length(),
                          20 )
        del svf

    def test_increase_maximal_vector_length( self ) :
        file_name = tmp_filename()
        svf = ShortVectorFile__python( file_name, [[2, 1], [1, 2]], 20 )

        svf.write_vectors( 10, 20 * [(12, -13)] )
        svf.increase_maximal_vector_length( 100 )

        self.assertEqual( svf.maximal_vector_length(),
                          100 )
        self.assertEqual( svf.stored_vectors(),
                          [(10, 20)] )
        self.assertEqual( svf.read_vectors(10),
                          20 * [(12, -13)] )

        del svf
        svf = ShortVectorFile__python( file_name )
        self.assertEqual( svf.maximal_vector_length(),
                          100 )
        self.assertEqual( svf.stored_vectors(),
                          [(10, 20)] )
        self.assertEqual( svf.read_vectors(10),
                          20 * [(12, -13)] )

        del svf
        
if __name__ == '__main__':
    unittest.main(verbosity = 2)
