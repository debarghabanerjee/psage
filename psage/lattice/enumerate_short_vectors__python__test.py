#===============================================================================
# 
# Copyright (C) 2013 Martin Raum
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

from psage.lattice.enumerate_short_vectors__python import enumerate_short_vectors__python
from sage.matrix.all import matrix
from sage.modules.all import vector
from sage.quadratic_forms.all import QuadraticForm


class TestEnumerateShortVectors( unittest.TestCase ) :
    
    def setUp( self ) :
        pass

    def test__enumerate_short_vectors__python__binary( self ) :
        self._test_quadratic_form( matrix(2, [2, 1, 1, 2]), 10, 20 )
        self._test_quadratic_form( matrix(2, [2, 1, 1, 2]), 1000, 1954 )
        self._test_quadratic_form( 5 * matrix(2, [2, 1, 1, 2]), 100, 200 )

        self._test_quadratic_form( matrix(2, [2, 1, 1, 10]), 50, 100 )
        self._test_quadratic_form( matrix(2, [4, 1, 1, 10]), 50, 100 )
        self._test_quadratic_form( matrix(2, [10, 1, 1, 4]), 50, 100 )

    def test__enumerate_short_vectors__python__ternary( self ) :
        self._test_quadratic_form( matrix(3, [2, 1, 1,  1, 2, 1,  1, 1, 2]), 20, 40 )
        # segmentation fault:
        # self._test_quadratic_form( matrix(3, [2, 1, 1,  1, 2, 1,  1, 1, 2]), 2, 4 )
        # infinite loop:
        # self._test_quadratic_form( matrix(3, [2, 1, 1,  1, 2, 1,  1, 1, 2]), 2, 6 )

    def test__enumerate_short_vectors__python__quartary( self ) :
        self._test_quadratic_form( matrix(2, [10, 1, 1, 4]).
                                    block_sum( matrix(2, [2, 1, 1, 10]) ),
                                   20, 40 )
        ## the test fails for length 34
        ## the vector (0, -2, -3, 0) is not included
        

    def _test_quadratic_form( self, qf_mat, lower_bound, upper_bound ) :
        qf = QuadraticForm( qf_mat )
        qf_svss = qf.short_vector_list_up_to_length( upper_bound // 2 + 1 )

        svss = enumerate_short_vectors__python( map(list, qf_mat.rows()), lower_bound, upper_bound )
        self.assertEqual( list(sorted(svss.keys())),
                          range(lower_bound, upper_bound + 1, 2) )

        for (length, svs) in svss.iteritems() :
            qf_svs = qf_svss[length // 2]
            self.assertEqual( len(qf_svs),
                              2 * len(svs) )
            for v in svs :
                self.assertIn( vector(v),
                               qf_svs )

if __name__ == '__main__':
    unittest.main(verbosity = 2)
