r"""
Modules of Jacobi forms of fixed index and weight.

AUTHOR :
    - Martin Raum (2012 - 09 - 11) Initial version based on code for
                                   Jacobi forms of scalar index.
"""

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

import operator
from psage.modform.fourier_expansion_framework.modularforms.modularform_ambient import ModularFormsModule_generic
from psage.modform.jacobiforms.jacobiformd1_fourierexpansion  import JacobiFormD1WeightCharacter
from psage.modform.vector_valued import DiscriminantForm
from sage.rings.integer_ring import ZZ

#===============================================================================
# JacobiFormD1Module
#===============================================================================

class JacobiFormD1Module( ModularFormsModule_generic ) :
    def _theta_decomposition(self, phi) :
        r"""
        Apply the theta decomposition to a Fourier expansion `\phi`.

        OUTPUT:

        - A dictionary whose keys are lifts of elements of a discrimiant group, and whose
          values are dictionaries whose keys are rationals (the exponents of `q`) and whose
          values are also rationals (the corresponding coefficients). 
        """
        jacobi_index = self.type().index()

        fe_ambient = phi.parent()
        disc_form = DiscriminantForm(jacobi_index.matrix())

        Ladj = fe_ambient.action()._Ladjoint()
        Ldet = jacobi_index.det()
        ch = JacobiFormD1WeightCharacter( self.type().weight(), jacobi_index.dim() )

        f = dict( (disc_form._from_jacobi_fourier_index(rs[0]), dict())
                  for rs in fe_ambient.action()._r_representatives )

        for (n, r) in phi.precision() :
            b = disc_form._from_jacobi_fourier_index(r)
            disc = n - Ladj(r) / ZZ(2 * Ldet)
            f[b][disc] = phi[(ch,(n,r))]
            if b != -b :
                f[-b][disc] = phi[(ch,(n,tuple(map(operator.neg, r))))]

        return f
