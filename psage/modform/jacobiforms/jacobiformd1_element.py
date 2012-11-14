r"""
Elements of modules of Jacobi forms.

AUTHORS:

- Martin Raum (2012 - 11 - 13) Initial version.
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

from psage.modform.fourier_expansion_framework.modularforms.modularform_element import ModularForm_generic
from psage.modform.vector_valued.discriminant_form import DiscriminantForm
from psage.modform.jacobiforms.jacobiformd1_fourierexpansion import JacobiFormD1WeightCharacter
from sage.rings.all import ZZ
import operator

class JacobiFormD1_class ( ModularForm_generic ) :

    def theta_decomposition(self) :
        r"""
        Apply the theta decomposition to a Fourier expansion `\phi` and, if not ``None`` apply the
        isomorphism of discriminant modules ``mor`` to the labels of the components.

        OUTPUT:

        - A dictionary whose keys are lifts of elements of a discrimiant group, and whose
          values are dictionaries whose keys are rationals (the exponents of `q`) and whose
          values are also rationals (the corresponding coefficients). 
        """
        jacobi_index = self.parent().type().index()

        fe = self.fourier_expansion()
        fe_ambient = fe.parent()
        disc_form = DiscriminantForm(jacobi_index.matrix())

        Ladj = fe_ambient.action()._Ladjoint()
        Ldet = jacobi_index.det()
        ch = JacobiFormD1WeightCharacter( self.parent().type().weight(), jacobi_index.dim() )

        f = dict( (disc_form._from_jacobi_index(rs[0]), dict())
                  for rs in fe_ambient.action()._r_representatives )

        for (n, r) in fe.precision() :
            b = disc_form._from_jacobi_index(r)
            disc = n - Ladj(r) / ZZ(2 * Ldet)
            f[b][disc] = fe[(ch,(n,r))]
            if b != -b :
                f[-b][disc] = fe[(ch,(n,tuple(map(operator.neg, r))))]

        return f








