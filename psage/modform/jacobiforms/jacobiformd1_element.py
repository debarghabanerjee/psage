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
        Apply the theta decomposition to a Jacobi form `\phi`.

        OUTPUT:

        - A dictionary whose keys are lifts of elements of a discrimiant group, and whose
          values are dictionaries whose keys are rationals (the exponents of `q`) and whose
          values are also rationals (the corresponding coefficients). 
        """
        return self.parent()._theta_decomposition(self.fourier_expansion())








