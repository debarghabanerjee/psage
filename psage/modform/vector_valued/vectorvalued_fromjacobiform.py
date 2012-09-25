r"""
Extract the Fourier expansion of a vector valued modular forms from a Jacobi form.

TODO:

- This should eventually be an element constructor for the module of vector valued
  modular forms.

AUTHOR:

- Martin Raum : Initial version.
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

from psage.modform.vector_valued.discriminant_group import DiscriminantGroup
from psage.modform.jacobiforms.jacobiformd1nn_fourierexpansion import JacobiFormD1WeightCharacter
from sage.rings.all import ZZ
import operator

#===============================================================================
# to_vector_valued_modular_form
#===============================================================================

def to_vector_valued_modular_form(phi, phi_weight, mor = None) :
    r"""
    Apply the theta decomposition to a Fourier expansion `\phi` and, if not ``None`` apply the
    isomorphism of discriminant modules ``mor`` to the labels of the components.
    
    INPUT:
    
    - ``phi`` -- A Jacobi form with lattice index.
    
    - ``phi_weight`` - The weight of ``phi``.
    
    - ``mor`` -- Either ``None`` or an isomorphism of discrimiant groups (default: ``None``).
    
    OUTPUT:
    
    - A dictionary whose keys are lifts of elements of a discrimiant group, and whose
      values are dictionaries whose keys are rationals (the exponents of `q`) and whose
      values are also rationals (the corresponding coefficients). 
    """
    if mor is not None and not phi.parent().monoid().jacobi_index().matrix() == mor.codomain().lattice() :
        raise ValueError( "The Jacobi index and the lattice attached to the codomain of the morphism must equal.")
    
    if mor is None :
        disc_group = DiscriminantGroup(phi.parent().monoid().jacobi_index().matrix())
    else :
        disc_group = mor.codomain()
    
    Ladj = phi.precision().monoid()._Ladjoint()
    Ldet = phi.precision().jacobi_index().det()
    ch = JacobiFormD1WeightCharacter(phi_weight, phi.precision().jacobi_index().matrix().nrows())
    
    f = dict( (disc_group._from_jacobi_index(rs[0]), dict())
              for rs in phi.precision().monoid()._r_representatives )
    
    for (n, r) in phi.precision() :
        b = disc_group._from_jacobi_index(r)
        disc = n - Ladj(r) / ZZ(4 * Ldet)
        f[b][disc] = phi[(ch,(n,r))]
        if b != -b :
            f[-b][disc] = phi[(ch,(n,tuple(map(operator.neg, r))))]
    
    ## coerce disc group
    if mor is None :
        return f
    else :
        return dict( (b, f[mor(b)]) for b in mor.domain() )
