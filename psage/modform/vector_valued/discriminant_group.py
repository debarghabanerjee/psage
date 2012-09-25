r"""
Discriminant groups for arbitrary lattices.

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

from sage.groups.additive_abelian.additive_abelian_group import AdditiveAbelianGroup_class, \
                                                    cover_and_relations_from_invariants
from sage.matrix.all import identity_matrix, matrix
from sage.modules.all import FreeModule, vector
from sage.rings.all import QQ, ZZ
from copy import copy
import operator

#===============================================================================
# DiscriminantGroup
#===============================================================================

class DiscriminantGroup( AdditiveAbelianGroup_class ) :
    
    def __init__(self, L) :
        r"""
        A discriminant group attached to an integral lattice `L` (with a
        homomorphism from `L^\#` to it)
        
        INPUT:
        
        - `L` -- A symmetric matrix over `\ZZ` with even diagonal entries.
        """
        self._L = copy(L)
        assert L.base_ring() is ZZ
        (elementary_divisors, _, pre_basis) = L.smith_form()
        self._dual_basis = pre_basis * elementary_divisors.inverse()
        elementary_divisors = elementary_divisors.diagonal()
        
        AdditiveAbelianGroup_class.__init__(self, *cover_and_relations_from_invariants(elementary_divisors))
    
    def lattice(self) :
        return self._L

    def unimodular_transformation(self, u) :
        r"""
        Apply the transformation `u` to the underlying lattice `L`: `u^\tr L u`.
        
        INPUT:
        
        - `u` -- A unimodular matrix.
        
        OUTPUT:
        
        - A pair of a discriminant group and a homomorphism from self to this group.
        """
        n_disc = DiscriminantGroup(u.transpose() * self._L * u)
        
        uinv = u.inverse()
        basis_images = [ sum(map(operator.mul, b.lift(), self._dual_basis.columns()))
                         for b in self.smith_form_gens() ]
        basis_images = [ n_disc._dual_basis.solve_right(uinv * b).list()
                         for b in basis_images ]
        
        coercion_hom = self.hom([ sum(map( operator.mul, map(ZZ, b), map(n_disc, FreeModule(ZZ, self._L.nrows()).gens()) ))
                                  for b in basis_images ])
        
        return (n_disc, coercion_hom)

    def _to_jacobi_indices(self) :
        r"""
        Return indices `r` that correspond to the generators of this discriminant group.
        
        OUTPUT:
        
        - A list of tuples.
        """
        return map(tuple, (self._L * self._dual_basis).columns())

    def add_unimodular_lattice(self, L = None) :
        r"""
        Add a unimodular lattice (which is not necessarily positive definite) to the underlying lattice
        and the resulting discriminant group and an isomorphism to ``self``.
        
        INPUT:
        
        - `L` -- The Gram matrix of a unimodular lattice or ``None`` (default: ``None``).
                 If ``None``, `E_8` will be added.
        
        OUTPUT:
        
        - A pair of a discriminant group and a homomorphism from self to this group.
        """
        if L is None :
            L = matrix(8, [ 2,-1, 0, 0,  0, 0, 0, 0,
                           -1, 2,-1, 0,  0, 0, 0, 0,
                            0,-1, 2,-1,  0, 0, 0,-1,
                            0, 0,-1, 2, -1, 0, 0, 0,
                            0, 0, 0,-1,  2,-1, 0, 0,
                            0, 0, 0, 0, -1, 2,-1, 0,
                            0, 0, 0, 0,  0,-1, 2, 0,
                            0, 0,-1, 0,  0, 0, 0, 2])
        else :
            if L.base_ring() is not ZZ or all(e % 2 == 0 for e in L.diagonal()) \
              or L.det() != 1 :
                 raise ValueError( "L must be the Gram matrix of an even unimodular lattice")   
        
        nL = self._L.block_sum(L)
        n_disc = DiscriminantGroup(nL)
        
        basis_images = [ sum(map(operator.mul, b.lift(), self._dual_basis.columns()))
                         for b in self.smith_form_gens() ]
        basis_images = [ n_disc._dual_basis.solve_right(vector(QQ, b.list() + L.nrows() * [0])).list()
                         for b in basis_images ]
        coercion_hom = self.hom([ sum(map( operator.mul, map(ZZ, b), map(n_disc, FreeModule(ZZ, nL.nrows()).gens()) ))
                                  for b in basis_images  ])
        
        return (n_disc, coercion_hom)
    
    def split_off_unimodular_lattice(self, L_basis_matrix, check = True) :
        r"""
        Split off a unimodular lattice that is an orthogonal summand of self.
        
        INPUT:
        
        - ``L_basis_matrix`` -- A matrix over `\ZZ` whose number of rows equals the
                                size of the underlying lattice.
        
        - ``check`` -- A boolean (default: ``True``).  If ``True`` it will be
                       checked whether the given sublattice is a direct summand.

        
        OUTPUT:
        
        - A pair of a discriminant group and a homomorphism from self to this group.
        """
        if check and L_basis_matrix.base_ring() is not ZZ :
            raise ValueError( "L_basis_matrix must define a sublattice" )
            
        pre_bases = matrix(ZZ, [ [ l * self._L * b for l in L_basis_matrix.columns()]
                                 for b in FreeModule(ZZ, self._L.nrows()).gens() ]) \
                            .augment(identity_matrix(ZZ, self._L.nrows())).echelon_form()
        
        if check and pre_bases[:L_basis_matrix.ncols(),:L_basis_matrix.ncols()] != identity_matrix(ZZ, L_basis_matrix.ncols()) :
            raise ValueError( "The sublattice defined by L_basis_matrix must be unimodular" )
            
        K_basis_matrix = pre_bases[L_basis_matrix.ncols():,L_basis_matrix.ncols():].transpose()
        if check and L_basis_matrix.column_module() + K_basis_matrix.column_module() != FreeModule(ZZ, self._L.nrows()) :
            raise ValueError( "The sublattice defined by L_basis_matrix must be an orthogonal summand" )

        K = K_basis_matrix.transpose() * self._L * K_basis_matrix
        n_disc = DiscriminantGroup(K)
        
        total_basis_matrix = L_basis_matrix.change_ring(QQ).augment(K_basis_matrix * n_disc._dual_basis)

        basis_images = [ sum(map(operator.mul, b.lift(), self._dual_basis.columns()))
                         for b in self.smith_form_gens() ]
        basis_images = [ total_basis_matrix.solve_right(b)[-K_basis_matrix.ncols():]
                         for b in basis_images ]
        
        coercion_hom = self.hom([ sum(map(operator.mul, map(ZZ, b), map(n_disc, FreeModule(ZZ, K.nrows()).gens()) ))
                                  for b in basis_images ])
        
        return (n_disc, coercion_hom)

    def _repr_(self) :
        return "Discrimiant group of lattice of size {0} ( isomorphic to {1} )".format(self._L.nrows(), self.short_name())
    