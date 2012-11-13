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
                                                    cover_and_relations_from_invariants, AdditiveAbelianGroupElement
from sage.matrix.all import identity_matrix, matrix, diagonal_matrix
from sage.misc.all import cached_method, prod, flatten
from sage.modules.all import FreeModule, vector
from sage.modular.arithgroup.all import GammaH
from sage.quadratic_forms.all import QuadraticForm
from sage.rings.all import lcm
from sage.rings.all import QQ, ZZ, CyclotomicField, PolynomialRing
from copy import copy
import operator


#===============================================================================
# DiscriminantFormElement
#===============================================================================

class DiscrimiantFormElement( AdditiveAbelianGroupElement ) :

    def elementary_representation(self) :
        return tuple(self.lift())

#===============================================================================
# DiscriminantForm
#===============================================================================

class DiscriminantForm( AdditiveAbelianGroup_class ) :

    Element = DiscrimiantFormElement

    def __init__(self, L) :
        r"""
        A discriminant group attached to an integral lattice `L` (with a
        homomorphism from `L^\#` to it)
        
        INPUT:
        
        - `L` -- A symmetric matrix over `\ZZ` with even diagonal entries
                or a quadratic form.
        """
        try :
            L = L.matrix()
        except AttributeError :
            pass

        self._L = copy(L)
        assert L.base_ring() is ZZ
        (elementary_divisors, _, pre_basis) = L.smith_form()
        self._dual_basis = pre_basis * elementary_divisors.inverse()
        elementary_divisors = elementary_divisors.diagonal()
        
        AdditiveAbelianGroup_class.__init__(self, *cover_and_relations_from_invariants(elementary_divisors))
    
    def positive_quadratic_form(self) :
        r"""
        Find the Gram matrix of a positive definite quadratic form
        which is stabily equivalent to `L`.
        """
        E8_gram = matrix(ZZ, 8,
                [2, -1, 0, 0,  0, 0, 0, 0,
                -1, 2, -1, 0,  0, 0, 0, 0,
                0, -1, 2, -1,  0, 0, 0, -1,
                0, 0, -1, 2,  -1, 0, 0, 0,
                0, 0, 0, -1,  2, -1, 0, 0,
                0, 0, 0, 0,  -1, 2, -1, 0,
                0, 0, 0, 0,  0, -1, 2, 0,
                0, 0, -1, 0,  0, 0, 0, 2])
        E8 = QuadraticForm(E8_gram)
        L = self._L
        
        ## This is a workaround since GP crashes
        is_positive_definite = lambda L: all( L[:n,:n].det() > 0 for n in range(1, L.nrows() + 1) )
        def _split_hyperbolic(L) :
            cur_cor = 2
            Lcor = L + cur_cor * identity_matrix(L.nrows())
            while not is_positive_definite(Lcor) :
                cur_cor += 2
                Lcor = L + cur_cor * identity_matrix(L.nrows())

            a = FreeModule(ZZ, L.nrows()).gen(0)
            if a * L * a >= 0 :
                Lcor_length_inc = max(3, a * L * a)
                cur_Lcor_length = Lcor_length_inc
                while True :
                    short_vectors = flatten(QuadraticForm(Lcor).short_vector_list_up_to_length( a * Lcor * a )[cur_Lcor_length - Lcor_length_inc: cur_Lcor_length], max_level = 1)
                    for a in short_vectors :
                        if a * L * a < 0 :
                            break
                    else :
                        continue
                    break
            n = -a * L * a // 2

            short_vectors = E8.short_vector_list_up_to_length(n + 1)[-1]

            for v in short_vectors :
                for w in short_vectors :
                    if v * E8_gram * w == 2 * n - 1 :
                        LE8_mat = L.block_sum(E8_gram)
                        v_form = vector( list(a) + list(v) ) * LE8_mat
                        w_form = vector( list(a) + list(w) ) * LE8_mat
                        Lred_basis = matrix(ZZ, [v_form, w_form]).right_kernel().basis_matrix().transpose()
                        Lred_basis = matrix(ZZ, Lred_basis)

                        return Lred_basis.transpose() * LE8_mat * Lred_basis

        while not is_positive_definite(L) :
            L = _split_hyperbolic(L)
        
        return L

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
        n_disc = DiscriminantForm(u.transpose() * self._L * u)
        
        uinv = u.inverse()
        basis_images = [ sum(map(operator.mul, b.lift(), self._dual_basis.columns()))
                         for b in self.smith_form_gens() ]
        basis_images = [ n_disc._dual_basis.solve_right(uinv * b).list()
                         for b in basis_images ]
        
        coercion_hom = self.hom([ sum(map( operator.mul, map(ZZ, b), map(n_disc, FreeModule(ZZ, self._L.nrows()).gens()) ))
                                  for b in basis_images ])
        
        return (n_disc, coercion_hom)

    @cached_method
    def _jacobi_indices_matrix(self) :
        r"""
        A basis change matrix mapping indices `r` of Jacobi forms to elements of ``self``.  
        
        OUTPUT:
        
        - A matrix over `\ZZ`.
        """
        return (self._L * self._dual_basis).inverse()
    
    def _from_jacobi_index(self, r) :
        r"""
        Return an element of ``self`` that corresponds to a given index of a Jacobi Fourier expansion.
        
        TESTS:
        
            sage: from psage.modform.vector_valued.discriminant_group import *
            sage: A = DiscriminantForm(matrix(2, [2, 1, 1, 2]))
            sage: A._from_jacobi_index((0,0))
            (0, 0)
            sage: A._from_jacobi_index((0,1))
            (0, 1)
        """
        return self(self._jacobi_indices_matrix() * vector(ZZ, r))

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
        n_disc = DiscriminantForm(nL)
        
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
        n_disc = DiscriminantForm(K)
        
        total_basis_matrix = L_basis_matrix.change_ring(QQ).augment(K_basis_matrix * n_disc._dual_basis)

        basis_images = [ sum(map(operator.mul, b.lift(), self._dual_basis.columns()))
                         for b in self.smith_form_gens() ]
        basis_images = [ total_basis_matrix.solve_right(b)[-K_basis_matrix.ncols():]
                         for b in basis_images ]
        
        coercion_hom = self.hom([ sum(map(operator.mul, map(ZZ, b), map(n_disc, FreeModule(ZZ, K.nrows()).gens()) ))
                                  for b in basis_images ])
        
        return (n_disc, coercion_hom)

    def weil_representation(self) :
        r"""
        OUTPUT:
        
        - A pair of matrices corresponding to T and S.
        """
        disc_bilinear = lambda a, b: (self._dual_basis * vector(QQ, a.lift())) * self._L * (self._dual_basis * vector(QQ, b.lift()))
        disc_quadratic = lambda a: disc_bilinear(a, a) / ZZ(2)
        
        zeta_order = ZZ(lcm([8, 12, prod(self.invariants())] + map(lambda ed: 2 * ed, self.invariants())))
        K = CyclotomicField(zeta_order); zeta = K.gen()

        R = PolynomialRing(K, 'x'); x = R.gen()
#        sqrt2s = (x**2 - 2).factor()
#        if sqrt2s[0][0][0].complex_embedding().real() > 0 :        
#            sqrt2  = sqrt2s[0][0][0]
#        else : 
#            sqrt2  = sqrt2s[0][1]
        Ldet_rts = (x**2 - prod(self.invariants())).factor()
        if Ldet_rts[0][0][0].complex_embedding().real() > 0 :
            Ldet_rt  = Ldet_rts[0][0][0] 
        else :
            Ldet_rt  = Ldet_rts[0][0][0]
                
        Tmat  = diagonal_matrix( K, [zeta**(zeta_order*disc_quadratic(a)) for a in self] )
        Smat = zeta**(zeta_order / 8 * self._L.nrows()) / Ldet_rt  \
               * matrix( K,  [ [ zeta**ZZ(-zeta_order * disc_bilinear(gamma,delta))
                                 for delta in self ]
                               for gamma in self ])
        
        return (Tmat, Smat)

    def weil_representation_kernel(self) :
        r"""
        Compute a scalar `r` and a group `\Gamma` which contains the translation `T`
        so that `\diag(1, 1/r) \Gamma \diag(1, r) \subseteq \ker( \rho_D )`.

        OUTPUT:
        
        - A pair of an integer and an arithmetic group.
        """
        if self._L == matrix(2, [2, 1, 1, 2]) or self._L == -matrix(2, [2, 1, 1, 2]) :
            return (ZZ(3),  GammaH(9, [4]))

        level = 2 * self._L.inverse().denominator()
        return ( level, GammaH( level**2, filter(lambda n: n % level == 1, range(level**2)) ))

        ## We compute the kernel of the Weil representation of L0 = (2, 1; 1, 2)
        ## The step that we need to perform are commented out. The result, GG,
        ## is defined below.

        # (Tmat, Smat) = disc_mL0.weil_representation()
        # Rmat = Smat * Tmat.inverse() * Smat.inverse()
        # compute_gmat = lambda g: prod( ((Tmat if lr == 0 else Rmat)**mult for (lr, mult) in sl2z_word_problem(g)), identity_matrix(3) )

        # G = FareySymbol(Gamma(12))

        # neg_mat = compute_gmat(-identity_matrix(2))
        # pre_trivial_cosets_inv = [(matrix(2, list(g)), compute_gmat(g)) for g in G.coset_reps()]
        # pre_trivial_cosets_inv = pre_trivial_cosets_inv + [(-g, neg_mat * m) for (g,m) in pre_trivial_cosets_inv]
        # trivial_cosets = [g.inverse() for (g,m) in pre_trivial_cosets_inv if m == identity_matrix(3)]
         # ## This equals
        # # assert trivial_cosets == [g for (g,m) in pre_trivial_cosets if m.rows()[1] == vector((0,1,0))]

        # rescaling_factor = ZZ(3)
        # gen_factor = diagonal_matrix([rescaling_factor, 1])
        # gen_factor_inv = gen_factor.inverse()
        # additional_gens = [matrix(ZZ, gen_factor_inv * g * gen_factor) for g in trivial_cosets]

        ## Kernel of the Weil representation is
        # GG = GammaH(9, [4])
        # assert GG.index() == Gamma(12).index() / len(additional_gens)
        # assert all(g in GG for g in additional_gens)


        ## This is an additional check for concistency
        # Gcosets_inv = map(lambda m: matrix(2, list(m)), G.coset_reps())
        # Gcosets_inv = Gcosets_inv + map(operator.neg, Gcosets_inv)
        # Gcosets = [g.inverse() for g in Gcosets_inv]
        # get_coset = lambda m: filter(lambda ginv: m * ginv in Gamma(12), Gcosets_inv)[0].inverse()

        # ll = [get_coset(g * h) for (i,g) in enumerate(trivial_cosets) for h in trivial_cosets[i:]]
        # assert all(g in trivial_cosets for g in ll)

    def _an_element_(self) :
        return DiscrimiantFormElement(len(self.invariants())*[0])


    def _repr_(self) :
        return "Discrimiant group of lattice of size {0} ( isomorphic to {1} )".format(self._L.nrows(), self.short_name())
