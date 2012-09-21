r"""
A dimension formula for vector-valued modular forms, and functions that
apply it to the case of Jacobi forms
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

from sage.functions.all import exp, sqrt, sign
from sage.matrix.all import diagonal_matrix, identity_matrix, matrix
from sage.misc.all import sum, mrange
from sage.modules.all import vector 
from sage.rings.all import ComplexField, ZZ, QQ, lcm
from sage.quadratic_forms.all import QuadraticForm 
from sage.symbolic.all import I, pi
from copy import copy
import mpmath
import operator

#===============================================================================
# dimension__jacobi
#===============================================================================

def dimension__jacobi(k, L) :
    r"""
    INPUT:
    
    - `k` -- An integer.
    
    - `L` -- A quadratic form or an even symmetric matrix (over `\ZZ`).
    """
    try :
        Lmat = L.matrix()
    except AttributeError :
        Lmat = L
        L = QuadraticForm(Lmat)

    return dimension__vector_valued(k - ZZ(Lmat.ncols()) / 2, L)

#===============================================================================
# dimension__vector_valued
#===============================================================================

def dimension__vector_valued(k, L) :
    r"""
    Compute the dimension of the space of weight `k` vector valued modular forms
    for the Weil representation attached to the lattice `-L`.
    
    See [Borcherds, Borcherds - Reflection groups of Lorentzian lattices] for a proof
    of the formula that we use here.
    
    INPUT:
    
    - `k` -- A half-integer.
    
    - ``L`` -- An quadratic form.
    
    OUTPUT:
        An integer.

    TESTS::

        sage: TODO: Make precisions rigid
    """
    if 2 * k not in ZZ :
        raise ValueError( "Weight must be half-integral" ) 
    if k <= 0 :
        return 0
    if k < 2 :
        raise NotImplementedError( "Weight <2 is not implemented." )

    if L.matrix().rank() != L.matrix().nrows() :
        raise ValueError( "The lattice (={0}) must be non-degenerate.".format(L) )

    L_dimension = L.matrix().nrows()
    if L_dimension % 2 != ZZ(2 * k) % 2 :
        return 0
    
    plus_basis = ZZ(L_dimension + 2 * k) % 4 == 0 

    ## The bilinear and the quadratic form attached to L
    quadratic = lambda x: L(x) // 2
    bilinear = lambda x,y: L(x + y) - L(x) - L(y)

    ## A dual basis for L
    (elementary_divisors, dual_basis_pre, _) = L.matrix().smith_form()
    elementary_divisors = elementary_divisors.diagonal()
    dual_basis = map(operator.div, list(dual_basis_pre), elementary_divisors)
    
    L_level = ZZ(lcm([ b.denominator() for b in dual_basis ]))
    
    (elementary_divisors, _, discriminant_basis_pre) = (L_level * matrix(dual_basis)).change_ring(ZZ).smith_form()
    elementary_divisors = filter( lambda d: d not in ZZ, (elementary_divisors / L_level).diagonal() )
    elementary_divisors_inv = map(ZZ, [ed**-1 for ed in elementary_divisors])
    discriminant_basis = matrix(map( operator.mul,
                                     discriminant_basis_pre.inverse().rows()[:len(elementary_divisors)],
                                     elementary_divisors )).transpose()
    ## This is a form over QQ, so that we cannot use an instance of QuadraticForm
    discriminant_form = discriminant_basis.transpose() * L.matrix() * discriminant_basis

    disc_quadratic = lambda x : -x * discriminant_form * x / 2
    disc_bilinear = lambda x,y : disc_quadratic(x + y) - disc_quadratic(x) - disc_quadratic(y)

    ## red gives a normal form for elements in the discriminant group
    red = lambda x : vector(map(operator.mod, x, elementary_divisors_inv))
    ## singls and pairs are elements of the discriminant group that are, respectively,
    ## fixed and not fixed by negation. 
    singls = filter(lambda x: red(-x) - x == 0, mrange(elementary_divisors_inv, vector))
    pairs = filter(lambda x: red(-x) - x != 0 and x < red(-x), mrange(elementary_divisors_inv, vector))

    if plus_basis :
        subspace_dimension = len(singls + pairs)
    else :
        subspace_dimension = len(pairs)

    ## 20 bits are, by far, sufficient to distinguish 12-th roots of unity
    ## by increasing the precision by 4 for each additional dimension, we
    ## compensate, by far, the errors introduced by the QR decomposition,
    ## which are of the size of (absolute error) * dimension
    CC = ComplexField(2000 + subspace_dimension * 4)

    zeta_order = ZZ(lcm([8, 12] + map(lambda ed: 2 * ed, elementary_divisors_inv)))

    zeta = CC(exp(2 * pi * I / zeta_order))
    sqrt2  = CC(sqrt(2))
    drt  = CC(sqrt(L.det()))

    Tmat  = diagonal_matrix(CC, [zeta**(-zeta_order*disc_quadratic(a)) for a in (singls + pairs if plus_basis else pairs)])
    if plus_basis :        
        Smat = zeta**(zeta_order / 8 * L_dimension) / drt  \
               * matrix( CC, [  [zeta**(-zeta_order * disc_bilinear(gamma,delta)) for delta in singls]
                              + [sqrt2 * zeta**(-zeta_order * disc_bilinear(gamma,delta)) for delta in pairs]
                              for gamma in singls] \
                           + [  [sqrt2 * zeta**(-zeta_order * disc_bilinear(gamma,delta)) for delta in singls]
                              + [zeta**(-zeta_order * disc_bilinear(gamma,delta)) + zeta**(-zeta_order * disc_bilinear(gamma,-delta)) for delta in pairs]
                              for gamma in pairs] )
    else :
        Smat = zeta**(zeta_order / 8 * L_dimension) / drt  \
               * matrix( CC, [  [zeta**(-zeta_order * disc_bilinear(gamma,delta)) - zeta**(-zeta_order * disc_bilinear(gamma,-delta))  for delta in pairs]
                               for gamma in pairs ] )
    STmat = Smat * Tmat
    
    def eigenvalue_multiplicity(mat, ev) :
        #TODO: Adjust
        eps = 10**-50
        mat = matrix(CC, mat - ev * identity_matrix(subspace_dimension))
        return len(filter( lambda row: all( abs(e) < eps for e in row), _qr(mat, eps) ))
    
    i = CC(exp(2 * pi * I / 8))
    S_ev_multiplicity = [eigenvalue_multiplicity(Smat, i**n) for n in range(8)]
    assert sum(S_ev_multiplicity) == subspace_dimension

    rho = CC(exp(2 * pi * I / 12))
    ST_ev_multiplicity = [eigenvalue_multiplicity(STmat, rho**n) for n in range(12)]
    assert sum(ST_ev_multiplicity) == subspace_dimension

    T_evs = [ ZZ((-zeta_order * disc_quadratic(a)) % zeta_order) / zeta_order
              for a in (singls + pairs if plus_basis else pairs) ]

    return subspace_dimension * (1 + QQ(k) / 12) \
           - ZZ(sum( (ST_ev_multiplicity[n] * ((-2 * k - n) % 12)) for n in range(12) )) / 12 \
           - ZZ(sum( (S_ev_multiplicity[n] * ((2 * k + n) % 8)) for n in range(8) )) / 8 \
           - sum(T_evs)

def _qr(mat, eps = 10**-10) :
    r"""
    Compute the R matrix in QR decomposition using Housholder reflections.
    
    This is an adoption of the implementation in mpmath by Andreas Strombergson. 
    """
    CC = mat.base_ring()
    mat = copy(mat)
    m = mat.nrows()
    n = mat.ncols()

    p = []
    cur_row = 0
    for j in range(0, n) :
        s = sum( (abs(mat[i,j]))**2 for i in xrange(cur_row, m) )
        if abs(s) <= eps :
            for i in range(cur_row, m) :
                mat[i,j] = CC(0)
            continue
            
        p = sqrt(s)
        kappa = 1 / (s - p * mat[cur_row,j])

        mat[cur_row,j] -= p
        for k in range(j + 1, n) :
            y = sum(mat[i,j].conjugate() * mat[i,k] for i in xrange(cur_row, m)) * kappa
            for i in range(cur_row, m):
                mat[i,k] -= mat[i,j] * y

        mat[cur_row,j] = p
        for i in range(cur_row + 1, m) :
            mat[i,j] = CC(0)
        
        cur_row += 1
    
    return mat