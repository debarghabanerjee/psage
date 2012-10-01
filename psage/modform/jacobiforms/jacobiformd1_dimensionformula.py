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
from sage.rings.all import ComplexIntervalField, ZZ, QQ, lcm
from sage.rings.all import moebius, gcd, QuadraticField, fundamental_discriminant, kronecker_symbol
from sage.quadratic_forms.all import QuadraticForm, BinaryQF_reduced_representatives 
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

    return dimension__vector_valued(k - ZZ(Lmat.ncols()) / 2, L, conjugate = True)

def dimension__jacobi_scalar(k, m) :
    raise RuntimeError( "There is a bug in the implementation" )
    m = ZZ(m)
    
    dimension = 0
    for d in (m // m.squarefree_part()).isqrt().divisors() :
        m_d = m // d**2
        dimension += sum ( dimension__jacobi_scalar_f(k, m_d, f)
                            for f in m_d.divisors() )
    
    return dimension
            
def dimension__jacobi_scalar_f(k, m, f) :
    if moebius(f) != (-1)**k :
        return 0
    
    ## We use chapter 6 of Skoruppa's thesis
    ts = filter(lambda t: gcd(t, m // t) == 1, m.divisors())
    
    ## Eisenstein part
    eis_dimension = 0
    
    for t in ts :
        eis_dimension +=   moebius(gcd(m // t, f)) \
                        * (t // t.squarefree_part()).isqrt() \
                        * (2 if (m // t) % 4 == 0 else 1) 
    eis_dimension = eis_dimension // len(ts)
    
    if k == 2 and f == 1 :
        eis_dimension -= len( (m // m.squarefree_part()).isqrt().divisors() ) 
    
    ## Cuspidal part
    cusp_dimension = 0
    
    tmp = ZZ(0)
    for t in ts :
        tmp += moebius(gcd(m // t, f)) * t
    tmp = tmp / len(ts)
    cusp_dimension += tmp * (2 * k - 3) / ZZ(12)
    print "1: ", cusp_dimension
    
    if m % 2 == 0 :
        tmp = ZZ(0)
        for t in ts :
            tmp += moebius(gcd(m // t, f)) * kronecker_symbol(-4, t)
        tmp = tmp / len(ts)
        
        cusp_dimension += 1/ZZ(2) * kronecker_symbol(8, 2 * k - 1) * tmp
        print "2: ", 1/ZZ(2) * kronecker_symbol(8, 2 * k - 1) * tmp
        
    tmp = ZZ(0)
    for t in ts :
        tmp += moebius(gcd(m // t, f)) * kronecker_symbol(t, 3)
    tmp = tmp / len(ts)
    if m % 3 != 0 :
        cusp_dimension += 1 / ZZ(3) * kronecker_symbol(k, 3) * tmp
        print ": ", 1 / ZZ(3) * kronecker_symbol(k, 3) * tmp
    elif k % 3 == 0 :
        cusp_dimension += 2 / ZZ(3) * (-1)**k * tmp
        print "3: ", 2 / ZZ(3) * (-1)**k * tmp
    else :
        cusp_dimension += 1 / ZZ(3) * (kronecker_symbol(k, 3) + (-1)**(k - 1)) * tmp
        print "3: ", 1 / ZZ(3) * (kronecker_symbol(k, 3) + (-1)**(k - 1)) * tmp
    
    tmp = ZZ(0)
    for t in ts :
        tmp +=   moebius(gcd(m // t, f)) \
               * (t // t.squarefree_part()).isqrt() \
               * (2 if (m // t) % 4 == 0 else 1)
    tmp = tmp / len(ts)
    cusp_dimension -= 1 / ZZ(2) * tmp
    print "4: ", -1 / ZZ(2) * tmp
    
    tmp = ZZ(0)
    for t in ts :
        tmp +=   moebius(gcd(m // t, f)) \
               * sum(   (( len(BinaryQF_reduced_representatives(-d, True))
                           if d not in [3, 4] else ( 1 / ZZ(3) if d == 3 else 1 / ZZ(2) ))
                         if d % 4 == 0 or d % 4 == 3 else 0 )
                      * kronecker_symbol(-d, m // t)
                      * ( 1 if (m // t) % 2 != 0 else
                          ( 4 if (m // t) % 4 == 0 else 2 * kronecker_symbol(-d, 2) ))
                      for d in (4 * m).divisors() )
    tmp = tmp / len(ts)
    cusp_dimension -= 1 / ZZ(2) * tmp
    print "5: ", -1 / ZZ(2) * tmp
    
    if k == 2 :
        cusp_dimension += len( (m // f // (m // f).squarefree_part()).isqrt().divisors() )
    
    return eis_dimension + cusp_dimension
#===============================================================================
# dimension__vector_valued
#===============================================================================

def dimension__vector_valued(k, L, conjugate = False) :
    r"""
    Compute the dimension of the space of weight `k` vector valued modular forms
    for the Weil representation (or its conjugate) attached to the lattice `L`.
    
    See [Borcherds, Borcherds - Reflection groups of Lorentzian lattices] for a proof
    of the formula that we use here.
    
    INPUT:
    
    - `k` -- A half-integer.
    
    - ``L`` -- An quadratic form.
    
    - ``conjugate`` -- A boolean; If ``True``, then compute the dimension for
                       the conjugated Weil representation.
    
    OUTPUT:
        An integer.

    TESTS::

        sage: ??
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

    if conjugate :
        disc_quadratic = lambda x : -x * discriminant_form * x / 2
    else :
        disc_quadratic = lambda x : x * discriminant_form * x / 2
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

    ## 200 bits are, by far, sufficient to distinguish 12-th roots of unity
    ## by increasing the precision by 4 for each additional dimension, we
    ## compensate, by far, the errors introduced by the QR decomposition,
    ## which are of the size of (absolute error) * dimension
    CC = ComplexIntervalField(200 + subspace_dimension * 4)

    zeta_order = ZZ(lcm([8, 12] + map(lambda ed: 2 * ed, elementary_divisors_inv)))

    zeta = CC(exp(2 * pi * I / zeta_order))
    sqrt2  = CC(sqrt(2))
    drt  = CC(sqrt(L.det()))

    Tmat  = diagonal_matrix(CC, [zeta**(zeta_order*disc_quadratic(a)) for a in (singls + pairs if plus_basis else pairs)])
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
    
    ## This function overestimates the number of eigenvalues, if it is not correct
    def eigenvalue_multiplicity(mat, ev) :
        mat = matrix(CC, mat - ev * identity_matrix(subspace_dimension))
        return len(filter( lambda row: all( e.contains_zero() for e in row), _qr(mat).rows() ))
    
    rti = CC(exp(2 * pi * I / 8))
    S_ev_multiplicity = [eigenvalue_multiplicity(Smat, rti**n) for n in range(8)]
    ## Together with the fact that eigenvalue_multiplicity overestimates the multiplicities
    ## this asserts that the computed multiplicities are correct
    assert sum(S_ev_multiplicity) == subspace_dimension

    rho = CC(exp(2 * pi * I / 12))
    ST_ev_multiplicity = [eigenvalue_multiplicity(STmat, rho**n) for n in range(12)]
    ## Together with the fact that eigenvalue_multiplicity overestimates the multiplicities
    ## this asserts that the computed multiplicities are correct
    assert sum(ST_ev_multiplicity) == subspace_dimension

    T_evs = [ ZZ((zeta_order * disc_quadratic(a)) % zeta_order) / zeta_order
              for a in (singls + pairs if plus_basis else pairs) ]

    return subspace_dimension * (1 + QQ(k) / 12) \
           - ZZ(sum( (ST_ev_multiplicity[n] * ((-2 * k - n) % 12)) for n in range(12) )) / 12 \
           - ZZ(sum( (S_ev_multiplicity[n] * ((2 * k + n) % 8)) for n in range(8) )) / 8 \
           - sum(T_evs)

def _qr(mat) :
    r"""
    Compute the R matrix in QR decomposition using Housholder reflections.
    
    This is an adoption of the implementation in mpmath by Andreas Strombergson. 
    """
    CC = mat.base_ring()
    mat = copy(mat)
    m = mat.nrows()
    n = mat.ncols()

    cur_row = 0
    for j in range(0, n) :
        if all( mat[i,j].contains_zero() for i in xrange(cur_row + 1, m) ) :
            if not mat[cur_row,j].contains_zero() :
                cur_row += 1
            continue

        s = sum( (abs(mat[i,j]))**2 for i in xrange(cur_row, m) )
        if s.contains_zero() :
            raise RuntimeError( "Cannot handle imprecise sums of elements that are too precise" )
        
        p = sqrt(s)
        if (s - p * mat[cur_row,j]).contains_zero() :
            raise RuntimeError( "Cannot handle imprecise sums of elements that are too precise" )
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