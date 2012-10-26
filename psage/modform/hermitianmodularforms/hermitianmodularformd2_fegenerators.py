"""
Functions for treating the fourier expansion of hermitian modular forms.

AUTHOR :
    -- Martin Raum (2009 - 11 - 07) Initial version
    -- Dominic Gehre (2009 - 02) Implement generators for vector valued
                modular forms if D = -3
    
REFERENCE :
    -- [D] Tobias Dern, Hermitsche Modulformen zweiten Grades, PhD thesis
           RWTH Aachen University, Germany, 2001

NOTE:
    The Fourier expansion of vector-valued modular forms that we use
    a preimages maps to hermitian modular forms has the following form:
    It is a dictionary of dictionaries. The keys of the former are tuples
    `(b1, b2)` that are reduced according to :meth:`HermitianModularFormD2Factory._reduce_vector_valued_index`.
    The keys of the latter are discriminants of elements in the dual of `\mathcal{o}_{\QQ(D)}`
    that are in the same coset as `(b1,b2)`.
"""

#===============================================================================
# 
# Copyright (C) 2009 Dominic Gehre, Martin Raum
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

from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import \
             HermitianModularFormD2Filter_diagonal, HermitianModularFormD2FourierExpansionRing
from sage.misc.cachefunc import cached_method
from sage.modular.congroup import Gamma1
from sage.modular.modform.constructor import ModularForms
from sage.modular.modform.element import ModularFormElement
from sage.rings.arith import fundamental_discriminant, divisors, bernoulli, sigma, gcd, crt, lcm
from sage.rings.integer import Integer
from sage.rings.integer_ring import ZZ
from sage.rings.power_series_ring import PowerSeriesRing
from sage.rings.rational_field import QQ
from sage.structure.sage_object import SageObject
import operator

#===============================================================================
# HermitianModularFormD2AdditiveLift
#===============================================================================

def HermitianModularFormD2AdditiveLift( precision, forms, discriminant = None, weight = None,
                                        is_symmetric = True, with_determinant_character = False,
                                        with_nu_character = False, is_integral = False ) :
    """
    Borcherd's additive lift for hermitian modular forms of vector valued elliptic
    modular forms, a hermitian modular form of degree `2`. The vector valued form
    is given by a list of scalar valued forms, that are the coefficients with
    respect to a basis of the first module of modular forms over the ring of the
    later.
    
    INPUT:
        - ``precision``      -- An integer or an  instance of a precision class.
        - ``forms``          -- A list of elliptic modular forms or their q-expansion.
                                See above for a more detailed explaination.
        - ``discriminant``   -- A negative integer or ``None`` (default: ``None``);
                                The fundamental discriminant of an imaginary quadratic field.
                                Can be omited if the precision is not an integer.
        - ``weight``         -- A positive integer or ``None`` (default: ``None``);
                                The weight of the image. Can be omitted if at least
                                one element of ``forms`` is a modular form.
        - ``is_symmetric``   -- A boolean (default: ``True``); Whether the resulting Fourier
                                expansion is symmetric with respect to the transposition or not.
        - ``with_determinant_character`` -- A boolean (defaul: ``True``); Whether in addition to
                                            the contribution of weight, there is a character
                                            `\mathrm{det}^{|\mathfrak{o}^\times|/2\}.
        - ``with_nu_character`` -- A boolean (default: ``False``); Determines whether
                                   the image will have trivial character `\nu` or not.
        - ``is_integral``    -- A boolean (default: ``False``); If ``True`` data
                                types optimized for integral Fourier coefficients
                                will be used.

    OUTPUT:
        An element of the ring of all Fourier expansions of hermitian modular forms.

    EXAMPLES::
        sage: from psage.modform.hermitianmodularforms import HermitianModularFormD2AdditiveLift
        sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal
        sage: HermitianModularFormD2AdditiveLift(2, [ModularForms(1, 4).0, 0, 0], -3)
        Equivariant monoid power series in Ring of equivariant monoid power series over Non-reduced quadratic forms over o_QQ(\sqrt -3)
    
    TESTS::
        sage: HermitianModularFormD2AdditiveLift(2, [ModularForms(1, 4).0, 0, 0], -3).non_zero_components()
        [1]
        sage: lift = HermitianModularFormD2AdditiveLift(2, [0, 0, 1], -3, 9, is_integral = True )
        sage: lift.parent().coefficient_domain()
        Integer Ring
        sage: lift.precision()
        Reduced diagonal filter for discriminant -3 with bound 2
        sage: HermitianModularFormD2AdditiveLift(HermitianModularFormD2Filter_diagonal(4, -3), [0, 0, 0], weight = 5)
        Equivariant monoid power series in Ring of equivariant monoid power series over Non-reduced quadratic forms over o_QQ(\sqrt -3)
        sage: HermitianModularFormD2AdditiveLift(HermitianModularFormD2Filter_diagonal(4, -3), [0, 0, 0], -4, weight = 5)
        Traceback (most recent call last):
        ...
        ValueError: Discriminant must coinside with the precision's discriminant.
        sage: HermitianModularFormD2AdditiveLift(2, [0, 0, 0], -3)
        Traceback (most recent call last):
        ...
        ValueError: If no modular form is passed, the weight has to be set.
        sage: HermitianModularFormD2AdditiveLift(2, [ModularForms(1, 4).0, 0, 0], -3, 4)
        Traceback (most recent call last):
        ...
        ValueError: Incorrect weight of form 0.
        sage: HermitianModularFormD2AdditiveLift(2, [ModularForms(1, 4).0, ModularForms(1, 4).0, 0], -3 )
        Traceback (most recent call last):
        ValueError: Incorrect weight of form 1.
        sage: HermitianModularFormD2AdditiveLift(2, [ModularForms(1, 4).0], -3, 4)
        Traceback (most recent call last):
        ...
        ValueError: Exactly 3 forms have to be passed.
        sage: HermitianModularFormD2AdditiveLift(2, [ModularForms(1, 4).0, 0, 0], -3, with_nu_character = True )
        Traceback (most recent call last):
        ...
        ValueError: Character nu is only admissible if D = -4.
        sage: HermitianModularFormD2AdditiveLift(2, [ModularForms(1, 4).0, 0, 0], -4, with_nu_character = True )
        Traceback (most recent call last):
        ...
        NotImplementedError: Only discriminant -3 is implemented.
    """
    if with_nu_character and discriminant != -4 :
        raise ValueError( "Character nu is only admissible if D = -4.")    
#    if with_nu_character :
#        raise NotImplementedError( "Characters for additive lifts are not implemented.")

    fac = HermitianModularFormD2Factory(precision, discriminant)
    vvb_weights = fac._additive_lift_vector_valued_basis_weights(with_nu_character)
    
    if len(vvb_weights) != len(forms) :
        raise ValueError( "Exactly %s forms have to be passed." % (len(vvb_weights),) )
    
    if weight is None :
        for (k,f) in zip(vvb_weights, forms) :
            try :
                weight = k + f.weight()
            except AttributeError :
                continue
            break
    if weight is None :
        raise ValueError( "If no modular form is passed, the weight has to be set." )
    
    for (i, (k,f)) in enumerate(zip(vvb_weights, forms)) :
        try :
            if weight != k + f.weight() :
                raise ValueError( "Incorrect weight of form %s." % (i,))
        except AttributeError :
            continue
    
    def const_funct(i) :
        return lambda p : i
    
    forms = [ f.qexp
              if isinstance(f, ModularFormElement)
              else ( const_funct(f)
                     if f in QQ
                     else f )
              for f in forms ]
    
    fering = HermitianModularFormD2FourierExpansionRing(ZZ if is_integral else QQ, fac.precision().discriminant())

    if discriminant == -4 :
        character = (0 if is_symmetric else 1, 1 if with_determinant_character else 0, 1 if with_nu_character else 0)
    else :
        character = (0 if is_symmetric else 1, 1 if with_determinant_character else 0)
    character = fering.characters()(list(character))
    
    res = fering({character : fac.additive_lift( forms, weight, with_character = with_nu_character, is_integral = is_integral ) })
    res._set_precision(precision)
    
    return res

#===============================================================================
# HermitianModularFormD2Factory
#===============================================================================

_hermitianmodularformd2factory_cache = dict()

def HermitianModularFormD2Factory( precision, discriminant = None ) :
    """
    Create an instance of a factory for Fourier expansions of Hermitian
    modular forms.
    
    INPUT:
        - ``precision``      -- An integer or an  instance of a precision class.
        - ``discriminant``   -- A negative integer or ``None`` (default: ``None``);
                                The fundamental discriminant of an imaginary quadratic field.
                                Can be omited if the precision is not an integer.
    
    OUTPUT:
        An instance of :class:~`.HermitianModularFormD2Factory_class`.
    
    SEE:
        :class:~`.HermitianModularFormD2Factory_class`.
    
    TESTS::
        sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fegenerators import HermitianModularFormD2Factory
        sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal 
        sage: h = HermitianModularFormD2Factory(3, -3)
        sage: h.precision()
        Reduced diagonal filter for discriminant -3 with bound 3
        sage: HermitianModularFormD2Factory(HermitianModularFormD2Filter_diagonal(2, -3))
        Factory for Fourier expansions of hermitian modular forms with precision Reduced diagonal filter for discriminant -3 with bound 2
        sage: HermitianModularFormD2Factory(HermitianModularFormD2Filter_diagonal(2, -3), -4)
        Traceback (most recent call last):
        ...
        ValueError: Discriminant must coinside with the precision's discriminant.
        sage: HermitianModularFormD2Factory(2, -1)
        Traceback (most recent call last):
        ...
        ValueError: Discriminant must be a fundamental discriminant.
        sage: HermitianModularFormD2Factory(2, -4)
        Traceback (most recent call last):
        ...
        NotImplementedError: Only discriminant -3 is implemented.
    """
    if not isinstance(precision, HermitianModularFormD2Filter_diagonal) :
        if discriminant is None :
            raise ValueError( "If precision is not a precision class, discriminant must be set." )
        
        precision = HermitianModularFormD2Filter_diagonal(precision, discriminant)
    else :
        if not discriminant is None and precision.discriminant() != discriminant :
            raise ValueError( "Discriminant must coinside with the precision's discriminant." )
        else :
            discriminant = precision.discriminant() 
    
    if discriminant >= 0 :
        raise ValueError( "Discriminant must be negative." )
    if fundamental_discriminant(discriminant) != discriminant :
        raise ValueError( "Discriminant must be a fundamental discriminant." )
        
    if discriminant != -3 :
        raise NotImplementedError( "Only discriminant -3 is implemented." )
        
    key = (precision)
    global _hermitianmodularformd2factory_cache
    try :
        return _hermitianmodularformd2factory_cache[key]
    except :
        factory = HermitianModularFormD2Factory_class(precision)
        _hermitianmodularformd2factory_cache[key] = factory
        
        return factory

#===============================================================================
# HermitianModularFormD2Factory_class
#===============================================================================

class HermitianModularFormD2Factory_class ( SageObject ) :
    """
    A factory for Fourier expansions of hermitian modular forms of
    degree `2`.
    """
    
    def __init__(self, precision) :
        """
        INPUT:
            - ``precision``  -- A precision class for hermitian modular forms.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fegenerators import HermitianModularFormD2Factory_class
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal 
            sage: h = HermitianModularFormD2Factory_class( HermitianModularFormD2Filter_diagonal(4, -3) )
        """
        self.__precision = precision
        self.__D = precision.discriminant()

        ## Conversion of power series is not expensive but powers of interger
        ## series are much cheaper then powers of rational series
        self._power_series_ring_ZZ = PowerSeriesRing(ZZ, 'q')
        self._power_series_ring = PowerSeriesRing(QQ, 'q')

    def precision(self) :
        """
        The for all produced Fourier expansions.
        
        OUTPUT:
            An instance of a precision class.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fegenerators import HermitianModularFormD2Factory_class
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal 
            sage: h = HermitianModularFormD2Factory_class( HermitianModularFormD2Filter_diagonal(4, -3) )
            sage: h.precision() == HermitianModularFormD2Filter_diagonal(4, -3)
            True
        """
        return self.__precision
        
    def power_series_ring(self) :
        """
        The cached instance of the power series ring for q-expansions.
        
        OUTPUT:
            A power series ring.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fegenerators import HermitianModularFormD2Factory_class
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal 
            sage: HermitianModularFormD2Factory_class( HermitianModularFormD2Filter_diagonal(4, -3) ).power_series_ring()
            Power Series Ring in q over Rational Field
        """
        return self._power_series_ring
    
    def integral_power_series_ring(self) :
        """
        The cached instance of the power series ring for q-expansions over `\Z`.
        
        OUTPUT:
            A power series ring.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fegenerators import HermitianModularFormD2Factory_class
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal 
            sage: HermitianModularFormD2Factory_class( HermitianModularFormD2Filter_diagonal(4, -3) ).integral_power_series_ring()
            Power Series Ring in q over Integer Ring
        """
        return self._power_series_ring_ZZ
    
    def _pari(self) :
        """
        Return a cached instance of the Pari-GP interface.
        
        OUTPUT:
            A Pari-GP interface.
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fegenerators import HermitianModularFormD2Factory_class
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal 
            sage: h = HermitianModularFormD2Factory_class( HermitianModularFormD2Filter_diagonal(4, -3) )
            sage: h._pari()
            Interface to the PARI C library
            sage: h._pari() is h._pari()
            True
        """
        try :
            return self.__gp_instance
        except AttributeError :
            import sage.libs.pari.gen
            
            self.__gp_instance = sage.libs.pari.gen.PariInstance()
            
            return self.__gp_instance

    ###########################################################################
    ### Additive lift
    ###########################################################################
    
    def _qexp_precision(self) :
        """
        The precision for q-expansions reaching over all possible discriminants below
        ``self.precision()``.
        
        OUTPUT:
            An integer.

        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fegenerators import HermitianModularFormD2Factory_class
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal 
            sage: HermitianModularFormD2Factory_class( HermitianModularFormD2Filter_diagonal(4, -3))._qexp_precision()
            28
        """
        return self.__precision._enveloping_discriminant_bound()
    
    def _reduce_vector_valued_index(self, t) :
        """
        Return a canonical reduction of `t \in (\alpha + \mathfrak{o}^#) / (a + \mathfrak{o})`.

        INPUT:
            - `t` -- A pair of integers; Representing an element of `\mathfrak{o}^`
                     with respect to the integral basis `(1/\sqrt{D}, (1 + \sqrt{D}) / 2)` 

        NOTE:
            The tuple `(0,0)` is guaranteed to be reduced.

        SEE:
            [p.82, D]
        
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fegenerators import HermitianModularFormD2Factory_class
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal 
            sage: h = HermitianModularFormD2Factory_class( HermitianModularFormD2Filter_diagonal(4, -3), )
            sage: h._reduce_vector_valued_index((0,0))
            (0, 0)
            sage: h._reduce_vector_valued_index((7,4))
            (1, 0)
            sage: h._reduce_vector_valued_index((4,2))
            (1, 0)
            sage: h._reduce_vector_valued_index((2,2))
            (-1, 0)
            sage: h = HermitianModularFormD2Factory_class( HermitianModularFormD2Filter_diagonal(4, -4) )
            sage: h._reduce_vector_valued_index((0,0))
            Traceback (most recent call last):
            ...
            NotImplementedError: Only discriminant -3 is implemented.
        """
        if self.__D == -3 :
            h = t[0] % 3
            if h == 2 : h = -1
            
            return (h, 0)
        
        raise NotImplementedError( "Only discriminant -3 is implemented." )
    
    def _D_3_eisensteinseries(self, i) :
        """
        Here we calculate the needed vector valued Eisenstein series of weights 3
        and 5 up to precision self._qexp_precision().
        The components of these Eisensteinseries are elliptic modular forms with
        respect to `\Gamma_1(36)` and `\Gamma(6)`, respectively. More precisely
        they lie in the subspace of Eisenstein series.
        
        INPUT:
            - i -- Integer; The parameter 0 returns the weight 3 case, 1 the weight 5 case. 
        
        OUTPUT:
            - Element of self.integral_power_series_ring().
        
        NOTE:
            With the help of some calculated Fourier coefficients it is hence possible
            to represent the components as linear combinations of elliptic modular
            forms, which is faster than calculating them directly via certain other
            formulas.

        TESTS::
            sage: fac = HermitianModularFormD2Factory(2,-3)
            sage: fac._D_3_eisensteinseries(0)
            {(1, 0): 27*q^4 + 216*q^10 + 459*q^16 + O(q^19), (0, 0): 1 + 72*q^6 + 270*q^12 + 720*q^18 + O(q^24), (-1, 0): 27*q^4 + 216*q^10 + 459*q^16 + O(q^19)}
            sage: fac._D_3_eisensteinseries(1)
            {(1, 0): -45*q^4 - 1872*q^10 - 11565*q^16 + O(q^19), (0, 0): 1 - 240*q^6 - 3690*q^12 - 19680*q^18 + O(q^24), (-1, 0): -45*q^4 - 1872*q^10 - 11565*q^16 + O(q^19)}
            sage: fac._D_3_eisensteinseries(2)
            Traceback (most recent call last):
            ...
            NotImplementedError: Only weight 3 and 5 implemented. Parameter i should be 0 or 1.
        """
        R = self.integral_power_series_ring()
        q = R.gen(0)
        
        # weight 3 case
        if i == 0 :
            # the factors for the linear combinations were calculated by comparing
            # sufficiently many Fourier coefficients
            linear_combination_factors = ( (1,72,270,720,936,2160,2214,3600,4590,6552,
                          5184,10800,9360,12240,13500,17712,14760,25920,19710,26064,
                          28080,36000,25920,47520,37638,43272,45900,59040,46800,75600,
                          51840,69264,73710,88560,62208,108000,85176,97740,122400,88128),
                         (0,0,0,0,27,0,0,0,0,0,
                          216,0,0,0,0,0,459,0,0,0,
                          0,0,1080,0,0,0,0,0,1350,0,
                          0,0,0,0,2592,0,0,0,0,2808) )
            echelon_basis_eisenstein_subspace_gamma_6_weight_3 = \
                          ModularForms(Gamma1(36),3).eisenstein_subspace() \
                            .q_echelon_basis(6*(self._qexp_precision() - 1) + 1)
            echelon_basis_eisenstein_subspace_gamma1_36_weight_3 = \
                          ModularForms(Gamma1(36),3).eisenstein_subspace() \
                            .q_echelon_basis(self._qexp_precision())
                            
            e1 = R(sum( map(operator.mul, linear_combination_factors[0],
                                          echelon_basis_eisenstein_subspace_gamma1_36_weight_3) ) )
            e1 = e1.subs({q : q**6})
            e2 = R(sum( map(operator.mul, linear_combination_factors[1],
                                          echelon_basis_eisenstein_subspace_gamma_6_weight_3) ) )

            return { (0,0) : e1, (1,0) : e2, (-1,0) : e2 }
        # weight 5 case
        elif i == 1 :
            # the factors for the linear combinations were calculated by comparing sufficiently
            # many Fourier coefficients
            linear_combination_factors = ( (1,-240,-3690,-19680,-57840,-153504,-295290,-576480,
                          -948330,-1594320,-2246400,-3601440,-4742880,-6854880,-8863380,
                          -12284064,-14803440,-20545920,-23914890,-31277280,-36994464,
                          -47271360,-52704000,-68840640,-75889530,-93600240,-105393780,
                          -129140160,-138931680,-173990880,-184204800,-221645280,
                          -242776170,-288203040,-300672000,-368716608,-384231120,
                          -480888180,-562100160,-577324800),
                         (0,0,0,0,-45,0,0,0,0,0,
                          -1872,0,0,0,0,0,-11565,0,0,0,
                          0,0,-43920,0,0,0,0,0,-108090,0,
                          0,0,0,0,-250560,0,0,0,0,-451152) )
            echelon_basis_eisenstein_subspace_gamma_6_weight_5 = \
                          ModularForms(Gamma1(36),5).eisenstein_subspace() \
                            .q_echelon_basis(6*(self._qexp_precision() - 1)  + 1)
            echelon_basis_eisenstein_subspace_gamma1_36_weight_5 = \
                          ModularForms(Gamma1(36),5).eisenstein_subspace() \
                            .q_echelon_basis(self._qexp_precision())
            
            e1 = R( sum( map(operator.mul, linear_combination_factors[0],
                                           echelon_basis_eisenstein_subspace_gamma1_36_weight_5) ) )
            e1 = e1.subs({q : q**6})
            e2 = R( sum( map(operator.mul, linear_combination_factors[1],
                                           echelon_basis_eisenstein_subspace_gamma_6_weight_5) ) )

            return { (0,0) : e1, (1,0) : e2, (-1,0) : e2 }
        else:
            raise NotImplementedError( "Only weight 3 and 5 implemented. " + \
                                       "Parameter i should be 0 or 1." )

    def _D_3_f8(self) :
        """
        Return the 16-th power of `\eta`, with exponents multiplied by 6.
        
        OUTPUT:
            - Element of self.integral_power_series_ring().

        TESTS::
            sage: fac = HermitianModularFormD2Factory(2,-3)
            sage: fac._D_3_f8()
            {(1, 0): q^4 - 16*q^10 + 104*q^16 - 320*q^22 + O(q^28), (0, 0): O(q^19), (-1, 0): -q^4 + 16*q^10 - 104*q^16 + 320*q^22 + O(q^28)}
        """
        pari = self._pari()
        R = self.integral_power_series_ring()
        q = R.gen(0)
        
        etapw = ( R(pari('eta(q + O(q^%s))' % (self._qexp_precision(),)))
                  .add_bigoh(self._qexp_precision())**16 )
        etapw = etapw.subs({q : q**6}).shift(4)

        return { (0,0) : R(0).add_bigoh(6 * (self._qexp_precision() - 1) + 1),
                 (1,0) : etapw, (-1,0) : -etapw }
    
    @cached_method
    def _additive_lift_vector_valued_basis(self, with_character = False) :
        """
        Return a basis for the '[SL_2(Z),*,1]' module of vector valued modular forms
        associated with the Weil representation for discriminant ``self.__D``.
        The first element of the returned pair is a multiplicity `\mu` for
        the exponents. Namely, instead of `\sum a_n q^n` the returned series
        will be `\sum a_n q^{\mu n}`. The multiplicity is guaranteed to be
        divisible by ``self.__D``.

        INPUT:
            - ``with_character`` -- Boolean (optional; default is ``False``)

        OUTPUT:
            - A pair ``(mult, expansions)``. ``mult`` is an integer.
            ``expansions`` is a list of dictionaries. Keys in this
            dictionary are reduced vv-indices (see ``_reduce_vector_valued_index``).
            The values are elements of ``self.integral_power_series_ring()``.

        TESTS::
            sage: fac = HermitianModularFormD2Factory(2,-3)
            sage: fac._additive_lift_vector_valued_basis(True)
            Traceback (most recent call last):
            ...
            ValueError: Characters are only admissible if the discriminant is even.
            sage: fac._additive_lift_vector_valued_basis()
            (6, [{(1, 0): 27*q^4 + 216*q^10 + 459*q^16 + O(q^19), (0, 0): 1 + 72*q^6 + 270*q^12 + 720*q^18 + O(q^24), (-1, 0): 27*q^4 + 216*q^10 + 459*q^16 + O(q^19)}, {(1, 0): -45*q^4 - 1872*q^10 - 11565*q^16 + O(q^19), (0, 0): 1 - 240*q^6 - 3690*q^12 - 19680*q^18 + O(q^24), (-1, 0): -45*q^4 - 1872*q^10 - 11565*q^16 + O(q^19)}, {(1, 0): q^4 - 16*q^10 + 104*q^16 - 320*q^22 + O(q^28), (0, 0): O(q^19), (-1, 0): -q^4 + 16*q^10 - 104*q^16 + 320*q^22 + O(q^28)}])
        """
        if with_character and self.__D % 4 != 0 :
            raise ValueError( "Characters are only admissible if the discriminant is even.")
            
        if self.__D == -3 :
            return (6, [ self._D_3_eisensteinseries(0), self._D_3_eisensteinseries(1),
                         self._D_3_f8() ])

        raise NotImplemented( "Basis for vector valued modular forms not available for D = %s, %s ."
                              % (self.__D, with_character) )

    def _additive_lift_vector_valued_basis_weights(self, with_character = False) :
        """
        Returns the weights of the basis of vector valued modular forms
        used for the additive lift

        INPUT:
            - ``with_character`` -- Boolean (optional; default is ``False``)

        OUTPUT:
            - Tuple of integers

        TESTS::
            sage: fac = HermitianModularFormD2Factory(2,-3)
            sage: fac._additive_lift_vector_valued_basis_weights(True)
            Traceback (most recent call last):
            ...
            ValueError: Characters are only admissible if the discriminant is even.
            sage: fac._additive_lift_vector_valued_basis_weights()
            (3, 5, 8)
        """
        if with_character and self.__D % 4 != 0 :
            raise ValueError( "Characters are only admissible if the discriminant is even.")
        
        if self.__D == -3 :
            return (3, 5, 8)

        raise NotImplemented( "Basis for vector valued modular forms not available for D = %s, %s ."
                              % (self.__D, with_character) )

    @cached_method
    def _iterator_content(self, with_character = False) :
        """
        Return an iterator over all contents ``eps`` of Fourier indices, which are admissable
        given a ``precision``.
        
        INPUT:
            - ``with_character`` -- A boolean (default: ``False``); Distinguish whether
                                    the Fourier indices of a form with character are
                                    enumerated. 
        """
        
        if with_character  :
            return xrange(1, self.__precision._enveloping_content_bound(), 2)
        else :
            return xrange(1, self.__precision._enveloping_content_bound())
    
    def _semireduced_vector_valued_indices_with_discriminant_offset(self, epsilon, with_character = False) :
        """
        Return a list of all possible indices `\theta` of a vector valued modular
        form that occure as the upper right entry of a Fourier index with content ``eps``.
        These indices are semireduced in the sense that they are reduced to the extend
        that `\epsilon \mid \theta` and `\theta/\epsilon` is reduced.
        In addition the corresponding offset of the index' discriminant is iterated.
        
        INPUT::
            - `\epsilon`         -- Integer; the content of a binary quadratic form.
            - ``with_character`` -- Boolean (optional; default is ``False``)
        
        OUTPUT::
            List of pairs `(\theta, \mathrm{offset})`.
            `\theta`          -- An element of `\mathcal{o}^# / \mathcal{o}` represented
                                 as in ``_reduce_vector_valued_index``.
            `\mathrm{offset}` -- Integer; Modulo `-D`.

        TESTS::
            sage: fac = HermitianModularFormD2Factory(2,-3)
            sage: fac._semireduced_vector_valued_indices_with_discriminant_offset(1)
            [((0, 0), 0), ((1, 0), 2), ((-1, 0), 2)]
            sage: fac._semireduced_vector_valued_indices_with_discriminant_offset(3)
            [((0, 0), 0), ((3, 0), 0), ((-3, 0), 0)]
            sage: fac._semireduced_vector_valued_indices_with_discriminant_offset(2)
            [((0, 0), 0), ((2, 0), 2), ((-2, 0), 2)]
            sage: fac._semireduced_vector_valued_indices_with_discriminant_offset(2,True)
            Traceback (most recent call last):
            ...
            ValueError: Character forms are only available if 4 \div D.
        """
        D = self.__D
        
        if with_character :
            if D % 4 != 0 :
                raise ValueError( "Character forms are only available if 4 \div D.")
        
            raise NotImplementedError
        
            ## This depends on 2 \ndiv eps. Is this true? Are the offsets correct?
            if self.__D % 8 == 0 :
                return map( lambda t: ((t,0), (-t**2) % (-D)), range(0, -2*D*epsilon, 2*epsilon) )
            else :
                return map( lambda t: ((t,0), (-t**2) % (-D)), range(epsilon, -2*D*epsilon, 2*epsilon) )
        else :
            return map( lambda t: ( tuple(map( lambda e: epsilon * e, self._reduce_vector_valued_index((t,0)) )),
                                    (- t**2 * epsilon**2) % (-D) ),
                        range(0, -D) )
                
    def _iterator_discriminant(self, epsilon, offset) :
        """
        Return iterator over all possible discriminants of a Fourier index given a content ``eps``
        and a discriminant offset modulo `D`.
        
        INPUT:
            - `\epsilon` -- Integer; Content of an index.
            - ``offset`` -- Integer; Offset modulo D of the discriminant.

        OUTPUT:
            - Iterator over Integers.

        TESTS::
            sage: fac = HermitianModularFormD2Factory(4,-3)
            sage: list(fac._iterator_discriminant(2, 3))
            [0, 12, 24]
            sage: list(fac._iterator_discriminant(2, 2))
            [8, 20]
            sage: list(fac._iterator_discriminant(1, 2))
            [2, 5, 8, 11, 14, 17, 20, 23, 26]
        """
        mD = -self.__D
        periode = lcm(mD, epsilon**2)
        offset = crt(offset, 0, mD, epsilon**2) % periode 
        
        return xrange( offset,
                       self.__precision._enveloping_discriminant_bound(),
                       periode )
         
    def additive_lift(self, forms, weight, with_character = False, is_integral = False) :
        """
        Borcherds additive lift to hermitian modular forms of
        degree `2`. This coinsides with Gritsenko's arithmetic lift after
        using the theta decomposition.
        
        INPUT:
            - ``forms``          -- A list of functions accepting an integer and
                                    returning a q-expansion.
            - ``weight``         -- A positive integer; The weight of the lift.
            - ``with_character`` -- A boolean (default: ``False``); Whether the
                                    lift has nontrivial character.
            - ``is_integral``    -- A boolean (default: ``False``); If ``True``
                                    use rings of integral q-expansions over `\Z`.
        
        ALGORITHME:
            We use the explicite formulas in [D].

        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fegenerators import HermitianModularFormD2AdditiveLift
            sage: HermitianModularFormD2AdditiveLift(4, [1,0,0], -3, 4).coefficients()
            {(2, 3, 2, 2): 720, (1, 1, 1, 1): 27, (1, 0, 0, 2): 270, (3, 3, 3, 3): 2943, (2, 1, 1, 3): 2592, (0, 0, 0, 2): 9, (2, 2, 2, 2): 675, (2, 3, 2, 3): 2160, (1, 1, 1, 2): 216, (3, 0, 0, 3): 8496, (2, 0, 0, 3): 2214, (1, 0, 0, 3): 720, (2, 1, 1, 2): 1080, (0, 0, 0, 1): 1, (3, 3, 2, 3): 4590, (3, 1, 1, 3): 4590, (1, 1, 1, 3): 459, (2, 0, 0, 2): 1512, (1, 0, 0, 1): 72, (0, 0, 0, 0): 1/240, (3, 4, 3, 3): 2808, (0, 0, 0, 3): 28, (3, 2, 2, 3): 4752, (2, 2, 2, 3): 1350}
            sage: HermitianModularFormD2AdditiveLift(4, [0,1,0], -3, 6).coefficients()
            {(2, 3, 2, 2): -19680, (1, 1, 1, 1): -45, (1, 0, 0, 2): -3690, (3, 3, 3, 3): -306225, (2, 1, 1, 3): -250560, (0, 0, 0, 2): 33, (2, 2, 2, 2): -13005, (2, 3, 2, 3): -153504, (1, 1, 1, 2): -1872, (3, 0, 0, 3): -1652640, (2, 0, 0, 3): -295290, (1, 0, 0, 3): -19680, (2, 1, 1, 2): -43920, (0, 0, 0, 1): 1, (3, 3, 2, 3): -948330, (3, 1, 1, 3): -1285290, (1, 1, 1, 3): -11565, (2, 0, 0, 2): -65520, (1, 0, 0, 1): -240, (0, 0, 0, 0): -1/504, (3, 4, 3, 3): -451152, (0, 0, 0, 3): 244, (3, 2, 2, 3): -839520, (2, 2, 2, 3): -108090}
        """
        if with_character and self.__D % 4 != 0 :
            raise ValueError( "Characters are only possible for even discriminants." )

        ## This will be needed if characters are implemented
        if with_character :
            if (Integer(self.__D / 4) % 4) in [-2,2] :
                alpha = (-self.__D / 4, 1/2)
            else :
                alpha = (-self.__D / 8, 1/2)
        
        #minv = 1/2 if with_character else 1
        
        R = self.power_series_ring()
        q = R.gen(0)
            
        (vv_expfactor, vv_basis) = self._additive_lift_vector_valued_basis()
        
        vvform = dict((self._reduce_vector_valued_index(k), R(0)) for (k,_) in self._semireduced_vector_valued_indices_with_discriminant_offset(1))

        for (f,b) in zip(forms, vv_basis) :
            ## We have to apply the scaling of exponents to the form
            f = R( f(self._qexp_precision()) ).add_bigoh(self._qexp_precision()) \
                 .subs({q : q**vv_expfactor})
            
            if not f.is_zero() :
                for (k,e) in b.iteritems() :
                    vvform[k] = vvform[k] + e * f
        
        ## the T = matrix(2,[*, t / 2, \bar t / 2, *] th fourier coefficients of the lift
        ## only depends on (- 4 * D * det(T), eps = gcd(T), \theta \cong t / eps)
        ## if m != 1 we consider 2*T
        maass_coeffs = dict()

        ## TODO: use divisor dictionaries
        if not with_character :
            ## The factor for the exponent of the basis of vector valued forms
            ## and the factor D in the formula for the discriminant are combined
            ## here 
            vv_expfactor = vv_expfactor // (- self.__D)
            for eps in self._iterator_content() : 
                for (theta, offset) in self._semireduced_vector_valued_indices_with_discriminant_offset(eps) :
                    for disc in self._iterator_discriminant(eps, offset) :
                        maass_coeffs[(disc, eps, theta)] = \
                             sum( a**(weight-1) *
                                  vvform[self._reduce_vector_valued_index((theta[0]//a, theta[1]//a))][vv_expfactor * disc // a**2]
                                  for a in divisors(eps))
        else :
            ## The factor for the exponent of the basis of vector valued forms
            ## and the factor D in the formula for the discriminant are combined
            ## here 
            vv_expfactor = (2 * vv_expfactor) // (- self.__D) 

            if self.__D // 4 % 2 == 0 :
                for eps in self._iterator_content() : 
                    for (theta, offset) in self._semireduced_vector_valued_indices_with_discriminant_offset(eps) :
                        for disc in self._iter_discriminant(eps, offset) :
                            maass_coeffs[(disc, eps, theta)] = \
                                 sum( a**(weight-1) * (1 if (theta[0] + theta[1] - 1) % 4 == 0 else -1) *
                                      vvform[self._reduce_vector_valued_index((theta[0]//a, theta[1]//a))][vv_expfactor * disc // a**2]
                                      for a in divisors(eps))
            else :
                for eps in self._iterator_content() : 
                    for (theta, offset) in self._semireduced_vector_valued_indices_with_discriminant_offset(eps) :
                        for disc in self._iter_discriminant(eps, offset) :
                            maass_coeffs[(disc, eps, theta)] = \
                                 sum( a**(weight-1) * (1 if (theta[1] - 1) % 4 == 0 else -1) *
                                      vvform[self._reduce_vector_valued_index((theta[0]//a, theta[1]//a))][vv_expfactor * disc // a**2]
                                      for a in divisors(eps) )
        lift_coeffs = dict()
        ## TODO: Check whether this is correct. Add the character as an argument.
        for ((a,b1,b2,c), eps, disc) in self.precision().iter_positive_forms_for_character_with_content_and_discriminant(for_character = with_character) :
            (theta1, theta2) = self._reduce_vector_valued_index((b1/eps, b2/eps))
            theta = (eps * theta1, eps * theta2)
            try:
                lift_coeffs[(a,b1,b2,c)] = maass_coeffs[(disc, eps, theta)]
            except :
                raise RuntimeError(str((a,b1,b2,c)) + " ; " + str((disc, eps, theta)))

        # Eisenstein component
        for (_,_,_,c) in self.precision().iter_semidefinite_forms_for_character(for_character = with_character) :
            if c != 0 :
                lift_coeffs[(0,0,0,c)] = vvform[(0,0)][0] * sigma(c, weight - 1)
            
        lift_coeffs[(0,0,0,0)] = - vvform[(0,0)][0] * bernoulli(weight) / Integer(2 * weight)
        if is_integral :
            lift_coeffs[(0,0,0,0)] = ZZ(lift_coeffs[(0,0,0,0)])
        
        return lift_coeffs
    
    def _repr_(self) :
        """
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fegenerators import HermitianModularFormD2Factory_class
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal 
            sage: HermitianModularFormD2Factory_class(HermitianModularFormD2Filter_diagonal(2, -3))
            Factory for Fourier expansions of hermitian modular forms with precision Reduced diagonal filter for discriminant -3 with bound 2
        """
        return "Factory for Fourier expansions of hermitian modular forms with precision %s" % (self.__precision,)

    def __hash__(self) :
        """
        TESTS::
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fegenerators import HermitianModularFormD2Factory_class
            sage: from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal 
            sage: hash(HermitianModularFormD2Factory_class(HermitianModularFormD2Filter_diagonal(2, -3)))
            5669639111474785029 # 64-bit
            ?? # 32-bit
        """
        return hash(self.__precision) + hash(self.power_series_ring())
