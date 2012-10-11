"""
Types of Hermitian modular forms of degree 2 for the full modular group.

AUTHOR :
    -- Martin Raum, Dominic Gehre (2010 - 04 - 26) Initial version.
"""

#===============================================================================
# 
# Copyright (C) 2010 Martin Raum, Dominic Gehre
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

from psage.modform.fourier_expansion_framework.gradedexpansions.gradedexpansion_grading import DegreeGrading
from psage.modform.fourier_expansion_framework.modularforms.modularform_ambient import ModularFormsRing_generic
from psage.modform.fourier_expansion_framework.modularforms.modularform_types import ModularFormType_abstract
from psage.modform.hermitianmodularforms.hermitianmodularformd2_fegenerators import HermitianModularFormD2AdditiveLift
from psage.modform.hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2FourierExpansionRing
from operator import xor
from sage.categories.number_fields import NumberFields
from sage.misc.cachefunc import cached_method
from sage.misc.latex import latex
from sage.modular.modform.eis_series import eisenstein_series_qexp
from sage.rings.all import Integer, fundamental_discriminant
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.rational_field import QQ
from sage.structure.all import Sequence

#===============================================================================
# HermitianModularFormsD2
#===============================================================================

_hermitianmodularforms_cache = dict()

def HermitianModularFormsD2(A, type, precision, *args, **kwds) :
    """
    Create a globaly unique ring of hermitian modular forms of specified type.
    
    INPUT:
        - `A`           -- A ring; The ring of Fourier coefficients.
        - ``type``      -- A modular forms type instance.
        - ``precision`` -- A precision class; Precision of associated Fourier
                           expansions.
    
    OUTPUT:
        An instance of :class:~fourier_expansion_framework.modularforms.modularform_ambient.ModularFormsAmbient_abstract`. 

    TESTS::
        sage: from hermitianmodularforms import *
            sage: from hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal
        sage: HermitianModularFormsD2(QQ, HermitianModularFormD2_Gamma(-3), HermitianModularFormD2Filter_diagonal(4, -3))
        Graded expansion ring with generators HE4, HE6, HE10, HE12, Hphi9
    """
    global _hermitianmodularforms_cache
    k = (A, type, precision)
    
    try :
        return _hermitianmodularforms_cache[k]
    except KeyError :
        if isinstance(type, HermitianModularFormD2_Gamma) :
            M = ModularFormsRing_generic(A, type, precision)
        else :
            raise TypeError( "%s must be a type for hermitian modular forms of degree 2." % (type,) )
        
        _hermitianmodularforms_cache[k] = M
        return M

#===============================================================================
# HermitianModularFormsD2_Gamma
#===============================================================================

class HermitianModularFormD2_Gamma ( ModularFormType_abstract ) :
    """
    Type of hermitian modular forms of degree `2` associated to
    the full modular group over `\Q(\sqrt{D})`.
    
    This is currently only available for `D = -3`.  
    """
    def __init__(self, D) :
        """
        INPUT:
            - `D` -- A negative integer; The fundamental discriminant of
                     an imaginary quadratic field.
        
        TESTS::
            sage: from hermitianmodularforms import *
            sage: t = HermitianModularFormD2_Gamma(-3)
            sage: HermitianModularFormD2_Gamma(-1)
            Traceback (most recent call last):
            ...
            ValueError: D must be a negative fundamental discriminant.
            sage: HermitianModularFormD2_Gamma(5)
            Traceback (most recent call last):
            ...
            ValueError: D must be a negative fundamental discriminant.
            sage: HermitianModularFormD2_Gamma(-4)
            Traceback (most recent call last):
            ...
            NotImplementedError: Only discriminant -3 is implemented.
        """
        if D >= 0 or not D == fundamental_discriminant(D) :
            raise ValueError( "D must be a negative fundamental discriminant." )
        
        if D != -3 :
            raise NotImplementedError( "Only discriminant -3 is implemented." )
        
        self.__D = D
        
    def _ambient_construction_function(self) :
        """
        Return a function that will construct the ambient ring or module
        of modular forms.
        
        OUTPUT:
            A function with INPUT:
                - `A`      -- A ring or module; The Fourier coefficients' domain.
                - ``type`` -- A type of modular forms.
                - ``precision`` -- A precision class; The Fourier expansion's precision.
            and OUTPUT:
                A ring or module of modular forms.
        
        TESTS::
            sage: from hermitianmodularforms.hermitianmodularformd2_types import HermitianModularFormsD2, HermitianModularFormD2_Gamma
            sage: HermitianModularFormD2_Gamma(-3)._ambient_construction_function() is HermitianModularFormsD2
            True
        """
        return HermitianModularFormsD2
    
    def group(self) :
        """
        The modular group which ``self`` is associated with.
        
        OUTPUT:
            An arbitrary type.
        
        NOTE:
            The framwork might change later such that this function has
            to return a group.
        
        TESTS::
            sage: from hermitianmodularforms import *
            sage: HermitianModularFormD2_Gamma(-3).group()
            '\\mathrm{Sp}(2, \\mathfrak{o}_{\\QQ(-3)})'
        """
        
        
        return "\mathrm{Sp}(2, \mathfrak{o}_{\QQ(%s)})" % (self.__D,)
        
    def _e4_D3(self, precision) :
        """
        The Eisenstein series of weight `4` associated to the full modular
        group over `\QQ(-3)`.
        
        INPUT:
            -``precision`` -- A precision class; The precision of the Fourier
                              expansion.
        
        OUTPUT:
            An equivariant monoid power series over `\Z`.
        
        TESTS::
            sage: from hermitianmodularforms import *
            sage: from hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal
            sage: HermitianModularFormD2_Gamma(-3)._e4_D3(HermitianModularFormD2Filter_diagonal(2, -3)).parent().coefficient_domain()
            Integer Ring
        """
        return HermitianModularFormD2AdditiveLift( precision,
                    [240, 0, 0], weight = 4,
                    is_integral = True )

    def _e6_D3(self, precision) :
        """
        The Eisenstein series of weight `6` associated to the full modular
        group over `\QQ(-3)`.
        
        INPUT:
            -``precision`` -- A precision class; The precision of the Fourier
                              expansion.
        
        OUTPUT:
            An equivariant monoid power series over `\Z`.
        
        TESTS::
            sage: from hermitianmodularforms import *
            sage: from hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal
            sage: HermitianModularFormD2_Gamma(-3)._e6_D3(HermitianModularFormD2Filter_diagonal(2, -3)).parent().coefficient_domain()
            Integer Ring
        """
        return HermitianModularFormD2AdditiveLift( precision,
                    [0, -504, 0], weight = 6,
                    is_integral = True )
        
    def _e10_D3(self, precision) :
        """
        The Eisenstein series of weight `10` associated to the full modular
        group over `\QQ(-3)`.
        
        INPUT:
            -``precision`` -- A precision class; The precision of the Fourier
                              expansion.
        
        OUTPUT:
            An equivariant monoid power series over `\Z`.
        
        TESTS::
            sage: from hermitianmodularforms import *
            sage: from hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal
            sage: HermitianModularFormD2_Gamma(-3)._e10_D3(HermitianModularFormD2Filter_diagonal(2, -3)).parent().coefficient_domain()
            Integer Ring
        """
        ell_e4 = lambda p: -105336 * 240 * eisenstein_series_qexp(4, p)
        ell_e6 = lambda p: -108240 * (-504) * eisenstein_series_qexp(6, p)

        return HermitianModularFormD2AdditiveLift( precision,
                    [ell_e6, ell_e4, 0], weight = 10,
                    is_integral = True )

    def _e12_D3(self, precision) :
        """
        The Eisenstein series of weight `12` associated to the full modular
        group over `\QQ(-3)`.
        
        INPUT:
            -``precision`` -- A precision class; The precision of the Fourier
                              expansion.
        
        OUTPUT:
            An equivariant monoid power series over `\Z`.
        
        TESTS::
            sage: from hermitianmodularforms import *
            sage: from hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal
            sage: HermitianModularFormD2_Gamma(-3)._e12_D3(HermitianModularFormD2Filter_diagonal(2, -3)).parent().coefficient_domain()
            Integer Ring
        """
        ell_e6 = lambda p: 42588000 * (-504) * eisenstein_series_qexp(6, p)
        ell_e8 = lambda p: 78427440 * 480 * eisenstein_series_qexp(8, p)

        return HermitianModularFormD2AdditiveLift( precision,
                    [ell_e8, ell_e6, 0], weight = 12,
                    is_integral = True )
    
    def _phi9_D3(self, precision) :
        """
        The Borcherds product of weight `9` associated to the full modular
        group over `\QQ(-3)`.
        
        INPUT:
            -``precision`` -- A precision class; The precision of the Fourier
                              expansion.
        
        OUTPUT:
            An equivariant monoid power series over `\Z`.
        
        TESTS::
            sage: from hermitianmodularforms import *
            sage: from hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal
            sage: HermitianModularFormD2_Gamma(-3)._phi9_D3(HermitianModularFormD2Filter_diagonal(2, -3)).parent().coefficient_domain()
            Integer Ring
        """
        return HermitianModularFormD2AdditiveLift( precision,
                    [0, 0, 1], weight = 9,
                    is_symmetric = False,
                    is_integral = True )
        
    @cached_method
    def generators(self, K, precision) :
        """
        A list of Fourier expansions of forms that generate the ring
        or module of modular forms.
        
        INPUT:
            - `K`           -- A ring or module; The ring of Fourier coefficients.
            - ``precision`` -- A precision class; The precision of the Fourier
                               expansions.
        
        OUTPUT:
            A sequence of equivariant monoid power series.
        
        TESTS::
            sage: from hermitianmodularforms import *
            sage: from hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2Filter_diagonal
            sage: from hermitianmodularforms.hermitianmodularformd2_fourierexpansion import HermitianModularFormD2FourierExpansionRing
            sage: HermitianModularFormD2_Gamma(-3).generators(QQ, HermitianModularFormD2Filter_diagonal(2, -3)).universe() == HermitianModularFormD2FourierExpansionRing(QQ, -3)
            True
            sage: HermitianModularFormD2_Gamma(-3).generators(GF(2), HermitianModularFormD2Filter_diagonal(2, -3))
            Traceback (most recent call last):
            ...
            NotImplementedError: Only Fourier coefficients in a number fields are implemented.
        """
        if self.__D == -3 :
            if  K is QQ or K in NumberFields() :
                return Sequence( [ self._e4_D3(precision), self._e6_D3(precision),
                                   self._e10_D3(precision), self._e12_D3(precision),
                                   self._phi9_D3(precision) ],
                                 universe = HermitianModularFormD2FourierExpansionRing(QQ, -3) )
        
            raise NotImplementedError( "Only Fourier coefficients in a number fields are implemented." )
        
        raise NotImplementedError( "Discriminant %s is not implemented." % (self.__D,) )
    
    def grading(self, K) :
        """
        A grading for the ring or module of modular forms.
        
        INPUT:
            - `K` -- A ring or module; The ring of Fourier coefficients.
        
        OUTPUT:
            A grading class.
        
        NOTE:
            This coincides with the weight grading.
        
        TESTS::
            sage: from hermitianmodularforms import *
            sage: HermitianModularFormD2_Gamma(-3).grading(QQ)
            Degree grading (4, 6, 10, 12, 18)
            sage: HermitianModularFormD2_Gamma(-3).grading(GF(2))
            Traceback (most recent call last):
            ...
            NotImplementedError: Only Fourier coefficients in a number fields are implemented.
        """
        if self.__D == -3 :
            if K is QQ or K in NumberFields() :
                return DegreeGrading([4, 6, 10, 12, 18])

            raise NotImplementedError( "Only Fourier coefficients in a number fields are implemented." )
        
        raise NotImplementedError( "Discriminant %s is not implemented." % (self.__D,) )

    def _generator_names(self, K) :
        """
        Names of the generators returned by :meth:~`.generators` within the
        attached polynomial ring.
        
        INPUT:
            - `K` -- A ring or module; The ring of Fourier coefficients.
        
        OUTPUT:
            A list of strings.
        
        TESTS::
            sage: from hermitianmodularforms import *
            sage: HermitianModularFormD2_Gamma(-3)._generator_names(QQ)
            ['HE4', 'HE6', 'HE10', 'HE12', 'Hphi9']
            sage: HermitianModularFormD2_Gamma(-3)._generator_names(GF(2))
            Traceback (most recent call last):
            ...
            NotImplementedError: Only Fourier coefficients in a number fields are implemented.
        """
        if self.__D == -3 :
            if K is QQ or K in NumberFields() :
                return ['HE4', 'HE6', 'HE10', 'HE12', 'Hphi9']
            
            raise NotImplementedError( "Only Fourier coefficients in a number fields are implemented." )
        
        raise NotImplementedError( "Discriminant %s is not implemented." % (self.__D,) )

    def _generator_by_name(self, K, name) :
        """
        Return the generator ``name`` as an element of the attached
        polynomial ring.
        
        INPUT:
            - `K`      -- A ring or module; The ring of Fourier coefficients.
            - ``name`` -- A string; The generator's name.
        
        OUTPUT:
            An element of a polynomial ring.
        
        TESTS::
            sage: from hermitianmodularforms import *
            sage: HermitianModularFormD2_Gamma(-3)._generator_by_name(QQ, "HE4")
            HE4
            sage: HermitianModularFormD2_Gamma(-3)._generator_by_name(QQ, "???")
            Traceback (most recent call last):
            ...
            ValueError: Name ??? does not exist for Fourier coefficient domain Rational Field.
            sage: HermitianModularFormD2_Gamma(-3)._generator_by_name(GF(2), "HE4")
            Traceback (most recent call last):
            ...
            NotImplementedError: Only Fourier coefficients in a number fields are implemented.
        """
        if self.__D == -3 :
            if K is QQ or K in NumberFields() :
                R = self.generator_relations(K).ring()
                try :
                    return R.gen(self._generator_names(K).index(name))
                except ValueError :
                    raise ValueError, "Name %s does not exist for Fourier coefficient domain %s." % (name, K)

            raise NotImplementedError( "Only Fourier coefficients in a number fields are implemented." )
        
        raise NotImplementedError( "Discriminant %s is not implemented." % (self.__D,) )
    
    @cached_method
    def generator_relations(self, K) :
        """
        An ideal `I` in the attach polynomial ring `R`, such that the ring of
        modular forms is a quotient of `R / I`. This ideal must be unique for `K`.
        
        INPUT:
            - `K`      -- A ring or module; The ring of Fourier coefficients.
        
        OUTPUT:
            An ideal in a polynomial ring.
            
        TESTS::
            sage: from hermitianmodularforms import *
            sage: HermitianModularFormD2_Gamma(-3).generator_relations(QQ)
            Ideal (0) of Multivariate Polynomial Ring in HE4, HE6, HE10, HE12, Hphi9 over Rational Field
            sage: HermitianModularFormD2_Gamma(-3).generator_relations(GF(2))
            Traceback (most recent call last):
            ...
            NotImplementedError: Only Fourier coefficients in a number fields are implemented.
        """
        if self.__D == -3 :
            if K is QQ or K in NumberFields() :
                R = PolynomialRing(K, self._generator_names(K))
                return R.ideal(0)

            raise NotImplementedError( "Only Fourier coefficients in a number fields are implemented." )
        
        raise NotImplementedError( "Discriminant %s is not implemented." % (self.__D,) )
    
    def graded_submodules_are_free(self, K = None) :
        """
        Whether the modules of elements of fixed grading are free over
        their base ring `K' or over all base rings, respectively.
        
        INPUT:
            - `K` -- A ring or module or None (default: None) 
        
        OUTPUT:
            A boolean.
        
        TESTS::
            sage: from hermitianmodularforms import *
            sage: HermitianModularFormD2_Gamma(-3).graded_submodules_are_free()
            True
        """
        return True

    def __cmp__(self, other) :
        """
        TESTS::
            sage: from hermitianmodularforms import *
            sage: HermitianModularFormD2_Gamma(-3) == HermitianModularFormD2_Gamma(-3)
            True
        """
        c = cmp(type(self), type(other))
        
        if c == 0 :
            c = cmp(self.__D, other.__D)
                        
        return c
    
    def _repr_(self) :
        """
        TESTS::
            sage: from hermitianmodularforms import *
            sage: HermitianModularFormD2_Gamma(-3)
            Type for Hermitian modular forms associated to \mathrm{Sp}(2, \mathfrak{o}_{\QQ(-3)})
        """
        return "Type for Hermitian modular forms associated to %s" % (self.group(),)

    def _latex_(self) :
        """
            sage: from hermitianmodularforms import *
            sage: latex(HermitianModularFormD2_Gamma(-3))
            Type for Hermitian modular forms associated to \texttt{\mathrm{Sp}(2, \mathfrak{o}_{\QQ(-3)})}
        """
        return "Type for Hermitian modular forms associated to %s" % (latex(self.group()),)


    def __hash__(self) :
        """
        TESTS::
            sage: from hermitianmodularforms import *
            sage: hash(HermitianModularFormD2_Gamma(-3))
            -57
        """
        return 19 * hash(self.__D)
