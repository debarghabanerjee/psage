from sage.matrix.all import Matrix
from psage.modform.weilrep import VectorValuedModularForms


def test_real_quadratic(minp=1,maxp=100,minwt=2,maxwt=1000):
    for p in prime_range(minp,maxp):
        if p%4==1:
            print "p = ", p
            gram=Matrix(ZZ,2,2,[2,1,1,(1-p)/2])
            M=VectorValuedModularForms(gram)
            if is_odd(minwt):
                minwt=minwt+1
            for kk in range(minwt,round(maxwt/2-minwt)):
                k = minwt+2*kk
                if M.dimension_cusp_forms(k)-dimension_cusp_forms(kronecker_character(p),k)/2 != 0:
                    print "ERROR: ", k, M.dimension_cusp_forms(k), dimension_cusp_forms(kronecker_character(p),k)/2
                    return false
    return true

#sys.path.append('/home/stroemberg/Programming/Sage/sage-add-ons3/nils')
#from jacobiforms.all import *

def test_jacobi(index=1,minwt=4,maxwt=100,eps=-1):
    m=index
    gram=Matrix(ZZ,1,1,[-eps*2*m])
    M=VectorValuedModularForms(gram)
    if is_odd(minwt):
        minwt=minwt+1
    for kk in range(0,round(maxwt/2-minwt)+2):
        k = minwt+2*kk+(1+eps)/2
        print "Testing k = ", k
        if eps==-1:
            dimJ=dimension_jac_forms(k,m,-1)
            dimV=M.dimension(k-1/2)
        else:
            dimJ=dimension_jac_cusp_forms(k,m,1)
            dimV=M.dimension_cusp_forms(k-1/2)
        if dimJ-dimV != 0:
            print "ERROR: ", k, dimJ, dimV
            return false
    return true
