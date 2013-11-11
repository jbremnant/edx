import sys
import math
import re
import random as rn
import numpy as np

"""
import sys
import hw6 as h
sys.path.append('/home/jbkim/git/edx/cs1156x/week6')
reload(h)
sys.path.append('/home/jbkim/git/edx/cs1156x/week2')
import hw2 as h2
"""

"""
Overfitting and Deterministic Noise
Question 1

Deterministic noise depends on H, as some models approx f better than others.
Assume H' in H and that f is fixed. If we use H' instead of H, how does 
deterministic noise behave?

  [a] in general, deterministic noise will decrease?
"""

"""
Overfitting and Regularization With Weight Decay
Question 2,3,4,5,6

  wget http://work.caltech.edu/data/in.dta  # training
  wget http://work.caltech.edu/data/out.dta # test

din  = h.readfile('/home/jbkim/git/edx/cs1156x/week6/in.dta')
dout = h.readfile('/home/jbkim/git/edx/cs1156x/week6/out.dta')

Each line of file represents  x = (x1,x2) so that X in Real and corresponding
label of Y = {-1,1}. We are going to apply Linear Regression with non-linear
transformation for classification.
  
  phi(x1,x2) = (1, x1, x2, x1^2, x2^2, x1*x2, |x1-x2|, |x1+x2|)

Recall that the classification error is defined as the fact of mis-classifed pts.

2. Run Linear Regression on training set after performing transformation. What
   are the closest Ein and Eout?
    >>> reload(h)
    >>> h.q2()
    {'Eout': 0.084000000000000005, 'w': array([-1.64706706, -0.14505927,  0.10154121, -2.03296844, -1.82804373, 2.48152945,  4.15893861,  0.31651714]), 'Ein': 0.028571428571428571}
  
    Ein : 0.02857
    Eout: 0.08400
  
    [a] 0.03, 0.08

    >>> h.euclid(np.array([0.02857142, 0.0840000]),np.array([0.03,0.08]))
    0.004247451096410649
    >>> h.euclid(np.array([0.02857142, 0.0840000]),np.array([0.03,0.10]))
    0.016063649672985278
    >>> h.euclid(np.array([0.02857142, 0.0840000]),np.array([0.04,0.09]))
    0.012907844158355799

3. Now add weight decay to Linear Regression, that is add the term:
   (lambda/N)sum_{i=0}^{7}( w_i^2 ) to the squared in-sample error, using lambda=10^k.
   What are the closest values to the Ein and Eout for k=-3, lambda=0.001 ? 

    w_lin = (Z'Z)^-1 Z'y
    w_reg = (Z'Z + lambda*I)^-1 Z'y   

    w_reg is pretty much the same as ridge regression.
  
    >>> h.q3()
    {'Eout': 0.080000000000000002, 'w': array([-1.6432827 , -0.14333537,  0.10144329, -2.02456533, -1.81721505, 2.45550685,  4.14009201,  0.31960135]), 'Ein': 0.028571428571428571}

    Ein : 0.02857
    Eout: 0.08000
    
    The out of sample error gets better but it's still the same choice:

    [d] 0.03, 0.08

4. Now use k=3 (lambda = 1000). What are the closest Ein, Eout?
    
    >>> h.q4()
    {'Eout': 0.436, 'w': array([ 0.00435688, -0.00134416,  0.0024939 ,  0.00328695,  0.00484127, -0.00862023,  0.01786706, -0.00490192]), 'Ein': 0.37142857142857144}

    Ein : 0.371
    Eout: 0.436

    [e] 0.4, 0.4

    since
    >>> h.euclid(np.array([0.371428, 0.436]),np.array([0.3,0.4]))
    0.07998724388300923
    >>> h.euclid(np.array([0.371428, 0.436]),np.array([0.4,0.4]))
    0.04596040887546586

5. What value of k have the smallest Eout?

    >>> h.q5()
    [0.228, 0.124, 0.092, 0.056, 0.228]
    [    2,     1,     0,    -1,    -2]

    [d] -1

6. What value is closest to the min Eout achieved by varying k?
   Limiting k to integers

    >>> e = h.q6()
    >>> [h.trunc(i,3) for i in e]
    ['0.084', '0.084', '0.084', '0.08', '0.084', '0.056', '0.092', '0.124', '0.228', '0.436', '0.452', '0.456']

    Looks like it goes down to 0.056

    [b] 0.06
"""

def readfile(fname):
  d = []
  with open(fname) as f:
    for l in f:
      l = l.strip()
      tok = map(lambda(x): float(x), re.split('\s+',l))
      d.append(tok)
  return(d)

def getdata(fname='/home/jbkim/git/edx/cs1156x/week6/in.dta'):
  d = readfile(fname)
  x = np.array([i[0:2] for i in d])
  y = np.array([i[2] for i in d])
  return(x,y)

def lm(X,y):
    Xt = np.transpose(X)
    m = np.matrix(np.dot(Xt, X))
    mi = m.getI()  # what kind of inversion method is this?
    beta = np.dot(np.dot(mi, Xt), y)
    return(beta.getA()[0,:])

def lm_ridge(X,y,lam):
    Xt = np.transpose(X)
    k = X.shape[1]
    lambdaeye = lam*np.eye(k)
    m = np.matrix(np.dot(Xt, X)+lambdaeye)
    mi = m.getI()  # what kind of inversion method is this?
    beta = np.dot(np.dot(mi, Xt), y)
    return(beta.getA()[0,:])

def transform(x):
  x1 = x[0]; x2 = x[1]  
  return( [1,x1,x2,x1**2,x2**2,x1*x2,abs(x1-x2),abs(x1+x2)])

def calc_class(x,w):
    z = transform(x)
    y = np.dot(z,w)
    return(1 if y>=0 else -1)

def geterr(X,y,w):
    n = X.shape[0]
    yhat = np.apply_along_axis(calc_class, 1, X, w)
    # true/false numerical conversion gives 1/0
    errcount = 1.0*np.sum( y*yhat < 0 )
    return(errcount/n)

def runlm(lam=0):
  (xin,yin)   = getdata('/home/jbkim/git/edx/cs1156x/week6/in.dta')
  (xout,yout) = getdata('/home/jbkim/git/edx/cs1156x/week6/out.dta')
  zin = np.apply_along_axis(transform, 1, xin)
  # w = lm(zin,yin)
  w = lm_ridge(zin,yin,lam)
  Ein = geterr(xin,yin,w)
  Eout = geterr(xout,yout,w)
  return({'w':w, 'Ein':Ein, 'Eout':Eout})

def euclid(x,y):
  return( math.sqrt( np.sum((x - y)**2) ) )

def q2():
  return(runlm(lam=0))

def q3():
  lam = 10**-3
  return(runlm(lam=lam))

def q4():
  lam = 10**3
  return(runlm(lam=lam))

def q5():
  ks   = [2,1,0,-1,2]
  lams = [10**k for k in ks]
  Eouts = map(lambda(x): runlm(lam=x)['Eout'], lams)
  return(Eouts)

def trunc(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    slen = len('%.*f' % (n, f))
    return str(f)[:slen]

def q6():
  ks   = range(-6,6)
  lams = [10**k for k in ks]
  Eouts = map(lambda(x): runlm(lam=x)['Eout'], lams)
  return(Eouts)
