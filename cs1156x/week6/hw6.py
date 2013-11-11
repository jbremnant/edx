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
===================================

Deterministic noise depends on H, as some models approx f better than others.
Assume H' in H and that f is fixed. If we use H' instead of H, how does 
deterministic noise behave?

  [a] in general, deterministic noise will decrease?

  BOO! got this stupid question wrong.
"""


"""
Overfitting and Regularization With Weight Decay
================================================

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

# wow, much easier way exists:
# np.genfromtxt("http://work.caltech.edu/data/in.dta", dtype=float)
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



"""
Regularization With Polynomials
===============================

nonlinear transform phi: x -> z, where phi transforms scalar x into vector z of
Legendre polynomials (orthogonal),  z = (1,L1(x),L2(x),...,Lq(x))

  H_Q = { h|h(x) = w'z = sum_{q=0}^{Q}(wq Lq(x)) }

  L0(x) = 1


7. Consider the following hypothesis set defined by the constraint:

  H(Q,C,Q0) = { h|h(x) = w'z in H_Q; w_q = C for q>=Q0 }

  In words: hypothesis set considers Legendre polynomials where any order greater
  or equal to Q0 gets the weight w_q constrained by some constant C.  
  Thus, weight of any polynomial of order less than Q0 will be allowed to vary,
  while the weights greater or equal to Q0 will be fixed to C.

  [c] H(10,0,3) inter H(10,0,4) = H2

  Since 
    H(10,0,3) = L0 + w1*L1(x) + w2*L2(x) + 0*L3(x)  + 0*L4(x) + ... + 0
    H(10,0,4) = L0 + w1*L1(x) + w2*L2(x) + w3*L3(x) + 0*L4(x) + ... + 0
    H(10,0,3) inter H(10,0,4) = L0 +  w1*L1(x) + w2*L2(x) = H2

  The intersection of the 2 hypothesis would be just H2.
"""

"""
Neural Networks
===============
Watch Lecture 10. This topic is not covered in the book.

Neural network is all about notations. Print your neural network and then code it...

             | 1 <= l <= L             layers
  w_ij^(l) = | 0 <= i <= d^(l-1)       inputs
             | 1 <= j <= d^(l)         ouputs

  x_j^(l) = theta( s_j^(l) )  =  theta( sum_{i=0}^{d^(l-1)} w_ij^(l) x_i^(l-1) )

  where theta(s) = tanh(s) = (e^s - e^-s) / (e^s + e^-s)

Apply x to x_1^(0) ... x_{d^(0)}^(0) ->->  x_1^(L) = h(x)

SGD on neural networks:

All the weights w = {w_ij^(l)} determine h(x)

Error on example is (xvec_n, y_n) is
  e(h(xvec_n), y_n) = e(w)

To implement SGD, we need the gradient
  
  Delta e(w) :  partial_d e(w) / partial_d w_ij^(l)  for all i,j,l

    d e(w)/d w_ij^(l) = d e(w) / d s_j^(l)  x  d s_j^(l) / d w_ij^(l)

  we have         d s_j^(l) / d w_ij^(l) = x_i^(l-1)
  we only need:   d e(w) / d s_j^(l) = delta_j^(l)

  delta for the final layer: l = L and j = 1

    delta_j^(l) = d e(w) / d s_j^(l)  
    delta_1^(L) = d e(w) / d s_1^(L)  
    e(w) = (x_1^(L) - y_n)^2
    x_1^(L) = theta(s_1^(L))

    theta'(s) = 1 - theta(s)^2



8. A fully connected Neural Network has
    L = 2
    d^(0) = 5
    d^(1) = 3
    d^(2) = 1
  If only products of the form
    w_ij^(l) x_i^(l-1)
    w_ij^(l) d_j^(l)
    x_i^(l-1) d_j^(l)
  count as operations (even for x_0^(l-1) = 1), without counting anything else, which
  of the following is closest to the total number of operations required in a single
  iteration of backpropagation (using SGD on one data point?)

  [d] 45

  You have to carefully count the number of operations on 3 areas:
    
    1. forward calc for x_ij             :  w_ij^(l) x_i^(l-1)
    2. backward calc for all the deltas  :  w_ij^(l) d_j^(l)
    3. updating the weights w_ij         :  x_i^(l-1) d_j^(l)
  
    so it's    5*3 + 3*1  + (5*3+3*1) + (5+3+1)  =  18+18+9 = 45

A Neural Network has 10 input units (the constant x_0^(0) is counted here as a unit),
one output unit, and 36 hidden units (each x_0^(l) is also counted as a unit). The
hidden units can be arranged in any number of layer l=1,...,L-1,  and each layer is
fully connected to the layer above it.

* For these problems, refer to the diagram on slide 13

9. What is the minimum possible number of weights that such a network can have?

  If you had just 2 nodes per layer, you end up with 18 layers.
  The activation for intercept on layer l doesn't exist, so with 2 nodes,
  you only have 2 w's going into 1 node at next layer.
  So you will get 

    10*1 + 2*18 = 46

  [a] 46
    
10. What is the maximum possible number of weights that such a network can have?

  max seems to be when you have 2 layers.
  for example, with 23 nodes on 1st layer, 13 nodes on the 2nd layer
    >>> 10*22 + 23*12 + 13
    509

  The max occurs when you have 22 for the first layer, at 510.

    >>> zip( range(1,36), map(lambda(x): 10*(x-1) + (x)*(36-x-1) + (36-x), range(1,36)))
    [(1, 69), (2, 110), (3, 149), (4, 186), (5, 221), (6, 254), (7, 285), (8, 314), (9, 341), (10, 366), (11, 389), (12, 410), (13, 429), (14, 446), (15, 461), (16, 474), (17, 485), (18, 494), (19, 501), (20, 506), (21, 509), (22, 510), (23, 509), (24, 506), (25, 501), (26, 494), (27, 485), (28, 474), (29, 461), (30, 446), (31, 429), (32, 410), (33, 389), (34, 366), (35, 341)]

  [e] 510

"""
