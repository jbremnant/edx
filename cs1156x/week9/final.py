"""
CS1156x Final Exam 
2013.11.30

import sys
sys.path.append("/home/jbkim/Development/edx/cs1156x/week9")
import final as f
"""

import sys
import math
import re
import random as rn
import numpy as np
from sklearn import svm
from sklearn import cross_validation
import scipy.optimize as sci
from scipy.spatial.distance import pdist,squareform
import matplotlib.pyplot as pl

g_trainfile = "../week8/features.train"
g_testfile  = "../week8/features.test"

"""
Nonlinear transforms
====================
1. The polynomial tranform of order Q = 10 applied to X of dimension d = 2 results in 
   a Z space of what dimensionality (not counting the constant coordinate x0 = 1 or z0 = 1)?

    Looks like you have to take the cumulative dimensions of each "layer" in binomial
    expansion: For example:
    
    Q = 1  (x1,x2)^1 = (x1,x2)                      d = 2

    Q = 2  (x1,x2)^2 = (x1,x2,                      d = 2
                        x1^2,x1*x2,x2^2)              + 3
                                                        5

    Q = 3  (x1,x2)^3 = (x1,x2,                      d = 2
                        x1^2,x1*x2,x2^2,              + 3
                        x1^3,x1^2*x2,x1*x2^2,x2^3)    + 4
                                                        9

    Q = 10   cumsum(range(10+1))
      >>> 1+np.arange(1,Q+1)
      array([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
      >>> np.sum(1+np.arange(1,Q+1))
      65

    [e] none of the above
"""

"""
Bias and Variance
=================
2. Recall that the average hypothesis gbar was based on training the same model H on different
   data sets D to get g^(D) in H, and taking the expected value of g^(D) w.r.t D to get gbar.
   Which of the following model H could result in gbar in H?

    Wait... if you are training the same model H on just different datasets, can you end up
    with a hypothesis outside of the possible set of models that's dictated by H?
    Is this stupid "none of the above" question again?

    [a] A singleton H (H has one hypothesis) - then all training will always give 1 result
    [b] H is the set of constant, real-values hypothesis - h(x) = b all real numberse
    [c] H is the linear regression model - what does lm has to do with this?
  X [d] H is the logistic regression model - em.. same as [c]?
      : ah... the sigmoid function can't be added so..
    [e] none of the above?
"""

"""
Overfitting
===========
3. Which of the following statements is false? (here, FALSE!)
  
    overfitting definition:
      Ein(h1) < Ein(h2) AND Eout(h1) > Eout(h2)

    [a] If there is overfitting, there must be two or more hypotheses that have diff
        values of Ein
        : for models with different complexity, this is definitely possible, and varying
          hypotheses potentially leads to overfitting
    [b] If there is overfitting, there must be two or more hypotheses that have diff
        values of Eout
        : different Eout indicates that the hypotheses are different, more number of hypothesis
          will lead to overfitting?
    [c] If there is overfitting, there must be two or more hypotheses that have diff 
        values of (Eout - Ein)
        : this is the definition of the Omega function that expresses the diff of err terms
  X [d] We can always determine if there is overfitting by comparing the values of
        (Eout - Ein)
        : what if you had    Ein(h1,h2) = (0,2),  Eout(h1,h2) = (1,1) ??
                             Eout - Ein = (1,-1)   the second num is not overfitting?
    
    [e] We cannot determine overfitting based on one hypothesis only
        : yes we can

4. Which of the following statments is true? (here, TRUE!)

    [a] Deterministic noise cannot occur with stochastic noise
        : FALSE, why not?
    [b] Deterministic noise does not depend on the learning model
        : FALSE, yes it does depend on the learning model
    [c] Deterministic noise does not depend on the target function
        : FALSE, it does have some relationship, because target function lends itself to model
  X [d] Stochastic noise does not depend on the learning model
        : TRUE, regardless of the model you will have noise in the input data
    [e] Stochastic noise does not depend on the target distribution
        : FALSE, probabilistic target distribution is what gives rise to stochastic noise
"""

"""
Regularization
==============
5. The regularized weight w_reg is a solution to:

    minimize 1/N sum^{N}_{n=1} ( w'x_n - y_n )^2 
    s.t.  w' T'T w <= C

  where T is a matrix. If w_lin' T'T w_lin <= C, where w_lin is the linear regression
  solution, then what is w_reg?

    w_lin is the unconstrained solution, where as w_reg is the constrained, regularized solution.
    In lecture 12, slide 11:
    
    E_aug(w) = E_in(w) + lambda/N w' T'T w
             = 1/N [ (Zw - y)'(Zw - y) + lambda w' T'T w ]
      where  w' T'T w = ||w'T'||^2   if T is factorized matrix

    dE_aug(w) = 0  --> 
      Z'(Zw - y) + T w = 0
      Z'(Zw - y) + T w = 0
      Z'Zw - Z'y + T w = (Z'Z + T) w - Z'y
      w = (Z'Z + T)^-1 Z'y

    w_reg = (Z'Z + lambda*I)^-1 Z'y
    w_reg = (Z'Z + T)^-1 Z'y
    w_lin = (Z'Z)^-1 Z'y

  Hmm... the Tikhonov matrix is within inverse term?? None of the choices b,c,d,e are valid.
  Given the fact that the question is asking that w_lin already satisfies the constraint,
  w_lin is essentially equal to w_reg:

    [a] w_reg = w_lin
        

6. Soft-order constraints that regularize polynomial models can be

    [a] written as hard-order constraints
      : hard-order constraints do not have upper bound:  0 < alpha < infinity
  X [b] translated into augmented error
      : using constraints, you regularize the hypothesis which lends to Eaug (augmented error)
    [c] determined from the value of the VC dimension
      : no...
    [d] used to decrease both Ein and Eout
      : NO, regularization will penalize Ein and make it worse, but make Ein approximate Eout better
    [e] none of the above is true
      : [b] might be true?
"""

"""
Regularized Linear Regression
=============================
Use the data from previous hw on US postal service zip code.
Header:  digit symmetry intensity

Implement the regularized least-squares linear regression for classification that minimizes:

  (1/N) sum_{n=1}^{N} (w'z_n - y_n)^2 + (lambda/N) w'w

where w includes w0.

7. Set lambda = 1 and do not apply a feature transform (i.e. use z = x = (1,x1,x2)).
   Which among the following classifiers has the lowest Ein?

    >>> f.q7()
    {'Eout': array([ 0.0797,  0.0847,  0.0732,  0.0827,  0.0882]),
     'Ein':  array([ 0.0763,  0.0911,  0.0885,  0.0743,  0.0883])}

    [d] 8 versus all

8. Now, apply a feature transform z = (1,x1,z2,x1*x2,x1^2,x2^2), and set lambda = 1.
   Which among the following classifiers has the lowest Eout?

    >>> f.q8()
    {'Eout': array([ 0.1066,  0.0219,  0.0987,  0.0827,  0.0997]),
     'Ein':  array([ 0.1023,  0.0123,  0.1003,  0.0902,  0.0894])}

    [b] 1 versus all

9. If we compare using the transform versus not using it, and apply that to '0 versus all'
   through '9 versus all', which of the following statements is correct for lambda = 1?

    >>> np.set_printoptions(precision=4)
    >>> f.q9()
    {'Eout_x': array([ 0.1151,0.0224,0.0987,0.0827,0.0997,0.0797,0.0847,0.0732,0.0827,0.0882]),
     'Ein_x':  array([ 0.1093,0.0152,0.1003,0.0902,0.0894,0.0763,0.0911,0.0885,0.0743,0.0883]),
     'Eout_z': array([ 0.1066,0.0219,0.0987,0.0827,0.0997,0.0792,0.0847,0.0732,0.0827,0.0882]),
     'Ein_z':  array([ 0.1023,0.0123,0.1003,0.0902,0.0894,0.0763,0.0911,0.0885,0.0743,0.0883])}

    [a] Overfitting always occurs when we use the transform
      : FALSE, for digits 2-9, the errors are identical... regularization kicks in (check this)
    [b] The transform always improves the out-of-sample performance by at least 5%
      : FALSE, Eout better than Ein for some digits but not others
    [c] The transform does not make any difference in the out-of-sample performance
      : FALSE, first 2 digits are better on transform, but negligible
    [d] The transform always worsens the out-of-sample performance by at least 5%
      : FALSE, it makes 0,1 better
  X [e] The transform improves the out-of-sample performance of '5-versus-all,' but by less than 5%
      : TRUE, transformed 5-versus-all Eout better compared to non-tranformed Eout but by 0.6%

10. Train the '1 versus 5' classifier with z = (1,x1,x2,x1*x2,x1^2,x2^2) with lambda = 0.01
    and lambda = 1. Which of the following statments is correct?

    >>> f.q10()  # lam = [1, 0.01]
    {'Eout': array([ 0.0259434 ,  0.02830189]),
     'Ein':  array([ 0.00512492,  0.0044843 ])}

  X [a] Overfitting occurs (from lam=1 to lam=0.01)
      : TRUE, lam=0.01 gives smaller Ein but larger Eout compared to lam=1
    [b] The two classifiers have the same Ein 
      : FALSE
    [c] The two classifiers have the same Eout
      : FALSE
    [d] When lam goes up, both Ein and Eout go up
      : FALSE, lam goes up and Ein goes up, Eout goes down
    [e] When lam goes up, both Ein and Eout go down
      : FALSE
"""

def readdata(file=g_trainfile, intercept=True):
  d = np.genfromtxt(file, dtype=float)
  # adds the intercept term
  x = None
  if intercept:
    x = np.apply_along_axis(lambda(x): np.append(np.array([1.]),x[1:3]),1,d)
  else:
    x = np.apply_along_axis(lambda(x): x[1:3],1,d)
  y = np.apply_along_axis(lambda(x): x[0],1,d)
  return(x,y)

def getbinary(y,choice=0):
  z=np.ones(len(y))
  z[y!=choice] = -1.
  return(z)

def lm_ridge(x,y,lam=0):
  xt = np.transpose(x)
  k = x.shape[1]
  lambdaeye = lam*np.eye(k)
  m = np.matrix(np.dot(xt, x)+lambdaeye)
  mi = m.getI()  # what kind of inversion method is this?
  beta = np.dot(np.dot(mi, xt), y)
  w = beta.getA()[0,:]
  return(w)

# use this to check solutions: 
# (xin, yin) = f.readdata(f.g_trainfile)
# (xout, yout) = f.readdata(f.g_trainfile)
# f.runlm(f.transform(xin),f.getbinary(yin,5),f.transform(xout),f.getbinary(yout,5), lam=1.0)
# f.runlm(f.transform(xin),f.getbinary(yin,5),f.transform(xout),f.getbinary(yout,5), lam=0.01)
def runlm(x,y,xout,yout,lam=0):
  w = lm_ridge(x,y,lam=lam)
  Ein  = np.sum(np.dot(x,w) * y < 0)/(1.*y.shape[0])
  Eout = np.sum(np.dot(xout,w) * yout < 0)/(1.*yout.shape[0])
  return({'w':w,'Ein':Ein,'Eout':Eout})

def transform(x):
  z = np.apply_along_axis(lambda(x): np.array([1.,x[1],x[2],x[1]**2,x[2]**2,x[1]*x[2]]), 1, x)
  return(z)

# non-transform version: (1,x1,x2)
def q7(lam=1.0, ks=[5,6,7,8,9]):
  (x,yd)       = readdata(g_trainfile)
  (xout,ydout) = readdata(g_testfile)
  Ein  = np.array([])
  Eout = np.array([])
  for k in ks:
    y    = getbinary(yd,k)
    yout = getbinary(ydout,k)
    w    = lm_ridge(x,y, lam=lam)    
    yhat = np.dot(x, w) # only the sign of the prediction matters
    Ei   = np.sum(yhat*y  < 0) / (1.*x.shape[0])
    yhat = np.dot(xout,w)
    Eo   = np.sum(yhat*yout<0) / (1.*xout.shape[0])
    Ein  = np.append(Ein,  Ei) 
    Eout = np.append(Eout, Eo) 
  return({'Ein':Ein, 'Eout':Eout})

# transformed version: (1,x1,x2,x1^2,x2^2,x1*x2)
def q8(lam=1.0, ks=[0,1,2,3,4]):
  (xin, ydin)  = readdata(g_trainfile)
  (xout,ydout) = readdata(g_testfile)
  Ein  = np.array([])
  Eout = np.array([])
  for k in ks:
    yin,zin   = getbinary(ydin,k), transform(xin)
    yout,zout = getbinary(ydout,k), transform(xout)
    w    = lm_ridge(zin,yin, lam=lam)
    yhat = np.dot(zin, w)
    Ei   = np.sum(yhat*yin < 0) / (1.*zin.shape[0])
    yhat = np.dot(zout, w)
    Eo   = np.sum(yhat*yout < 0) / (1.*zout.shape[0])
    Ein  = np.append(Ein,  Ei) 
    Eout = np.append(Eout, Eo) 
  return({'Ein':Ein, 'Eout':Eout})

def q9():
  rx = q7(lam=1., ks=range(10))  
  rz = q8(lam=1., ks=range(10))  
  return({'Ein_x':rx['Ein'],'Eout_x':rx['Eout'],'Ein_z':rz['Ein'],'Eout_z':rz['Eout']})

def q10(lams=[1.0, 0.01]):
  (xin, ydin)  = readdata(g_trainfile)
  (xout,ydout) = readdata(g_testfile)
  i = np.logical_or(ydin==1, ydin==5)
  xin, ydin   = xin[i,:], ydin[i,:]
  i = np.logical_or(ydout==1, ydout==5)
  xout,ydout   = xout[i,:], ydout[i,:]

  Ein  = np.array([])
  Eout = np.array([])
  for l in lams:
    yin,zin   = getbinary(ydin,5), transform(xin)
    yout,zout = getbinary(ydout,5), transform(xout)
    w    = lm_ridge(zin,yin, lam=l)
    yhat = np.dot(zin, w)
    Ei   = np.sum(yhat*yin < 0) / (1.*zin.shape[0])
    yhat = np.dot(zout, w)
    Eo   = np.sum(yhat*yout < 0) / (1.*zout.shape[0])
    Ein  = np.append(Ein,  Ei) 
    Eout = np.append(Eout, Eo) 
  return({'Ein':Ein, 'Eout':Eout})


"""
Support Vector Machines
=======================

11. Consider the following training set generated from a target function f: X -> {-1,+1} 
    where X = R^2

x = np.array([ [1,0], [0,1], [0,-1], [-1,0], [0,2], [0,-2], [-2,0] ])
y = np.array([ -1,    -1,    -1,     1,      1,     1,      1      ])

  Transform this training set into another two-dimensional space Z

   z1 = x2^2 - 2x1 - 1
   z2 = x1^2 - 2x2 + 1

  Using geometry (not quadratic programming), what values of w (without w0)
  and b specify the separating plane w'z + b = 0 in the Z space that maximizes the margin?
  The values of w1,w2,b are:

    >>> (x,y) = f.q11data()
    >>> z = f.zspace(x)
    array([[-3,  2],
       [ 0, -1],
       [ 0,  3],
       [ 1,  2],
       [ 3, -3],
       [ 3,  5],
       [ 3,  5]])
    
    # svm says this
    >>> r = f.runsvm_lin(z,y); print(r['w'], r['b'])
    (array([[  1.99980800e+00,   6.40000000e-05]]), array([-1.00002133]))
    # 2, 0, -1  -> can be scaled to 1, 0, -0.5

    # plot the values to see what they look like
    >>> pl.scatter(z[:,0],z[:,1],c=y,cmap=pl.cm.Paired); pl.show()

    Looks like a vertical line will best separate the two classes.
    Vector w is orthogonal to the decision line, so w1 should be non-zero, w2 should be zero.

    [c] 1, 0, -0.5

12. Consider the same training set of the previous problem, but instead of explicitly transforming
    the input space X, apply the SVM algorithm with the kernel

      K(x,x') = (1 + x x')^2

    (which corresponds to a second-order polynomial transformation). Set up the expression for
    L(a1,...a7) and solve for the optimal a1,...,a7 (numerically, using a quadratic programing
    package). The number of support vectors you get is in what range?

    Compare the vanilla svm vs mysvm:

    >>> x
    array([[ 1,  0],
       [ 0,  1],
       [ 0, -1],
       [-1,  0],
       [ 0,  2],
       [ 0, -2],
       [-2,  0]])

    >>> f.runsvm_pol(x,y,C=10e6, Q=2)
    {'clf': SVC(C=10000000.0, cache_size=200, class_weight=None, coef0=1, degree=2,
      gamma=1, kernel='poly', max_iter=-1, probability=False,
      random_state=None, shrinking=True, tol=0.001, verbose=False), 'b': array([-1.66633088]), 'n_support': 5, 'Ein': 0.0}

    >>> f.mysvm(x,y)
    Optimization terminated successfully.    (Exit mode 0)
            Current function value: -1.40740740625
            Iterations: 6
            Function evaluations: 56
            Gradient evaluations: 6
    {'alpha': array([ -1.39651e-14, 6.34278e-01, 7.73125e-01, 8.88880e-01, 2.24543e-01, 2.93980e-01,  -3.89725e-14]), 'b': -1.000027129754987, 'w': array([ -8.88880e-01,  -2.71298e-05])}

    Looks like 5 non-zero support vectors by observing the 'alpha' values.

    [c] 4-5
"""

def q11data():
  x = np.array([ [1,0], [0,1], [0,-1], [-1,0], [0,2], [0,-2], [-2,0] ])
  y = np.array([ -1,    -1,    -1,     1,      1,     1,      1      ])
  return(x,y)

def zspace(x):
  z = np.apply_along_axis(lambda(x): np.array([x[1]**2 - 2*x[0] - 1, x[0]**2 - 2*x[1] + 1]), 1, x)
  return(z)

# SVC implements several kernels:
#  linear:     (x,x')
#  polynomial: (gamma*(x,x') + coef0)^degree
#  rbf:        exp(-gamma*||x,x'||^2)
#  sigmoid:    tanh(-gamma*(x,x') + coef0)
def runsvm_lin(x,y,C=1000000):
  clf = svm.SVC(kernel='linear', C=C)
  clf.fit(x, y)
  yhat = clf.predict(x)
  Ein = np.sum( y*yhat < 0 ) / (1.*y.size)
  return({'Ein':Ein, 'w':clf.coef_,'b':clf.intercept_,
          'clf':clf, 'n_support':clf.support_vectors_.shape[0]})

def runsvm_pol(x,y, C=0.01, Q=2):
  clf = svm.SVC(kernel='poly', C=C, degree=Q, gamma=1, coef0=1)
  clf.fit(x, y)
  yhat = clf.predict(x)
  Ein = np.sum( y*yhat < 0 ) / (1.*y.size)
  return({'Ein':Ein, 'b':clf.intercept_,
          'clf':clf, 'n_support':clf.support_vectors_.shape[0]})

def runsvm_rbf(x,y, C=0.01,gamma=1.):
  clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
  clf.fit(x, y)
  yhat = clf.predict(x)
  Ein = np.sum( y*yhat < 0 ) / (1.*y.size)
  return({'Ein':Ein, 'b':clf.intercept_,
          'clf':clf, 'n_support':clf.support_vectors_.shape[0]})

def mysvm_lin(xmat,y,w=None,C=1e6):
  """
  Solve the problem:
    L(a) = sum_{n=1}^{N} a_n - 1/2 sum_{n=1}^{N} sum_{m=1}^{N} y_n y_m a_n a_m x_n' x_m

    min(a) 1/2 a' Q a + (-1') a
    s.t.  y'a = 0
          0 <= a <= C

    Use the scipy.optimize function called fmin_slsqp:
      http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

  NOTE: SVM doesn't require you to feed in the x0 for the intercept.

  Hmm.. checking this against the libsvm results in different answers!
  >>> f.mysvm_lin(x,y, C=1e6)['w']
  array([-64.30942,  79.1555 ])
  >>> f.runsvm_lin(x,y, C=1e6)['w']
  array([[-317.66406,  383.00197]])
  """
  # xN = (1+np.dot(xmat, xmat.transpose()))**2 # polynomial kernel
  xN = np.dot(xmat, xmat.transpose()) # N x N size matrix: this is for linear svm
  yN = np.outer(y, y)                 # because y is just a vector, you do outer product
  Q = yN * xN                         # itemwise mult
  # objfunc = lambda x: 0.5 * np.dot(np.dot(x.transpose(), Q), x) - np.dot(np.ones(len(x)),x)
  objfunc = lambda x: 0.5 * np.dot(np.dot(x.transpose(), Q), x) - np.sum(x)
  eqcon   = lambda x: np.dot(y,x)     # equality constraint, single scalar value
  ineqcon = lambda x: np.array(x)     # inequality constraint, vector constraints
  bounds  = [(0.0, C) for i in range(xmat.shape[0])]
  x0 = np.array([rn.uniform(0.,1./y.size) for i in range(xmat.shape[0])]) # random starting point
  alpha = sci.fmin_slsqp(objfunc, x0, eqcons=[eqcon], bounds=bounds, iter=1000, iprint=0)
  # alpha = sci.fmin_slsqp(objfunc, x0, eqcons=[eqcon], f_ieqcons=ineqcon)
  w = np.apply_along_axis(lambda x: np.sum(alpha * y * x), 0, xmat)
  iv = np.where(alpha > 0.001)[0] 
  i  = iv[0] # first support vector where alpha is greater than zero
  b = 1.0/y[i] - np.dot(w, xmat[i])
  yhat = np.dot(xmat, w) + b
  Ein = np.sum(yhat*y<0)/(1.*y.size)
  return({'alpha':alpha, 'w':w, 'b':b, 'Ein':Ein, 'n_support':len(iv)})


def mysvm_rbf(xmat,y,w=None, C=1e16, gamma=1.0):
  """
  RBF kernel needs NxN matrix of pairwise euclidean distances:
  http://stats.stackexchange.com/questions/15798/how-to-calculate-a-gaussian-kernel-effectively-in-numpy
  The resulting matrix is symmetric, with the diagonals being 0 in euclidean distances.
  The rest of the quadratic programming stays the same.
  Here includes the soft-margin constraint:
      0 <= alpha_n <= C
  """
  euc = squareform(pdist(xmat,'euclidean'))**2  # gives back NxN matrix
  xN  = np.exp(-gamma*euc)
  yN  = np.outer(y, y)                 # because y is just a vector, you do outer product
  Q   = yN * xN                        # itemwise mult
  objfunc = lambda x: 0.5 * np.dot(np.dot(x.transpose(), Q), x) - np.dot(np.ones(len(x)),x)
  eqcon   = lambda x: np.dot(y,x)     # equality constraint, single scalar value
  ineqcon = lambda x: np.array(x)     # inequality constraint, vector constraints
  bounds  = [(0.0, C) for i in range(xmat.shape[0])]
  x0 = np.array([rn.uniform(0.,1.) for i in range(xmat.shape[0])]) # random starting point
  alpha = sci.fmin_slsqp(objfunc, x0, eqcons=[eqcon], bounds=bounds, iter=1000)
  w = np.apply_along_axis(lambda x: np.sum(alpha * y * x), 0, xmat)
  iv = np.where(alpha > 0.001)[0] 
  i  = iv[0] 
  b = 1.0/y[i] - np.dot(w, xmat[i])
  yhat = np.dot(xmat, w) + b
  Ein = np.sum(yhat*y<0)/(1.*y.size)
  return({'alpha':alpha, 'w':w, 'b':b, 'Ein':Ein, 'n_support':len(iv)})

"""
Radial Basis Functions
======================
NOTE:
  * Regular RBF is classification algorithm that combines k-means clustering + linear regression
  * Kernel RBF is the hard-margin SVM using RBF as the kernel 

We experiment with the RBF model, both in regular form (Lloyd + pseudo-inverse) with K centers:

  sign( sum^K_{k=1} w_k exp(-gamma ||x - uk||^2) + b)

(notice that there is a bias term), and in kernel form (using the RBF kernel in hard-margin SVM):

  sign( sum{an>0} an*yn exp(-gamma ||x - xn||^2) + b)
  
The input space is X = [-1,1] x [-1,1] with uniform probability distribution, and the target is
  
  f(x) = sign(x2 - x1 + 0.25 sin(pi*x1))

which is slightly nonlinear in the X space. In each run, generate 100 training points at random
using this target, and apply both forms of RBF to these training points. 

Here are some guidelines:

* Repeat the experiment for as many runs as needed to get the answer to be stable
  (statistically away from flipping to the closest competing answer).
* In case a data set is not linearly separable in the 'Z space' by the RBF kernel using hard-margin
  SVM, discard the run but keep track of how often this happens.
* When you use Lloyd's algorithm, initialize the centers to random points in X and iterate until
  there is no change from iteration to iteration. If a cluster becomes empty, discard the run and 
  repeat. 

K-means Clustering
------------------
Some useful links for k-means Lloyd's algorithm:
  http://www.lucifero.us/toolbox/kmeans/
  https://gist.github.com/larsmans/4952848

Core of the Lloyd's algorithm:

  sum_{k=1}^{K} sum_{x in Sk} ||xn - uk||^2  w.r.t uk, Sk 

  * uk <- 1/|Sk| sum_{xn in Sk} xn

    compute the mean of the "center" 
  
  * Sk <- {xn : ||xn - uk|| <= all ||xn - ul||}

    group a sample into a cluster that gives the smallest distance

    for _ in xrange(n_iter):
        # assign each sample to the cluster that gives the minimum distance
        for i, x in enumerate(xs):
            cluster[i] = min(xrange(k), key=lambda j: dist(xs[i], centers[j]))
        # for each center, compute the mean by samples that are grouped within those centers
        for j, c in enumerate(centers):
            members = (x for i, x in enumerate(xs) if cluster[i] == j)
            centers[j] = mean(members, l)


13. For gamma = 1.5, how often do you get a dataset that is not linearly separable by the
    RBF kernel (using hard-margin SVM). Hint: Run the usual hard-margin SVM, then check that
    the solution has Ein = 0.

    * hard margin always try to separate the points and sometimes does not succeed
    * SVM of Q-order always succeed to separate N points if Q > N
    * SVM with RBF kernel has infinite dimensions (Q = inf)
    * Thus, Ein should be zero in all cases

    [a] <=5% of the time

14. If we use K=9 for regular RBF and take gamma = 1.5, how often does the kernel form beat
    the regular form (after discarding the runs mentioned above, if any) in terms of Eout?
  
    * Here, you first use K-means clustering to find 9 "centers"
    * The 9 centers become the features into the linear regression problem

    >>> f.q14(200)['kerwins'] 
    0.78

    [e] >75% of the time

15. If we use K=12 for regular RBF and take gamma = 1.5, how often does the kernel form beat
    the regular form (after discarding the runs mentioned above, if any) in terms of Eout? 

    >>> f.q15(200,12)
    0.66

    [d] >60% but <=90% of the time

16. Now we focus on regular RBF only, with gamma=1.5. If we go from K=9 clusters to K=12 clusters,
    which of the following 5 vases happens most often in your runs? 
    
    >>> e = f.q16(500,ks=[9,12]) 
    >>> f.checkcount(e)
    {'Ein_up_Eout_dn': 4.79,
     'Ein_dn_Eout_dn': 45.79, <<-- both go down
     'Ein_up_Eout_up': 5.40,
     'Ein_dn_Eout_up': 13.0}

    [a] Ein goes down but Eout goes up
    [b] Ein goes up but Eout goes down
    [c] Both Ein and Eout go up
  X [d] Both Ein and Eout go down
    [e] There is no change

17. For regular RBF with K=9, if we go from gamma=1.5 to gamma=2, which of the following 5 cases
    happens most often in your runs? 

    >>> e = f.q17(runs=500,gammas=[1.5,2.0])
    >>> f.checkcount(e)
    {'Ein_up_Eout_dn': 10.19,
     'Ein_dn_Eout_dn': 13.4,
     'Ein_up_Eout_up': 32.60, <<-- both go up
     'Ein_dn_Eout_up': 12.0}

    # average might not be representative of the counts, but without outliers, it's valid
    # this is interesting when you vary the gamma from 1.0 to 2.0 in 0.1 increments
    >>> g = f.q17() # gammas=[1.5,1.6,1.7,1.8,1.9,2.0]
    >>> np.mean(g['Ein'],axis=0)
    array([ 0.03435,  0.03585,  0.0359 ,  0.03665,  0.03875,  0.03975])
    >>> np.mean(g['Eout'],axis=0)
    array([ 0.05555,  0.0578 ,  0.05685,  0.06125,  0.06435,  0.0693 ])


    [a] Ein goes down but Eout goes up
    [b] Ein goes up but Eout goes down
  X [c] Both Ein and Eout go up
    [d] Both Ein and Eout go down
    [e] There is no change


18. What is the percentage of time that regular RBF achieves Ein=0 with K=9 and gamma=1.5?

    >>> g = f.q17(500,k=9,gammas=[1.5])
    >>> np.sum(g['Ein'][:,0]==0.0) / (1.*g['Ein'].shape[0])
    0.043999999999999997

    [a] <=10% of the time
"""

def gendata(n=100):
  x = np.array([[rn.uniform(-1,1),rn.uniform(-1,1)] for i in xrange(n)])
  y = np.apply_along_axis(lambda x: np.sign(x[1]-x[0]+0.25*np.sin(np.pi*x[0])), 1, x)
  return(x,y)

def minpos(x1, mu):
  e = np.sqrt(np.sum((x1 - mu)**2, axis=1)) # euclidean distances to all centers from single point
  return(np.argmin(e)) # id of the center with minimum distance

# Lloyd's algorithm using numpy implementation: flexible for d dimensions
#  (x,y) = f.gendata(100)
#  f.kmeans(x)
def kmeans(x,k=9):
  mu = np.array([[rn.uniform(-1,1),rn.uniform(-1,1)] for i in xrange(k)]) # random centers
  i = 0
  lastgroup = np.ones(x.shape[0])
  group     = np.zeros(x.shape[0])
  while not np.all(lastgroup==group) and i<1000:
    lastgroup = group
    group = np.apply_along_axis(minpos, 1, x, mu) # determine which cluster each x belongs to   
    ug = np.unique(group).tolist()  # how many unique groups?          
    for g in ug:
      # xsub = x[group==g]; print("group %d has %d members" % (g, len(xsub)))
      mu[g] = np.mean(x[group==g], axis=0) # update the centers
    i += 1

  if len(np.unique(group))!=k:
    print("entering kmeans again because one or more clusters do not have members")
    (x,y) = gendata(x.shape[0]) # regen
    kmeans(x,k) # do it again if any cluster with no members found
  return({'group':group, 'mu':mu, 'iter':i})

# for verification
# >>> (x,y)=f.gendata(100);f.showkmeans(x,f.kmeans(x)['mu'])
def showkmeans(x,mu):
  fig = pl.figure()
  ax = fig.add_subplot(111)
  ax.scatter(x[:,0],x[:,1], c='b')
  ax.scatter(mu[:,0],mu[:,1], c='r', marker='o', s=50)
  fig.show()

def rbf1(x,mu0,gamma=1.5):
  return( np.exp(-gamma*np.sum((x-mu0)**2)) )

def rbf(x,mu,gamma=1.5):
  z = np.empty((x.shape[0], 1+mu.shape[0]))  # N x (K+1), extra 1 for intercept
  z[:,0] = np.ones(x.shape[0])
  for i in xrange(mu.shape[0]):
    z[:,i+1] = np.apply_along_axis(rbf1, 1, x, mu[i],gamma)
  return(z)

# lecture 16, slide 14/20
def runreg_rbf(x,y,gamma=1.5,k=9):
  r = kmeans(x,k)
  mu = r['mu']
  z = rbf(x,mu,gamma)
  # create the Phi matrix
  w = lm_ridge(z,y,lam=0.0) # standard linear regression
  yhat = np.dot(z,w)
  Ein = np.sum(yhat*y<0)/(1.*y.size)
  return({'Ein':Ein, 'w':w, 'mu':mu}) 

def q13(runs=1000):
  Ein = np.empty(runs)
  nsv = np.empty(runs)
  for i in xrange(runs): 
    (x,y) = gendata(100)
    r = runsvm_rbf(x,y, C=10e6, gamma=1.5)
    Ein[i] = r['Ein']
    nsv[i] = r['n_support']
  return({'num_err':np.sum(Ein != 0)/(1.*Ein.size), 'avg_nsv':np.mean(nsv)})

def q14(runs=100,k=9):
  Eout_reg = np.empty(runs)
  Eout_ker = np.empty(runs)
  for i in xrange(runs):
    (x,y)       = gendata(100)
    (xout,yout) = gendata(100)
    # regular RBF Eout
    r = runreg_rbf(x,y, gamma=1.5, k=k)
    mu,w = r['mu'],r['w']
    z    = rbf(xout,mu) # recalc the mu or use the same ones from training??
    yhat = np.dot(z,w) 
    Eout_reg[i] = np.sum(yhat*yout<0)/(1.*yout.size)
    # kernel RBF Eout
    r = runsvm_rbf(x,y, C=1e6, gamma=1.5)
    clf,nsv = r['clf'],r['n_support']
    yhat = clf.predict(xout)
    Eout_ker[i] = np.sum(yhat*yout<0)/(1.*yout.size)
  return({'Eout_reg':Eout_reg, 'Eout_ker':Eout_ker, 'kerwins':np.sum(Eout_ker<Eout_reg)/(1.*runs)})

def q16(runs=100,ks=[9,10,11,12]):
  Ein = np.empty((runs,len(ks)))
  Eout = np.empty((runs,len(ks)))
  for i in xrange(runs):
    (xin,yin)   = gendata(100)
    (xout,yout) = gendata(200)
    for j in xrange(len(ks)):
      k = ks[j]
      r = runreg_rbf(xin,yin, gamma=1.5, k=k)
      mu,w = r['mu'],r['w']
      z    = rbf(xout,mu) # recalc the mu or use the same ones from training??
      yhat = np.dot(z,w) 
      Ein[i,j] = r['Ein']
      Eout[i,j] = np.sum(yhat*yout<0)/(1.*yout.size)
  return({'Ein':Ein,'Eout':Eout})

def q17(runs=100,k=9,gammas=[1.5,1.6,1.7,1.8,1.9,2.0]):
  Ein = np.empty((runs,len(gammas)))
  Eout = np.empty((runs,len(gammas)))
  for i in xrange(runs):
    (xin,yin)   = gendata(100)
    (xout,yout) = gendata(200)
    for j in xrange(len(gammas)):
      gamma = gammas[j]
      r = runreg_rbf(xin,yin, gamma=gamma, k=k)
      mu,w = r['mu'],r['w']
      z    = rbf(xout,mu) # recalc the mu or use the same ones from training??
      yhat = np.dot(z,w) 
      Ein[i,j] = r['Ein']
      Eout[i,j] = np.sum(yhat*yout<0)/(1.*yout.size)
  return({'Ein':Ein,'Eout':Eout})

def checkcount(r):
  Ein  = r['Ein']
  Eout = r['Eout']
  n = 1.*Ein.shape[0]
  d = dict()
  d['Ein_dn_Eout_up'] = 100.*np.sum( np.logical_and(Ein[:,0]>Ein[:,1], Eout[:,0]<Eout[:,1]) ) / n
  d['Ein_up_Eout_dn'] = 100.*np.sum( np.logical_and(Ein[:,0]<Ein[:,1], Eout[:,0]>Eout[:,1]) ) / n
  d['Ein_up_Eout_up'] = 100.*np.sum( np.logical_and(Ein[:,0]<Ein[:,1], Eout[:,0]<Eout[:,1]) ) / n
  d['Ein_dn_Eout_dn'] = 100.*np.sum( np.logical_and(Ein[:,0]>Ein[:,1], Eout[:,0]>Eout[:,1]) ) / n
  return(d)

"""
Bayesian Priors
===============

19. Let f in [0,1] be the unknown probability of getting a heart attack for people in a certain
    population. notice that f is just a constant, not a function, for simplicity. We want to model
    f using a hypothesis h in [0,1]. Before we see any data, we assume that P(h=f) is uniform
    over h in [0,1] (the prior). We pick one sample from the population, and it turns out that they
    had a heart attack. Which of the following is true about the posterior probability that h = f
    given this sample point?

               P(D|h=f) P(h=f)
    P(h=f|D) = ---------------  proportional to    P(D|h=f) P(h=f)
                    P(D)

    lecture 18, slide 10/23: "If we knew the prior"
    
      ... we could compute P(h=f|D) for every h in H
      * we can find the most probable h given the data
      * we can derive Expectation[h(x)] for every X
      * we can derive the error bar for every X
      * we can derive everthing in a principled way

    Here, 

      P(h=f)   is unif[0,1]
      P(D|h=f) given D, we can compute this qty? is this Ein? (lecture 18, slide 8)   

    NOTE: read the discussion forum on this:
    https://courses.edx.org/courses/CaltechX/CS1156x/Fall2013/discussion/forum/i4x-Caltech-CS156-course-Fall2013_Week9/threads/5293b0cd45d9975fff0000e3 

    [a] The posterior is uniform over [0,1]
  X [b] The posterior increases linearly over [0,1]
    [c] The posterior increases nonlinearly over [0,1]
    [d] The posterior is a delta function at 1 (implying f has to be 1)
    [e] The posterior cannot be evaluated based on the given information


Aggregation
===========

20. Given two learned hypotheses g1 and g2, we construct the aggregate hypothesis g giving by 
    g(x) = 1/2(g1(x) + g2(x)) for all x in X. If we use mean-squared error, which of the following
    statements is true?

    Should do this on paper or try to write a simple program for it.
    Check: Cauchy inequality. Someone mentioned it in the discussion forums.

    >>> f.q20()
    >>> np.mean(e,axis=0) # Eout avg for : g1, g2, g
    array([ 0.0571,  0.0539,  0.0539])

    NOTE: look at the proof on the image of my notes. But can't believe I got this one wrong!

    [a] Eout(g) cannot be worse than Eout(g1)
      : FALSE, there's no particular information about Eout(g1) over Eout(g2)
    [b] Eout(g) cannot be worse than the smaller of Eout(g1) and Eout(g2)
      : FALSE, despite from testing, g always seems to be better than smaller of g1 or g2
  X [c] Eout(g) cannot be worse than the average of Eout(g1) and Eout(g2)
      : worked this out on paper. The result gives:
          E[g] = 0.5 ( E[g1] + E[g2] - 0.5 E[g1-g2] )
    [d] Eout(g) has to be between Eout(g1) and Eout(g2) (including the end values of that interval)
    [e] None of the above
"""

def runreg1(x,y):
  w = lm_ridge(x,y,lam=0.0) # standard linear regression
  yhat = np.dot(x,w)
  Ein = np.sum(yhat*y<0)/(1.*y.size)
  return({'Ein':Ein, 'w':w}) 

def addintercept(x):
  return(np.apply_along_axis(lambda x: np.append(1.,x), 1,x))

# g1 = linear regression
# g2 = svm
def q20(runs=100):
  n = 100
  Ein  = np.empty((runs,3))   
  Eout = np.empty((runs,3))   
  for i in xrange(runs):
    (xin,yin)   = gendata(100)
    (xout,yout) = gendata(100)
    r1 = runreg(addintercept(xin),yin) # standard linear regression
    r2 = runsvm_lin(xin,yin) # svm
    Ein[i,0:2] = np.array([r1['Ein'],r2['Ein']]) 
    w1 = r1['w']
    w2 = np.append(r2['b'],r2['w'])
    print(w1,w2)
    yhat = np.dot(addintercept(xout),r1['w'])
    Eout[i,0] = np.sum(yhat*yout<0)/(1.*yout.size)
    yhat = r2['clf'].predict(xout)
    Eout[i,1] = np.sum(yhat*yout<0)/(1.*yout.size)
    # g = 0.5 ( g1 + g2 )
    w = 0.5*(r1['w'] + np.append(r2['b'],r2['w']))
    yhat = np.dot(addintercept(xout),w)
    Eout[i,2] = np.sum(yhat*yout<0)/(1.*yout.size)
  return(Eout)
