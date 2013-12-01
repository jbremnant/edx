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
    [d] H is the logistic regression model - em.. same as [c]?
  X [e] none of the above?
"""

"""
Overfitting
===========
3. Which of the following statements is false? (here, FALSE!)

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
    [d] We can always determine if there is overfitting by comparing the values of
        (Eout - Ein)
        : Eout <= Ein + Omega(N,d), so yes
  X [e] We cannot determine overfitting based on one hypothesis only
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
        : refer to svm soft,hard constraints 
  X [b] translated into augmented error
        : hmm..
    [c] determined from the value of the VC dimension
    
    [d] used to decrease both Ein and Eout
        : NO, appropriate regularization will bolster Eout
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

    >>> f.zspace(x)
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

    >>> f.runsvm(x,y,C=10e6, Q=2)
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

def runsvm(x,y, C=0.01, Q=2):
  clf = svm.SVC(kernel='poly', C=C, degree=Q, gamma=1, coef0=1)
  clf.fit(x, y)
  yhat = clf.predict(x)
  Ein = np.sum( y*yhat < 0 ) / (1.*y.size)
  return({'Ein':Ein, 'b':clf.intercept_,
          'clf':clf, 'n_support':clf.support_vectors_.shape[0]})

def runsvm_rbf(x,y, C=0.01):
  clf = svm.SVC(kernel='rbf', C=C, gamma=1.)
  clf.fit(x, y)
  yhat = clf.predict(x)
  Ein = np.sum( y*yhat < 0 ) / (1.*y.size)
  return({'Ein':Ein, 'b':clf.intercept_,
          'clf':clf, 'n_support':clf.support_vectors_.shape[0]})

def mysvm(xmat,y,w=None):
  """
  Solve the problem:
    L(a) = sum_{n=1}^{N} a_n - 1/2 sum_{n=1}^{N} sum_{m=1}^{N} y_n y_m a_n a_m x_n' x_m

    min(a) 1/2 a' Q a + (-1') a
    s.t.  y'a = 0
          0 <= a <= inf

    Use the scipy.optimize function called fmin_slsqp:
      http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

  NOTE: SVM doesn't require you to feed in the x0 for the intercept.
  """
  # xN = np.dot(xmat, xmat.transpose()) # N x N size matrix: this is for linear svm
  xN = (1+np.dot(xmat, xmat.transpose()))**2 # polynomial kernel
  yN = np.outer(y, y)                 # because y is just a vector, you do outer product
  Q = yN * xN                         # itemwise mult
  objfunc = lambda x: 0.5 * np.dot(np.dot(x.transpose(), Q), x) - np.dot(np.ones(len(x)),x)
  eqcon   = lambda x: np.dot(y,x)     # equality constraint, single scalar value
  ineqcon = lambda x: np.array(x)     # inequality constraint, vector constraints
  bounds  = [(0.0, 1e16) for i in range(xmat.shape[0])]
  x0 = np.array([rn.uniform(0,1) for i in range(xmat.shape[0])]) # random starting point
  # alpha = sci.fmin_slsqp(func, x0, eqcons=[eqcon], bounds=bounds)
  alpha = sci.fmin_slsqp(objfunc, x0, eqcons=[eqcon], f_ieqcons=ineqcon)

  w = np.apply_along_axis(lambda x: np.sum(alpha * y * x), 0, xmat)
  i = np.where(alpha > 0.001)[0][1] # first support vector where alpha is greater than zero
  b = 1.0/y[i] - np.dot(w, xmat[i])
  return({'alpha':alpha, 'w':w, 'b':b})
