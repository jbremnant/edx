import sys
import math
import re
import random as rn
import numpy as np
from sklearn import svm
import scipy.optimize as sci

"""
import sys
sys.path.append('/home/jbkim/Development/edx/cs1156x/week8')
sys.path.append('/home/jbkim/Development/edx/cs1156x/week7')
import hw8 as h
reload(h)
"""

"""
Primal versus Dual Problem
==========================

1. Recall that N is the size of the dat set and d is the dimensionality of the input space.
   The original forumulation of the hard-margin SVM problem (minimize 1/2 w'w subject
   to the inequality constraints), without going through the Lagrangian dual problem, is

    The original formulation (primal problem) is minimizing the weight vector w, without
    the intercept w- term.

      min 1/2 w'w
      s.t.  y_n (w' x_n + b) >= 1  for n = 1,2,...,N

    [c] a quadratic programming problem with d variables
"""


"""
Support Vector Machines With Soft Margins
=========================================

Download US Postal service zip code dataset with extracted features of symmetry and 
intensity for training and testing:

  http://www.amlbook.com/data/zip/features.train
  http://www.amlbook.com/data/zip/features.test

Recommended packages in libsvm:

  http://www.csie.ntu.edu.tw/~cjlin/libsvm/

Scikit-learn implements svm using this library already.

  http://scikit-learn.org/stable/modules/svm.html#kernel-functions

Implement SVM with soft margin on the above zip-code data set by solving

  min(a)  1/2 sum_{n=1}^{N} sum_{m=1}^{N} a_n a_m y_n y_m K(x_n,x_m - sum_{n=1}^{N} a_n
  s.t. sum_{n=1}^{N} y_n a_n = 0
       0 <= a_n <= C   n = 1,...,N

Wehn evaluating Ein and Eout of the resulting classifier, use binary classification error.

Polynomial Kernels
------------------

Consider the polynomial kernel K(x_n, x_m) = (1+ x_n' x_m)^Q,  where Q is the degree of the
polynomail.

2. With C = 0.01 and Q = 2, which of the following classifiers has the highest Ein?
    >>> h.q2()
    {'Ein0': 0.10588, 'Ein8': 0.07433, 'Ein2': 0.10026, 'Ein4': 0.08942, 'Ein6': 0.09107}

    [a] 0 versus all

3. With C = 0.01 and Q = 2, which of the following classifiers has the lowest Ein?
    >>> h.q3()
    {'Ein1': 0.0144, 'Ein3': 0.0902, 'Ein9': 0.0883, 'Ein5': 0.0762, 'Ein7': 0.0884}

    [a] 1 versus all

4. Comparing the two classifiers from Prob 2,3, which of the following values is the closest
   to the difference between the number of support vectors of these two classifiers?
    >>> h.runsvm(x,y,choice=0)['n_support'] - h.runsvm(x,y,choice=1)['n_support']
    1793

    [c] 1800

5. Consider the 1 versus 5 classifier with Q = 2 and C in {0.001,0.01,0.1,1}
   Which of the following statements is correct?
    # C : [0.001,0.01,0.1,1.] 
    >>> h.q5()
    {'Eout': [0.01650, 0.01886, 0.01886, 0.01886], 'nsv': [76, 34, 24, 24],
     'Ein':  [0.00448, 0.00448, 0.00448, 0.00320]}

    [a] The number of support vectors goes down when C goes up  (FALSE) 0.1 -> 1.0 no change
    [b] The number of support vectors goes up when C goes up    (FALSE)
    [c] Eout goes down when C goes up                           (FALSE)
  ->[d] Maximum C achieves the lowest Ein                       (TRUE)
    [e] None of the above

6. In the 1 versus 5 classifier, comparing Q = 2 with Q = 5, which of the following
   statments is correct?
    # C : [0.0001,0.001,0.01,0.1,1.0]
    >>> h.q6()
    {'Q2':
      {'Eout': [0.01650, 0.01650, 0.01886, 0.01886, 0.01886], 'nsv': [236, 76, 34, 24, 24],
       'Ein':  [0.00896, 0.00448, 0.00448, 0.00448, 0.00320]},
     'Q5':
      {'Eout': [0.01886, 0.02122, 0.02122, 0.01886, 0.02122], 'nsv': [26, 25, 23, 25, 21],
       'Ein':  [0.00448, 0.00448, 0.00384, 0.00320, 0.00320]}}
    
    [a] When C = 0.0001, Ein is higher at Q=5                          (FALSE)
  ->[b] When C = 0.001, the number of support vectors is lower at Q=5  (TRUE)
    [c] When C = 0.01, Ein is higher at Q=5                            (FALSE)
    [d] When C = 1, Eout is lower at Q=5                               (FALSE) 
    [e] None of the above
"""

# (x,y) = h.readdata()
def readdata(file="/home/jbkim/Development/edx/cs1156x/week8/features.train"):
  d = np.genfromtxt(file, dtype=float)
  return(np.apply_along_axis(lambda(x): x[1:3],1,d), np.apply_along_axis(lambda(x): x[0],1,d))

def getbinary(y,digit=0):
  z=np.ones(len(y))
  z[y!=digit] = -1
  return(z)

def poly_kernel(x,y):
  return (1. + np.dot(x, y.T))**2

# SVC implements several kernels:
#  linear:     (x,x')
#  polynomial: (gamma*(x,x') + coef0)^degree
#  rbf:        exp(-gamma*||x,x'||^2)
#  sigmoid:    tanh(-gamma*(x,x') + coef0)
def runsvm(x,ydigit, choice=0, C=0.01, Q=2):
  y = getbinary(ydigit, choice)
  # linear kernel would look like this
  # clf = svm.SVC(kernel='linear', C=C)
  # clf.fit(x, y2) 
  # also test the custom kernel to verify the logic of kernel trick
  # clf = svm.SVC(kernel=poly_kernel, C=C, degree=Q, gamma=1, coef0=1)
  clf = svm.SVC(kernel='poly', C=C, degree=Q, gamma=1, coef0=1)
  clf.fit(x, y) 
  yhat = clf.predict(x)
  Ein = np.sum( y*yhat < 0 ) / (1.*y.size)
  return({'Ein':Ein, 'b':clf.intercept_,
          'clf':clf, 'n_support':clf.support_vectors_.shape[0]})

def q2():
  (x,y) = readdata("/home/jbkim/Development/edx/cs1156x/week8/features.train")
  r0 = runsvm(x,y, choice=0, C=0.01, Q=2)
  r2 = runsvm(x,y, choice=2, C=0.01, Q=2)
  r4 = runsvm(x,y, choice=4, C=0.01, Q=2)
  r6 = runsvm(x,y, choice=6, C=0.01, Q=2)
  r8 = runsvm(x,y, choice=8, C=0.01, Q=2)
  return({
    'Ein0':r0['Ein'],
    'Ein2':r2['Ein'],
    'Ein4':r4['Ein'],
    'Ein6':r6['Ein'],
    'Ein8':r8['Ein'], })

def q3():
  (x,y) = readdata("/home/jbkim/Development/edx/cs1156x/week8/features.train")
  r1 = runsvm(x,y, choice=1, C=0.01, Q=2)
  r3 = runsvm(x,y, choice=3, C=0.01, Q=2)
  r5 = runsvm(x,y, choice=5, C=0.01, Q=2)
  r7 = runsvm(x,y, choice=7, C=0.01, Q=2)
  r9 = runsvm(x,y, choice=9, C=0.01, Q=2)
  return({
    'Ein1':r1['Ein'],
    'Ein3':r3['Ein'],
    'Ein5':r5['Ein'],
    'Ein7':r7['Ein'],
    'Ein9':r9['Ein'], })

def q4():
  (x,y) = readdata("/home/jbkim/Development/edx/cs1156x/week8/features.train")
  return(h.runsvm(x,y,choice=0)['n_support'] - h.runsvm(x,y,choice=1)['n_support'])

def q5(Cs=[0.001,0.01,0.1,1.0], Q=2):
  (x,y)       = readdata("/home/jbkim/Development/edx/cs1156x/week8/features.train")
  idx = np.logical_or(y==1,y==5)
  x = x[idx,:]
  y = y[idx]

  (xout,yout) = readdata("/home/jbkim/Development/edx/cs1156x/week8/features.test")
  idx = np.logical_or(yout==1,yout==5)
  xout = xout[idx,:]
  yout = yout[idx]
  ybin = getbinary(yout,1)

  Ein  = []
  Eout = []
  nsv  = []
  for C in Cs:
    r = runsvm(x,y, choice=1, C=C, Q=Q)
    Ein.append(r['Ein'])
    nsv.append(r['n_support'])
    clf  = r['clf']
    yhat = clf.predict(xout)
    Eout.append( np.sum( ybin*yhat < 0 ) / (1.*ybin.size) )
  return({'Ein':Ein,'Eout':Eout,'nsv':nsv})

def q6():
  r2 = q5(Cs=[0.0001,0.001,0.01,0.1,1.0], Q=2)
  r5 = q5(Cs=[0.0001,0.001,0.01,0.1,1.0], Q=5)
  return({'Q2':r2, 'Q5':r5})


"""
Cross Validation
================
In the next two problems, we will experiment with 10-fold cross validation for the polynomial
kernel. Because Ecv is a random variable that depends on the random partition of the data, we 
will try 100 runs with different partitions, and base our answer on the number of runs that lead
to a particular choice.

7. Consider the 1 versus

"""
