import sys
import math
import re
import random as rn
import numpy as np
from sklearn import svm
from sklearn import cross_validation
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

    [c] a quadratic programming problem with d variables  WRONG
    [d] d + 1 CORRECT

  Kidding me? I was supposed to count "b" too!!!??
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

def getbinary(y,choice=0):
  z=np.ones(len(y))
  z[y!=choice] = -1
  return(z)

def poly_kernel(x,y):
  return (1. + np.dot(x, y.T))**2

# SVC implements several kernels:
#  linear:     (x,x')
#  polynomial: (gamma*(x,x') + coef0)^degree
#  rbf:        exp(-gamma*||x,x'||^2)
#  sigmoid:    tanh(-gamma*(x,x') + coef0)
def runsvm(x,y, C=0.01, Q=2):
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
  r0 = runsvm(x,getbinary(y,choice=0), C=0.01, Q=2)
  r2 = runsvm(x,getbinary(y,choice=2), C=0.01, Q=2)
  r4 = runsvm(x,getbinary(y,choice=4), C=0.01, Q=2)
  r6 = runsvm(x,getbinary(y,choice=6), C=0.01, Q=2)
  r8 = runsvm(x,getbinary(y,choice=8), C=0.01, Q=2)
  return({
    'Ein0':r0['Ein'],
    'Ein2':r2['Ein'],
    'Ein4':r4['Ein'],
    'Ein6':r6['Ein'],
    'Ein8':r8['Ein'], })

def q3():
  (x,y) = readdata("/home/jbkim/Development/edx/cs1156x/week8/features.train")
  r1 = runsvm(x,getbinary(y, choice=1), C=0.01, Q=2)
  r3 = runsvm(x,getbinary(y, choice=3), C=0.01, Q=2)
  r5 = runsvm(x,getbinary(y, choice=5), C=0.01, Q=2)
  r7 = runsvm(x,getbinary(y, choice=7), C=0.01, Q=2)
  r9 = runsvm(x,getbinary(y, choice=9), C=0.01, Q=2)
  return({
    'Ein1':r1['Ein'],
    'Ein3':r3['Ein'],
    'Ein5':r5['Ein'],
    'Ein7':r7['Ein'],
    'Ein9':r9['Ein'], })

def q4():
  (x,y) = readdata("/home/jbkim/Development/edx/cs1156x/week8/features.train")
  return(runsvm(x,getbinary(y,choice=0))['n_support'] - 
         runsvm(x,getbinary(y,choice=1))['n_support'])

def q5(Cs=[0.001,0.01,0.1,1.0], Q=2):
  (x,y)       = readdata("/home/jbkim/Development/edx/cs1156x/week8/features.train")
  idx = np.logical_or(y==1,y==5)
  x   = x[idx,:]
  y   = getbinary(y[idx],1)
  (xout,yout) = readdata("/home/jbkim/Development/edx/cs1156x/week8/features.test")
  idx  = np.logical_or(yout==1,yout==5)
  xout = xout[idx,:]
  yout = getbinary(yout[idx],1)

  Ein  = []
  Eout = []
  nsv  = []
  for C in Cs:
    r = runsvm(x,y, C=C, Q=Q)
    Ein.append(r['Ein'])
    nsv.append(r['n_support'])
    clf  = r['clf']
    yhat = clf.predict(xout)
    Eout.append( np.sum( yout*yhat < 0 ) / (1.*yout.size) )
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

7. Consider the 1 versus 5 classifier with Q = 2. We use Ecv to select
   C in [0.0001,0.001,0.01,0.1,1]. If there is a tie in Ecv, select the smaller C. Within the 100
   random runs, which of the following statements is correct?

    # hmm.. randomness makes these results rather sensitive. Doing 500 iterations
    >>> h.q7()
    {'wins': [(0.0001, 0), (0.001, 22), (0.01, 33), (0.1, 23), (1.0, 22)],
     'Ecv':  [(0.0001, 0.005631),
              (0.001, 0.006023),
              (0.01, 0.005894),
              (0.1, 0.005639),
              (1.0, 0.0058949)]}

    
    [c] C = 0.01 is selected the most often  X Wrong!
    
    Drats!! this is stupid. I think the answer is [b]. On simulations, it was
    always either [b] or [c]

8. Again, consider the 1 versus 5 classifier with Q = 2. For the winning selection in the
   previous problem, the average value of Ecv over the 100 runs is closest to

    [c] 0.005

"""

# k-fold cross validation version of SVM
def runsvm_cv(x,y,C=0.0001,Q=2,folds=10):
  kf = cross_validation.KFold(len(y), n_folds=folds, shuffle=True)
  Ecv = np.array([])
  Ein = np.array([])
  nsv = np.array([])
  i = 0
  for train,test in kf:
    x_train, x_test, y_train, y_test = x[train],x[test],y[train],y[test]
    # print('fold %d: train_n %d, test_n %d' % (i, len(train), len(test)))
    r = runsvm(x_train,y_train,C=C,Q=Q)
    Ein = np.append(Ein, r['Ein'])
    nsv = np.append(nsv, r['n_support'])
    clf  = r['clf']
    y_pred = clf.predict(x_test) 
    Ecv = np.append(Ecv, np.sum(y_test*y_pred<0) / (1.*y_pred.size) )
    i += 1
  return({'Ecv':np.mean(Ecv), 'Ein':np.mean(Ein), 'nsv':np.mean(nsv)})

# use this for q8 as well
def q7(Cs=[0.0001,0.001,0.01,0.1,1.0], runs=100):
  (x,y) = readdata("/home/jbkim/Development/edx/cs1156x/week8/features.train")
  idx   = np.logical_or(y==1,y==5)
  x     = x[idx,:]
  y     = getbinary(y[idx], choice=1)

  wins      = [0 for i in range(len(Cs))]
  Ecv       = np.empty( (runs,len(Cs)) )
  for i in range(runs):
    print('iter: %d' % i)
    for j in range(len(Cs)):
      C = Cs[j]
      r = runsvm_cv(x,y,C=C,Q=2,folds=10)
      Ecv[i,j] = r['Ecv']
      # Ecv = np.append(Ecv, r['Ecv'])

    idx = np.argmin(Ecv[i,:])
    wins[idx] += 1 
  return({'wins':zip(Cs,wins), 'Ecv': zip(Cs,np.mean(Ecv,1).tolist())})


"""
RBF Kernel
==========
Consider the radial basis function (RBF) kernel K(x_n, x_m) = exp(-||x_n - x_m||^2).
Forcus on the 1 versus 5 classifier.

9. Which of the following values of C results in the lowest Ein?
  # C: [0.01,1,100,10**4,10**6]
  >>> h.q9()
  {'Eout': [0.023584, 0.021226, 0.018867, 0.023584, 0.023584],
   'nsv': [406, 31, 22, 19, 17],
   'Ein': [0.0038436, 0.004484, 0.003203, 0.002562, 0.000640]}

  [e] C = 10^6 

10. Which of the following values of C results in the lowest Eout?

  [c] C = 100
"""

# SVC implements several kernels:
#  linear:     (x,x')
#  polynomial: (gamma*(x,x') + coef0)^degree
#  rbf:        exp(-gamma*||x,x'||^2)
#  sigmoid:    tanh(-gamma*(x,x') + coef0)
def runsvm_rbf(x,y, C=0.01):
  clf = svm.SVC(kernel='rbf', C=C, gamma=1.)
  clf.fit(x, y) 
  yhat = clf.predict(x)
  Ein = np.sum( y*yhat < 0 ) / (1.*y.size)
  return({'Ein':Ein, 'b':clf.intercept_,
          'clf':clf, 'n_support':clf.support_vectors_.shape[0]})

# use this for q10 as well
def q9(Cs=[0.01,1,100,10**4,10**6]):
  (x,y)       = readdata("/home/jbkim/Development/edx/cs1156x/week8/features.train")
  idx = np.logical_or(y==1,y==5)
  x   = x[idx,:]
  y   = getbinary(y[idx],1)
  (xout,yout) = readdata("/home/jbkim/Development/edx/cs1156x/week8/features.test")
  idx  = np.logical_or(yout==1,yout==5)
  xout = xout[idx,:]
  yout = getbinary(yout[idx],1)
  Ein  = []
  Eout = []
  nsv  = []
  for C in Cs:
    r = runsvm_rbf(x,y, C=C)
    Ein.append(r['Ein'])
    nsv.append(r['n_support'])
    clf  = r['clf']
    yhat = clf.predict(xout)
    Eout.append( np.sum( yout*yhat < 0 ) / (1.*yout.size) )
  return({'Ein':Ein,'Eout':Eout,'nsv':nsv})
