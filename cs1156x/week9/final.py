import sys
import math
import re
import random as rn
import numpy as np
from sklearn import svm
from sklearn import cross_validation
import scipy.optimize as sci

# g_trainfile)
g_trainfile = "./features.train"
g_testfile  = "./features.test"


def lm_ridge(X,y,lam=0):
  Xt = np.transpose(X)
  k = X.shape[1]
  lambdaeye = lam*np.eye(k)
  m = np.matrix(np.dot(Xt, X)+lambdaeye)
  mi = m.getI()  # what kind of inversion method is this?
  beta = np.dot(np.dot(mi, Xt), y)
  return(beta.getA()[0,:])

def transform(x,k=3):
  x1 = x[0]
  x2 = x[1]
  trans = {
    3 : "[1, x1, x2, x1**2]",
    4 : "[1, x1, x2, x1**2, x2**2]",
    5 : "[1, x1, x2, x1**2, x2**2, x1*x2]",
    6 : "[1, x1, x2, x1**2, x2**2, x1*x2, abs(x1-x2)]",
    7 : "[1, x1, x2, x1**2, x2**2, x1*x2, abs(x1-x2), abs(x1+x2)]",
  }
  formula = trans[k]
  return(eval(formula))
  # return( [1,x1,x2,x1**2,x2**2,x1*x2,abs(x1-x2),abs(x1+x2)])

def calc_class(x,w,k=3):
  z = transform(x,k)
  y = np.dot(z,w)
  return(1 if y>=0 else -1)

def geterr(X,y,w,k=3):
  n = X.shape[0]
  yhat = np.apply_along_axis(calc_class, 1, X, w,k)
  # true/false numerical conversion gives 1/0
  errcount = 1.0*np.sum( y*yhat < 0 )
  return(errcount/n)

def q2():
  (xout,yout) = readdata("/home/jbkim/git/edx/cs1156x/week6/out.dta")
  (Eva, ws) = q1()  # q1 already computed the weights for 5 models
  Eout = dict()
  for k in ws.keys():
    w = ws[k]
    Eout[k] = geterr(xout,yout,w,k)
  return(Eout)


def readdata(file=g_trainfile):
  d = np.genfromtxt(file, dtype=float)
  return(np.apply_along_axis(lambda(x): x[1:3],1,d),
np.apply_along_axis(lambda(x): x[0],1,d))

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
