import sys
import math
import re
import random as rn
import numpy as np
from sklearn import svm
import scipy.optimize as sci

"""
import sys
import hw7 as h
sys.path.append('/home/jbkim/git/edx/cs1156x/week7')
reload(h)
sys.path.append('/home/jbkim/git/edx/cs1156x/week2')
import hw2 as h2
"""

"""
Validation
==========

In the following problems, use the data provided in the files from previous hw6.
Use non-linear transformation:

  1 x1 x2 x1^2 x2^2 x1*x2 abs(x1-x2) abs(x1+x2)

1. Split in.dta into training (first 25 samples) and validation (last 10 samples = K).
   Train on the 25 examples only, using the validation set of 10 examples to select
   between five models that apply linear regression to phi_0 through phi_k, with k = 3,4,5,6,7.
   For which model is the classification error on the validation set smallest?

    >>> (Eva, ws) = h.q1(); Eva
    {3: 0.3, 4: 0.5, 5: 0.20, 6: 0.0, 7: 0.10}

    [d] k = 6  error is zero??

2. Evaludate the out-of-sample error using out.dta on the 5 models to see how well the
   validation set predicted the best of the 5 models. Which model has the smallest Eout?

    >>> h.q2()
    {3: 0.42, 4: 0.416, 5: 0.188, 6: 0.084, 7: 0.072}
  
    [e] k = 7

3. Reverse the role of training and validation sets; now training with the last 10 examples 
   and validating with the first 25 examples. Which model has smallest E_validation?

    >>> (Eva,ws) = h.q3(); Eva
    {3: 0.28, 4: 0.36, 5: 0.20, 6: 0.08, 7: 0.12}

    [d] k = 6  # still the same conclusion?? stupid data or stupid coder?

4. Evaluate the Eout using out.dta. Which model has the smallest Eout?

    >>> h.q4()
    {3: 0.396, 4: 0.388, 5: 0.284, 6: 0.192, 7: 0.196}

    [d] k = 6

5. What values are closest to the out-of-sample classification error obtained for 
   the model chosen in each of the above two experiments, respectively?

    q2 had  k = 7  Eout 0.072
    q4 had  k = 6  Eout 0.192

    [b] 0.1, 0.2
"""


def readdata(file="/home/jbkim/git/edx/cs1156x/week6/in.dta"):
  d = np.genfromtxt(file, dtype=float)
  return (np.apply_along_axis(lambda(x): x[0:2], 1, d), np.apply_along_axis(lambda(x): x[2], 1, d))

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

def q1():
  (x,y) = readdata("/home/jbkim/git/edx/cs1156x/week6/in.dta")  
  xin,yin = x[0:25],y[0:25] # training set:   first 25
  xva,yva = x[25:],y[25:]   # validation set: last 10 
  Eva = dict()
  ws = dict()
  for k in range(3,8):
    zin = np.apply_along_axis(transform, 1, xin, k)
    w = lm_ridge(zin,yin,lam=0)
    # print(w)
    ws[k] = w
    Eva[k] = geterr(xva,yva,w,k)
  return(Eva,ws)

def q2():
  (xout,yout) = readdata("/home/jbkim/git/edx/cs1156x/week6/out.dta")  
  (Eva, ws) = q1()  # q1 already computed the weights for 5 models
  Eout = dict()
  for k in ws.keys():
    w = ws[k]
    Eout[k] = geterr(xout,yout,w,k)
  return(Eout)

def q3():
  (x,y) = readdata("/home/jbkim/git/edx/cs1156x/week6/in.dta")
  xva,yva = x[0:25],y[0:25] # validation set:  first 25
  xin,yin = x[25:],y[25:]   # training set:    last 10
  Eva = dict()
  ws = dict()
  for k in range(3,8):
    zin = np.apply_along_axis(transform, 1, xin, k)
    w = lm_ridge(zin,yin,lam=0)
    # print(w)
    ws[k] = w
    Eva[k] = geterr(xva,yva,w,k)
  return(Eva,ws)

def q4():
  (xout,yout) = readdata("/home/jbkim/git/edx/cs1156x/week6/out.dta")  
  (Eva, ws) = q3()  # q3 has reversed training and validation dataset
  Eout = dict()
  for k in ws.keys():
    w = ws[k]
    Eout[k] = geterr(xout,yout,w,k)
  return(Eout)


"""
Estimators
==========

6. Let e1 and e2 be independent random var, uniformly distributed over [0,1]. 
   Let e = min(e1,e2). The expected values of e1,e2,e are closest to:

    >>> h.q6(n=1000000)
    (0.49999217060921713, 0.5000692639532694, 0.33325569974328545)

    [c] 0.5,0.5,0.25 ?
    [d] 0.5,0.5,0.4 ??  # closer to d... hmmm...
"""

def q6(n=1000):
  e1sum = 0.0
  e2sum = 0.0
  esum  = 0.0
  for i in range(n):
    # e1,e2 = rn.uniform(0,1), rn.uniform(0,1)
    e1    = rn.uniform(0,1)
    e2    = rn.uniform(0,1)
    e     = min(e1,e2)
    e1sum += e1; e2sum += e2; esum += e
  return(e1sum/n, e2sum/n, esum/n)


"""
Cross Validation
================

You are givren the data points: (-1,0),(p,1),(1,0), p>=0, and a choice between 
two models: constant [h_0(x) = b] and linear [h_1(x) = ax+b]. For which value
of p would the two models be tied using leave-one-out cross-validation with the
squared error measure?

           p        
           |
      o----|----o
           |
           |

  run1 =  

"""


"""
PLA vs SVM
==========

Use Scikit Learn for this problem:
http://scikit-learn.org/stable/modules/svm.html

  >>> from sklearn import svm
  >>> X = [[0,0],[1,1]]
  >>> y = [0,1]
  >>> clf=svm.SVC()
  >>> clf.fit(X,y)
  SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
    kernel='rbf', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)
  >>> clf.predict([[2.,2.]])
  array([1])
  >>> clf.support_vectors_
  array([[ 0.,  0.],
         [ 1.,  1.]])
  >>> clf.support_
  array([0, 1])
  >>> clf.n_support_
  array([1, 1])

You can probably do it using a quadratic solver as well. Look into:

  scipy.optimize.fmin_slsqp

For each run, you will create your own target function f and dataset D. Take
d = 2 and choose a random line in the plane as your target function f
(do this by taking two runif points on [-1,1] x [-1,1] and taking the line
passing through them), where one side of the line maps to +1 and the other -1.
Choose the inputs x_n of the datset as random points in X = [-1,1]x[-1,1],
and evaludate the target function on each x_n to get the corresponding output
y_n. If all data points are on the one side of the line, discard the run and
start a new run.

Start PLA with the all-zero vector and pick the misclassified point for each
PLA iteration at random. Run PLA to find the final hypothesis g_PLA and measure
the disagreement between f and g_PLA as P[f(x) != g_PLA(x)] (you can either
calculate this exactly, or approximate it by generating a sufficiently large 
separate set of points to evaluate it). Now run SVM on the same data to find
the final hypothesis g_SVM by solving:

  min_w,b  1/2 w'w
  s.t      y_n (w'x_n + b) >= 1

using quadratic programming on the primal or the dual problem. Measure the 
disagreement between f and g_SVM as P[f(x) != g_SVM(x)], and count the number
of support vectors you get in each run.


8. For N = 10, repeat the above experiment for 100 runs. How often is g_SVM better
   than g_PLA in approximating f? The percentage of time is closest to:

  1st run
  {'E_pla': 0.090643000000000001, 'svm_win': 0.56100000000000005, 'E_svm': 0.081175999999999998}
  2nd run
  {'E_pla': 0.087777600000000011, 'svm_win': 0.51300000000000001, 'E_svm': 0.087779200000000002, 'n_support': 2.841}

  [c] 60%


9. Repeat the same experiment with N = 100

  {'E_pla': 0.01465, 'svm_win': 0.60699999999999998, 'E_svm': 0.011032, 'n_support': 2.999}
  {'E_pla': 0.018764800000000002, 'svm_win': 0.75600000000000001, 'E_svm': 0.0114786, 'n_support': 3.481}

  [d] 70% 


10. For the case N = 100, which is the closest to avg number of support vectors of g_SVM?

  [b] 3

"""

def getclass(x,w):
  y = np.dot(x,w)
  return(1 if y>=0 else -1)

def getsample(d=2):
  # first number always constant
  x = np.array([1] + [rn.uniform(-1,1) for i in range(d)])
  return(x)

# weight vector is orthogonal to the line: orthogonal vector has negative inverse of slope
#    y -  m*x -    b > 0 ? 1 : 0
# w2*y - w1*x - w2*b > 0 ? 1 : 0
def genline():
  a = getsample(2)
  b = getsample(2)
  w1 = (b[2]-a[2])  # change in y or x2
  w2 = (b[1]-a[1])  # change in x or x1
  slope = w1/w2
  inter = a[2] - slope * a[1]
  w = np.array([inter*w2, -w1, w2])
  return(w)

def genxy(n=100, w=genline()):
  x = np.array([getsample(2) for i in range(n)])
  y = np.apply_along_axis(getclass, 1, x, w)
  if(abs(np.sum(y)) == n):
    print "regenerating because output all in the same class"
    return(genxy(n,w)) # do it again if it's all the same class
  else:
    return(x,y)

def platrain(x,y,w=None):
  """
  An alternative on sklearn:
  >>> from sklearn import linear_model
  >>> clflin = linear_model.Perceptron()
  >>> clflin.fit(D, y)
  >>> y_PLA=clflin.predict(D_out)
  """
  converged = False
  n = x.shape[0]
  if w==None:
    m = x.shape[1]
    w = np.zeros(m)
  i = 0
  while not converged:
    i += 1
    misclassified = list() 
    for j in range(n):
      y1   = y[j]
      x1   = x[j]
      # print(x1)
      yhat = getclass(x1,w)
      if y1 * yhat < 0:
        misclassified.append(j)
    if len(misclassified)==0: 
      converged = True
    else:
      jrand = rn.sample(misclassified, 1)[0] # random point to use for updating
      w = w + y[jrand] * x[jrand]
  return({'w':w, 'iter':i})

def platest(x,y,w):
  n = x.shape[0]
  yhat = np.apply_along_axis(getclass, 1, x, w)
  # true/false numerical conversion gives 1/0
  errcount = 1.0*np.sum( y*yhat < 0 )
  return(errcount/n)


def svmtrain(x,y,w=None):
  """
  >>> from sklearn import svm
  >>> X = [[0,0],[1,1]]
  >>> y = [0,1]
  >>> clf=svm.SVC()
  >>> clf.fit(X,y)
  SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
    kernel='rbf', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)
  >>> clf.predict([[2.,2.]])
  array([1])
  >>> clf.support_vectors_
  array([[ 0.,  0.],
         [ 1.,  1.]])
  >>> clf.support_
  array([0, 1])
  >>> clf.n_support_
  array([1, 1])
  """
  # xl = x.tolist() 
  # yl = y.tolist()
  clf = svm.SVC(kernel='linear', C=1000)
  # clf = svm.SVC(kernel='linear')
  clf.fit(x,y)
  return({'w':clf.coef_, 'b':clf.intercept_,'clf':clf, 'n_support':clf.support_vectors_.shape[0]})

# the svm object clf contains function to predict
def svmtest(x,y,clf):
  xl = x.tolist()
  yl = y.tolist()
  yhat = clf.predict(xl)
  n = len(yhat)
  errcount = 0.0
  for i in range(n):
    if(yhat[i]*yl[i] < 0):
      errcount += 1
  return(errcount/n)


def mysvmtrain(xmat,y,w=None):
  """
  Solve the problem:
    L(a) = sum_{n=1}^{N} a_n - 1/2 sum_{n=1}^{N} sum_{m=1}^{N} y_n y_m a_n a_m x_n' x_m

    min(a) 1/2 a' Q a + (-1') a
    s.t.  y'a = 0
          0 <= a <= inf

    Use the scipy.optimize function called fmin_slsqp:
      http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
  """
  xmat = xmat[:,1:3] # take intercept out because SVM qp doesn't require this
  xN = np.dot(xmat, xmat.transpose())
  yN = np.outer(y, y) # because y is just a vector
  Q = yN * xN
  func    = lambda x: 0.5 * np.dot(np.dot(x.transpose(), Q), x) - np.dot(np.ones(len(x)),x)
  eqcon   = lambda x: np.dot(y,x)
  ineqcon = lambda x: np.array(x)
  bounds  = [(0.0, 1e16) for i in range(xmat.shape[0])] 
  x0 = np.array([rn.uniform(0,1) for i in range(xmat.shape[0])])
  # alpha = sci.fmin_slsqp(func, x0, eqcons=[eqcon], bounds=bounds)
  alpha = sci.fmin_slsqp(func, x0, eqcons=[eqcon], f_ieqcons=ineqcon)

  w = np.apply_along_axis(lambda x: np.sum(alpha * y * x), 0, xmat)
  i = np.where(alpha > 0.001)[0][1] # first support vector where alpha is greater than zero
  b = 1.0/y[i] - np.dot(w, xmat[i])
  return({'alpha':alpha, 'w':w, 'b':b})


# use this function for q9, q10 as well
def q8(N=10, repeat=1000):
  E_pla = np.zeros(repeat)    
  E_svm = np.zeros(repeat)
  n_svm_sum = 0.0
  for i in range(repeat):
    print "run %d" % i
    (xin,yin) = genxy(n=N)
    respla = platrain(xin,yin)
    ressvm = svmtrain(xin[:,1:3],yin)
    (xout,yout) = genxy(n=2000)
    E_pla[i] = platest(xout,yout,respla['w'])
    E_svm[i] = svmtest(xout[:,1:3],yout,ressvm['clf'])
    n_svm_sum += ressvm['n_support']

  svmbetter = 1.0*np.sum(E_svm < E_pla)
  return({'E_pla':np.mean(E_pla),'E_svm':np.mean(E_svm),'svm_win':svmbetter/repeat,'n_support':n_svm_sum/repeat})
