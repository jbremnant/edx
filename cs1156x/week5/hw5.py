import sys
import math
import csv
import random as rn
import numpy as np

"""
import sys
import hw5 as h
sys.path.append('/home/jbkim/git/edx/cs1156x/week5')
reload(h)
"""

"""
Linear Regression Error
Question 1

Consider noisy target
  y = w'x + e,  where x real number of dimension d and contains intercept term
  e ~ N(mu, sigma)

It can be expected that the Ein with respect to D is given by:

  ED[ Ein(w_lin) ] = sigma^2 ( 1 - (d+1)/N )

1. For sigma = 0.1, d = 8, which is the smaller number of N that will result in Ein greater
   than 0.008? 

   From 25 -> 100, the error crossed 0.008. As sample increases, the Ein should tend to 0.01 (0.1^2)
    >>> (0.1**2.)*(1.- (9./25.))
    0.006400000000000001
    >>> (0.1**2.)*(1.- (9./100.))
    0.009100000000000002
  
   [c] 100
"""

def q1(sigma,d,Ein):
  N = (d+1.0) / (1.0 - ( Ein / sigma**2 ))
  return(N)


"""
Nonlinear Transforms
Question 2,3

  pi(1, x1, x2) = (1, x1^2, x2^2)

2. Which of the following constraints on weights in Z space could correspond to the hyperbolic
   decision boundary in X depicted in the figure?  wbar0 can be selected to achieve desired boundary.

   The equation of the hyperbolic parabola is : X^2 - Y^2 = constant

    http://www.wolframalpha.com/input/?i=graph+of+x%5E2+-+y%5E2+%3D+2

   This implies that the   w1 > 0, w2 < 0 ???  no!
   The equation gives the shape of the DECISION BOUNDARY, not the w vector.
   Note that the vector w is the orthogonal projection of the decision boundary.
   The sign had to be flipped.

   [d] w1 < 0, w2 > 0,  NOT   [e] w1 > 0, w2 < 0

3. What is the smallest value among the following choices that is >= the VC dimension of 
   linear model in the transformed space?

   If you are using perceptron, what is the VC dimension of transformed space? Shouldn't it be just the
   number of dimensions?

   The answer is just the count of all the variables in the transformed space:

   [c] 15
"""


"""
Gradient Descent
Question 4,5,6,7

Algorithm (pg.95):

  1. Initialize the weights at the time step t = 0 to w(0).
  2. for t = 0,1,2... do
  3.   Compute the gradient 
        g_t = Delta E_in( w(t) )
  4.    Set the direction to move,  v_t = -g_t
  5.    Update the weights:  w(t+1) = w(t) + eta*v_t
  6.    Iterate to the next step until it is time to stop
  7. Return the final weights


Non-linear error surface  

  E(u,v) = (u e^v - 2v e^-u)^2

Start at (u,v) = (1,1) and minimize this error using gradient descent in the uv space.
Use eta = 0.1

4. Partial derivative of E(u,v) with respect to u: dE/du ?

  dE/du = 2(u e^v - 2v e^-u) (e^v + 2v e^-u)

  [e]

  dE/dv = 2(u e^v - 2v e^-u) (u e^v - 2e^-u)

5. How many iterations does it take to fall below 10^-14 for the first time?

  So here, we need to implement gradient descent.
  You need partial derivatives with respect to both u, v.

    >>> h.q5()['iter']
    10

  [d] 10

6. After running the iterations until the error dropped below 10^-14, what are the closest values
   to the final (u,v) in euclidean distance?

    >>> res = h.q5()
    >>> [res['u'],res['v']]
    [0.04473629039778207, 0.023958714099141746]

  [e] (0.045, 0.024)

  Euclidean distance:  math.sqrt( (myu - 0.045)**2 + (myv - 0.024)**2 )

7. Now, compare the performance of "coordinate descent". In each iteration, we have two steps
   along the 2 coordinates. Step 1 is to move only along the u coordinate to reduce the error,
   and step 2 is to reevaluate and move only along the v coordinate to reduce the error. 

    >>> r = h.q7()
    >>> r['Ein']
    0.13981379199615315

  [a] 10^-1 == 0.1

  Here, u gets updated before v, instead of simultaneous update in question 6.
  The updated u affects the gradient computation of v, and the Error is decreased
  at much slower rate in this example.
"""

def E(u,v):
  return (u*math.exp(v) - 2*v*math.exp(-u))**2

# partial derivatives
def dE_du(u,v):
  return 2 * (u*math.exp(v) - 2*v*math.exp(-u)) * (math.exp(v) + 2*v*math.exp(-u))

def dE_dv(u,v):
  return 2 * (u*math.exp(v) - 2*v*math.exp(-u)) * (u*math.exp(v) - 2*math.exp(-u))

# although you could've made gradient descent more generic, getting the idea down is more important
def q5(eta=0.1, init=[1.0,1.0], thresh=1e-14):
  u = init[0]
  v = init[1]
  Es = [];
  iter = 0
  Ein = E(u,v)
  Es.append(Ein)
  while( iter < 10000 and Ein > thresh ):
    g_u = dE_du(u,v)
    g_v = dE_dv(u,v)
    u   = u - eta * g_u
    v   = v - eta * g_v
    Ein = E(u,v)
    Es.append(Ein)
    iter += 1
  return({'Ein':Ein, 'iter':iter, 'Es':Es, 'u':u, 'v': v}) 

# two step iteration
def q7(eta=0.1, init=[1.0,1.0], thresh=1e-14):
  u = init[0]
  v = init[1]
  Es = [];
  iter = 0
  Ein = E(u,v)
  Es.append(Ein)
  while( iter < 15 ):
    # step 1
    g_u = dE_du(u,v)
    u   = u - eta * g_u
    Ein = E(u,v)
    Es.append(Ein)
    # step 2 : reevaluate using updated u
    g_v = dE_dv(u,v)
    v   = v - eta * g_v
    Ein = E(u,v)
    Es.append(Ein)
    iter += 1
  return({'Ein':Ein, 'iter':iter, 'Es':Es, 'u':u, 'v': v}) 


"""
Logistic Regression  (pg.88-98)
Question 8,9,10

Take target function 'f' to be 0/1 probability. Take two random unif distro points on [-1,1]x[-1,1]
and taking line passing through them as the boundary between y=+/- 1.
Generate N=100 samples, where X = [-1,1]x[-1,1] with uniform distro.
Take d = 2

Run Logistic Regression with Stochastic Gradient Descent to find g and estimate E_out (the cross 
entropy error) by generating sufficiently large out-of-sample data points. 
Repeate this 100 times and take the average of E_out.

* Initialize w(0) = 0
* Stop the algorithm when ||w(t-1) - w(t)|| < 0.01.
* w(t) is at the end of epoch t, which is full pass through randomized N data points.
* eta = 0.01

Formulas:

  Ein(w) = (1/N) sum_{n=1}^{N} ln(1 + exp(-y_n * w' x_n))   
  pointwise error:  e(h(x_n), y_n) = ln(1 + exp(-y_n * w' x_n))
  g_t = -(1/N) sum_{n=1}^{N} [ (y_n x_n) / ( 1 + exp(y_n w'(t) x_n) ) ]

For stochastic gradient descent, you just need to consider one sample at a time.
"""

def calc_class(x,w):
  y = np.dot(x,w)
  # return(1 if y>=0 else -1)
  return(1 if y>=0 else 0)

def getsample(d=2):
  # first number always constant
  x = np.array([1] + [rn.uniform(-1,1) for i in range(d)])
  return(x)

#    y -  m*x -    b > 0 ? 1 : 0
# w2*y - w1*x - w2*b > 0 ? 1 : 0
def genline():
  # (x1,y1), (x2,y2)
  a = getsample(2)
  b = getsample(2)
  w1 = (b[2]-a[2])  # change in y or x2
  w2 = (b[1]-a[1])  # change in x or x1
  slope = w1/w2
  inter = a[2] - slope * a[1]
  # weight vector is orthogonal to the line: orthogonal vector has negative inverse of slope
  w = np.array([inter*w2, -w1, w2])
  return(w)

def genxy(n=100, w=genline()):
  x = np.array([getsample(2) for i in range(n)])
  # last arg is the 2nd arg that goes into calc_class
  y = np.apply_along_axis(calc_class, 1, x, w)
  return(x,y)

# returns a vector of gradient per x_i
def g_logit1(y1, x1, w):
  delta_e = -(y1 * x1)/(1.0 + math.exp(y1* np.dot(w,x1)))
  return(delta_e)

def E_logit1(y1, x1, w):
  e = np.log(1.0 + math.exp(-y1 * np.dot(w,x1)))
  return(e)

def batch_logit(y, x, w, func=g_logit1):
  es = np.zeros(x.shape)
  if(func==E_logit1):
    es = np.zeros(y.shape) 
  for i in range(y.size):
    es[i] = func(y[i], x[i], w)
  return(np.mean(es, axis=0))

def diffnorm(w_prev,w):
  return( np.sqrt( np.sum((w_prev - w)**2)) )

def logit(x, y, wini=np.array([0.0,0.0,0.0]), eta=0.01):
  count = 0 
  n = y.size
  pos = rn.sample(range(n), n)
  w_prev = wini
  w = wini
  Es = []
  dn = 1.0
  while( dn >= 0.01 ):
    if(len(pos)==0):
      count += 1
      dn = diffnorm(w_prev, w)
      if count % 25 == 0 or dn < 0.01:
        print("run[%d] diffnorm: %.4f" % (count,dn)) 
      if( dn < 0.01 ): 
        # print("diffnorm of previous to current weight less than 0.01: %.4f" % dn)
        return({'w':w, 'count':count})

      w_prev = w 
      pos = rn.sample(range(n), n)

    i   = pos.pop()  
    E   = E_logit1(y[i], x[i], w)
    Es.append(E)
    # gradient descent
    g_E = g_logit1(y[i], x[i], w)
    w   = w - eta*g_E 

  return({'w':w,'Es':E, 'count':count}) 


def q8(runs=100):
  Esum = 0.0
  csum = 0.0
  for i in range(runs):
    print("TEST %d" %i)
    w = genline()  # target func
    (x,y) = genxy(n=100, w=w)
    r = logit(x,y)
    w = r['w']
    c = r['count'] 

    (xout,yout) = genxy(n=500, w=w)
    Eout = batch_logit(yout,xout,w, func=E_logit1)
    Esum += Eout
    csum += c

  Eoutavg = Esum/runs
  cavg    = csum/runs
  print(Eoutavg)
  return({'Eout':Eoutavg,'epochs':cavg})
