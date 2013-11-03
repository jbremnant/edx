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

   In transformed space, the x1, x2 are always positive number...


3. What is the smallest value among the following choices that is >= the VC dimension of 
   linear model in the transformed space?

   If you are using perceptron, what is the VC dimension of transformed space? Shouldn't it be just the
   number of dimensions?
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


Logistic Regression Gradient: 

        g_t = -(1/N) sum_{n=1}^{N} [ (y_n x_n) / ( 1 + exp(y_n w'(t) x_n) ) ]


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

7. Now, compare the performance of "coordinate descent". In each iteration,
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


