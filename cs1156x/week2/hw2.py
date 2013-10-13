#!/usr/bin/env python

"""
Hoeffding Inequality
Questions 1,2

    sys.path.import('path/to/this/hw2.py')
    import hw2 as h
    h.experiment1(100000)
"""

import numpy as np
import random as rn

def flip():
    x = rn.uniform(0,1)
    return(1 if x>0.5 else 0)

def heads_ratio(n):
    nhead = np.sum( np.array([flip() for i in range(n)]) )
    return(1.0*nhead/n)

def flip_coins(ncoin,n=10):
    vs = []
    for c in range(ncoin):
        v = heads_ratio(n)    
        vs.append(v)   
    return(vs)

def choose_coins(ncoin):
    vs = flip_coins(ncoin,10)
    # v_1, v_rand, v_min
    return(vs[0], vs[rn.randint(0,ncoin-1)], reduce(min,vs))

# hw2, Hoeffding Inequality questions 1,2
def experiment1(iter):
    v1sum = 0.0
    vrandsum = 0.0
    vminsum = 0.0
    for i in range(iter):
        (v1, vrand, vmin) = choose_coins(1000)
        v1sum = v1sum + v1
        vrandsum = vrandsum + vrand
        vminsum = vminsum + vmin
    return(v1sum/iter, vrandsum/iter, vminsum/iter)


"""
Error and Noise
Question 3,4

Hypothesis h makes error with probability, mu
Noisy target function makes error with probability, (1-lambba)

3. What is the probability of error that h makes in approximating y?

             | h right            h wrong
    ---------|----------------------------------
    f right  | (1-mu)*lambda      mu*lambda
    f wrong  | (1-mu)*(1-lambda)  mu*(1-lambda)

             | h right            h wrong
    ---------|----------------------------------
    f right  | no err             err
    f wrong  | err                no err

    Answer:
        f wrong * h right + f right * h wrong
        (1-mu)*(1-lambda) + mu*lambda   ??
    [e]

4. At what value of lambda will the performance of h independent of mu?

    Answer:
    In other words, what is the situation in which h's error has no relationship
    with mu. That can only happen when lambda is zero. Thus, h will be independent
    of mu only when the target function never generates a correct answer.
    [a]
"""


"""
Linear Regression
Questions 5,6,7



"""

def calc_class(x,w):
    y = np.dot(x,w)
    return(1 if y>=0 else -1)

def getsample(d=2):
    # first number always constant
    x = np.array([1] + [rn.uniform(-1,1) for i in range(d)])
    return(x)

def genline():
    # (x1,y1), (x2,y2)
    a = getsample(2)
    b = getsample(2)
    slope = (b[2] - a[2]) / (b[1] - a[1])
    inter = a[2] - slope * a[1]
    # weight vector is orthogonal to the line
    w = np.array([inter] + [b[2]-a[2], b[1]-a[1]])
    return(w)

def genxy(n,w):
    X = np.array([getsample(2) for i in range(n)])
    # last arg is the 2nd arg that goes into calc_class
    y = np.apply_along_axis(calc_class, 1, X, w)
    return(X,y)

def lm(X,y):
    Xt = np.transpose(X) 
    m = np.matrix(np.dot(Xt, X))
    mi = m.getI()
    # miX = np.dot(mi, Xt)
    # beta = np.dot(miX, y)
    beta = np.dot(np.dot(mi, Xt), y)
    return(beta)

def runlm():
    w = genline()
    (X,y) = genxy(n, w)

