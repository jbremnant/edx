#!/usr/bin/env python

"""
Hoeffding Inequality
Questions 1,2

The run the program in python interpreter

    sys.path.append('path/to/this/week1')
    sys.path.append('path/to/this/week2')
    import hw2 as h
    h.experiment1(100000)

1000 virtual coins, with 10 tosses for each one. 
Calculate the fraction of heads from:
    coin[1], coin[rand], coin[which min[heads]]

Run the experiment 100k times to determine the average ratio of heads. 

     v_1                 v_rand              v_min
    (0.4998239999999971, 0.5006619999999977, 0.03805699999997634) 

1. v_min is closest to:
    [b] 0.01 

2. Which coin or coins satisfy Hoeffiding's Inequality?
    Both v_1 and v_rand gives "nu" that's very close to "mu".
    [d] c_1 and c_rand  
    

"""

import numpy as np
from numpy import linalg 
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
# (0.4998239999999971, 0.5006619999999977, 0.03805699999997634) 
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
Questions 3,4

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

    WRONG!
"""


"""
Linear Regression
Questions 5,6,7

5. N = 100, use linear regression to find g and eval E_in: the fraction of
   in-sample points that got classified incorrectly. Repeat 1000 times and
   compute the avg of E_in.

6. N = 100, repeat but now compute E_out on 1000 out-of-sample data points

7. N = 10, first use lm and then use the weights as initial input into PLA.
   Check avg iterations required to converge on 1000 repeats.
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
    # getting the inverse : not sure what kind of inversion method it uses?
    mi = m.getI()
    # miX = np.dot(mi, Xt)
    # beta = np.dot(miX, y)
    beta = np.dot(np.dot(mi, Xt), y)
    return(beta.getA()[0,:])

def lm2(X,y):
    # cheating. linalg package already implements least sq regression
    return(linalg.lstsq(X,y)[0])

def geterr(X,y,w):
    n = X.shape[0]
    yhat = np.apply_along_axis(calc_class, 1, X, w)
    # true/false numerical conversion gives 1/0
    errcount = 1.0*np.sum( y*yhat < 0 )  
    return(errcount/n)

def runlm(n=100):
    wtarget = genline()
    (X,y) = genxy(n, wtarget)
    w = lm(X,y) 
    errratio = geterr(X,y,w)
    return(dict(E_in=errratio, w=w, wtarget=wtarget))

# Test: in-sample error, E_in, estimation.
# with 1000 iterations, average err E_in = 0.026879999999999977, 0.251, 0.0264
def experiment2(iter=1000):
    errsum = 0.0
    for i in range(iter):
        res = runlm(100) 
        errsum = errsum + res['E_in']
    return(errsum / iter)

# Test: out-of-sample error, E_out, estimation.
# with 1000 iterations, average err E_out = 0.03248500000000002, 0.0297, 0.0321
# E_out > E_in generally, but close
def experiment3(iter=1000):
    errsum = 0.0
    for i in range(iter):
        res = runlm(100)
        w       = res['w']
        wtarget = res['wtarget']
        # generate out-of-sample data
        (Xout,yout) = genxy(1000, wtarget)
        E_out = geterr(Xout,yout,w)
        errsum = errsum + E_out
    return(errsum / iter)

def runlmpla(n=10):
    import pla as p
    fw = genline()
    (X,y) = genxy(n, fw)
    wini = lm(X,y) 
    errratio = geterr(X,y,wini)
    # use the initial weights from regression
    res = p.pla(X[:,(1,2)], y, w=wini[1:3], b=wini[0])
    # target function values
    res['lmw'] = wini
    res['fw'] = fw[1:3]
    res['fb'] = fw[0]
    return(res)

# Test: use w from lm to run pla
# with 1000 iterations, average iter = 
def experiment4(iter=1000):
    itersum = 0.0
    for i in range(iter): 
        if(i%100==0):
            print("run %d" % i)
        res = runlmpla(10)
        itersum += res['iter']
    return(itersum/iter) 


"""
Nonlinear Transformation
Questions 8,9,10

Target function is now this:

    f(x1,x2) = sign(x1^2 + x2^2 - 0.6)

Instead of linear function of sign(x'*w)

8. Do lm without any transformation and compute avg E_in on 1000 runs
    [b]

9. Now transform the training data into:
    (1, x1, x2, x1*x2, x1^2, x2^2)
   Show the computed betas

10. calculate out-of-sample error, E_out, using the hypothesis above
""" 

def nl_target(x):
    xsum = np.sum(x[1:3]**2 - 0.6)
    return(1 if xsum>=0 else -1)

def genxynl(n,d=2):
    X = np.array([getsample(d) for i in range(n)])
    # last arg is the 2nd arg that goes into calc_class
    y = np.apply_along_axis(nl_target, 1, X)
    # now add the noise on 10% of dataset
    pos = rn.sample(range(n), n/10)
    y[pos] = -1*y[pos]
    return(X,y)

def runlmnl(n=1000,d=2):
    (X,y) = genxynl(n,d)
    w = lm2(X,y)  # should be faster, with same result
    E_in = geterr(X,y,w)
    res = dict(E_in=E_in, w=w)
    return(res)

# always gets right under 20% : 0.19201299999999974
# [a] 0.10
def experiment5(iter=1000):
    errsum = 0.0
    for i in range(iter):
        res = runlmnl(1000,2)
        errsum += res['E_in']
    return(errsum/iter)

def transformX(x): 
    return(np.array([x[0], x[1], x[2], x[1]*x[2], x[1]**2, x[2]**2]))

def runlmnl_trans(n=1000):
    (X,y) = genxynl(n)
    Xtran = np.apply_along_axis(transformX, 1, X)
    betas = lm2(Xtran, y)    
    return(betas)

# [a] g(x1, x2) = sign(−1 − 0.05x1 + 0.08x2 + 0.13x1x2 + 1.5x21 + 1.5x2)
# array([-1.12846996,  0.05227343, -0.08267919, -0.17047739,  0.66911726, 0.82513942])
#        -1         , -0.05      , -0.08      ,  0.13      ,  1.5       , 1.5
def experiment6(n=1000):
    return(runlmnl_trans(n))

def experiment7(iter=1000, n=1000):
    Eoutsum = 0.0
    for i in range(iter):
        (X,y) = genxynl(n)
        Xtran = np.apply_along_axis(transformX, 1, X)
        betas = lm2(Xtran, y)    
    
        (Xout,yout) = genxynl(n)
        Xouttran = np.apply_along_axis(transformX, 1, Xout)
        E_out = geterr(Xouttran,yout,betas)
        Eoutsum += E_out
    return(Eoutsum/iter) 
