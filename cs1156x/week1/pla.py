#!/usr/bin/env python
"""
To test:
    sys.path.append('/Users/jbkim/Development/edx/cs1156x/week1')
    import pla as p  
    p.genline()
"""

import numpy as np
import random as rn
 
def calc_class(x,w,b):
    y = np.dot(x,w) + b
    #print(x)
    #print(w)
    return(1 if y>=0 else -1)

def getsample(d=2):
    x = np.array([rn.uniform(-1,1) for i in range(d)])
    return(x)

def genline():
    # (x1,y1), (x2,y2)
    a = getsample(2)
    b = getsample(2)
    slope = (b[1] - a[1]) / (b[0] - a[0])
    inter = a[1] - slope * a[0]
    # weight vector is orthogonal to the line
    w = np.array([b[1]-a[1], b[0]-a[0]])
    return(w,inter)

def genxy(n,w,b):
    # (w,b) = genline()
    X = np.array([getsample(2) for i in range(n)])  
    y = np.apply_along_axis(calc_class, 1, X, w, b)
    return(X,y)

def pla(X,y,w=np.array([0 for i in range(2)]),b=0):
    separated = False 
    iter = 0
    n = X.shape[0]
    while not separated:
        separated = True
        for i in range(n):
            x = X[i,:]
            y1 = y[i]
            yhat = calc_class(x,w,b)
            if yhat*y1<0:
                w = w + y1*x
                b = b + y1*np.sqrt(np.dot(x,x))
                iter = iter + 1
                separated = False
    return(dict(w=w, b=b, iter=iter))
        
def runpla(n):
    (w,b) = genline() 
    (X,y) = genxy(n, w, b) 
    res = pla(X, y, w=np.array([0,0]), b=0)         
    # target function values
    res['fw'] = w
    res['fb'] = b
    return(res)

if __name__ == '__main__':
    main( sys.argv[1:] )
