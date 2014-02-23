Convex Optimization Problems
============================

Read ch 4 (skim 4.7)


Topics
======


Optimization problem in standard form
-------------------------------------


Convex optimization problems
----------------------------

* proof of optimality condition (pg.139)
  - null space?
* using linear algebra and optimality condition, you can express gradient of f(x) as something else. (A'(-nu))
* equivalent convex problems

  ```
  minimize    f0(x)
  subject to  fi(x) <= 0,   i=1,...,m
              Ax = b

  is equivalent to

  minimize(over z)   f0(Fz + x0)
  subject to         fi(Fz + x0) <= 0, i=1,...,m
  
  Ax = b  <==> x = Fz + x0  for some z
  ```

  - Naive view is that you should always eliminate equality constraints. However, almost always that's not the case.
    Instead, you do unelimination. That is, you introduce more equality constraints to the problem.

* introducing slack variables for linear inequalities
  - making inequality to equality + inequality constraints. This has to do with solvers.

* linear objective is universal for convex optimization


Quasiconvex optimization
------------------------

* in quasiconvex opt, local optima is not global optima
* convert quasiconvex into equivalent convex constraint functions
  - you than iteratively find the optimal set using bisection


Linear optimization
-------------------

* linear programming has this canonical form.
  ```
  minimize    c'x + d
  subject to  Gx <= h
              Ax  = b
  ```

* look at the diet problem example
* piecewise-linear minimization

* Chebyshev center of a polyhedron
  - linear programming implies affine. If it's affine, then you'd visualize flat, hyperplanes.
    Conversely, if you have a euclidean ball, it's a round, curve. It's not flat...
  - Cauchy-Schwartz inequality:  ai'u <= ||ai||_2 ||u||_2  

* generalized-fractional programming
  - example : Von Neumann growth problem.


Quadratic optimization
----------------------

* quadratic programming has this canonical form
  ```
  minimize    (1/2) x'P x + q'x + r
  subject to  Gx <= h
              Ax = b

* least squares
  ```  
  minimize ||Ax - b||_2^2
  equivalent to

    x'(A'A)x - 2(A'b)'x + b^2
  ```
  - can add linear constraints!
  - isotonic regression

* mean variance opt

* QCQP (Quadratically constrained quadratic program
  - so far we have:
  ```
  LP   is subset of   QP   is subset of    QCQP
  ```

* Second-order cone programming (SOCP)
  ```
  minimize    f'x
  subject to  ||A_i x + bi||_2 <=  c_i' x + d_i,   i = 1,...,m 
              Fx = g
  ```
  - this is the modern one
  - more general than QCQP and LP
  - this is 1990's

* robust linear programming (pg.157)
  - there are uncertainties in the data
  - stochastic model
  - example:
  ```
  minimize     c'x
  subject to   a_i' x <= b_i for all a_i in E_i,  i = 1,...,m

  if a_i are known to lie in given ellipsoids

    a_i in E_i = {abar_i + Pi u | ||u||_2 <= 1}
                [center] + [matrix boundary of ellipsoid]

  where P_i in R^{n x n}. P_i is an ellipsoid, which you fit over variability in a_i.

    sup{ a_i' x | a_i in E_i } <= b_i

  the lefthand side can be expressed as:
  
    sup{ a_i' x | a_i in E_i } = abar_i' x + sup { u' P_i' x | ||u||_2 <= 1 }
                               = abar_i' x + ||P_i' x||_2

  Thus, the robust linear constraint can be expressed as

    abar_i' x + ||P_i' x||_2 <= b_i

  which is evidently a second-order cone constraint. Hence, the robust LP can be expressed
  as the SOCP

    minimize     c'x
    subject to   abar_i' x + ||P_i' x||_2 <= b_i,    i=1,...,m
  ```


Geometric programming
---------------------

* monimials and posynomials
  ```
  f(x) = c x1^{a1} x2^{a2} ... xn^{an}
  ```
  - monimials have a's > 0. posynomials do not.
  - you can log transform GP 
  - GP converts to convex problem by log, log transformation

* example with cantilever beam on pg.164


Generalized inequality constraints
----------------------------------


Semidefinite programming
------------------------

* SDP - the most modern


Vector optimization
-------------------

* Mixing different objectives
  - mean + variance
  - gamma represented as "irritation" factor, scalarization
  

Homework
========

[CVX Setup] (http://cvxr.com/cvx/download/)
-----------

  ```
  wget http://cvxr.com/cvx/cvx-a64.tar.gz
  tar -zxvf cvx-a64.tar.gz
  
  # in matlab
  addpath(genpath('/path/to/cvx'))
  cvx_setup
  ```
