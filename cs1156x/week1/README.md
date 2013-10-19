The Learning Problem
====================

Perceptron, Hoeffding Inequality

Check out the [LIONsolver](http://lionsolver.com/LearningFromData/) that implements the hw problems.


Notes
-----

* learn the machine learning diagram shown on the book


Answers to HW problems
----------------------

1. [d]
2. [a]
3. [d]
    P(black|black_ball_picked) = P(black & bag_picked) / P(black)
    P(B|A) = ( P(A|B) P(B) ) / P(A)   or
           = P(A & B) / P(B)
           = (1*1/2) / (3/4) = (1/2) * (4/3) = 2/3

4. draw one sample
    P( |nu - mu| > epsilon ) <= 2*exp(-2 * epsilon^2 * N)

5. draw 1000 independent samples


6. [e]
>   D, five samples where X = {0,1}^3
>   y_n = f(x_n)

>   x    | y
>   0 0 0| 0
>   0 0 1| 1
>   0 1 0| 1
>   0 1 0| 0
>   1 0 0| 1

>   X outside of D
>   1 0 1| ?
>   1 1 0| ?   =>  8 possible target functions?
>   1 1 1| ?

>          target functions
>   1 0 1| 0, 1, 0, 1, 0, 1, 0, 1
>   1 1 0| 0, 0, 1, 1, 0, 0, 1, 1
>   1 1 1| 0, 0, 0, 0, 1, 1, 1, 1

>   g returns 1 for all three points = 1*3 + 4*2 + 7*1 + 1*0 = 18
>     1
>     1
>     1
>   g returns 0 for all three points = 1*3 + 4*2 + 7*1 + 1*0 = 18
>     0
>     0
>     0
>   g is XOR odd = 1, even = 0       = 1*3 + 4*2 + 7*1 + 1*0 = 18
>     0
>     0
>     1
>   g is ~XOR odd = 0, even = 1      = 1*3 + 4*2 + 7*1 + 1*0 = 18
>     1
>     1
>     0


7. X = [-1,1] x [-1,1] with uniform probability of picking each x in X


