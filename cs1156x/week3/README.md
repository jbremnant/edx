Training vs Testing
===================

Dichotomies
-----------
Hypotheses -> Dichotomies -> set of possible hypothese reduced.

Dichotomies are hypothesis set restricted to finite set of sample points.
You samples are from infinite set of possible numbers (here, [-1,+1]).
However, the samples are finite. You don't have infinite number of them.
The samples can be thought of as having a sheet of paper with holes
in them, and you overlay it over a canvas with lots of points on it.
What you see through those hole are essentially your samples.

As such, instead of having Hypothesis set that's infinite with the infinite
input space, you instead work with dichotomies that is _finite_ with the 
finite number of samples you have.

Growth Function
---------------
It's the maximum number of dichotomies you can have.

    `m_H(N) = max  [H(x1,x2,...,xN)]


Break Point
-----------
Break point of H:
    
    If not data set of size k can be _shattered_ by H,
    then the k is the break point of H.

For 2D perceptrons, k = 4.

    m_H(3) = 2^3 = 8
    m_H(4) = 2^4 = 16  ??? can you really have 16 hypothesis?

How many lines can separate the x's from the o's?
It's not 2^4 = 16.
Turns out, it's 14. You can probably count this.

    x2
    |
    |         o
    |              x
    |
    |    x
    |          o
    --------------------- x1

Maximum Number of Dichotomies
-----------------------------
Learn the puzzle the at the end of lecture 5.

>   x1   x2   x3
>   ------------  
>    o    o    o
>    o    o    x
>    o    x    o
>    x    o    o
>   


Theory of Generalization
========================



