1,2,3. Modified Hoeffding Inequality

  P[ |E_in(g) - E_out(g)| > epsilon] <= 2*M*exp(-2*N*epsilon^2)

  If epsilon = 0.05 and probability bound 2*M*exp(-2*N*epsilon^2) at most 0.03, what is the
  least number of examples N for M=1?

  > 2*M*exp(-2*N*epsilon^2) = 0.03 
  > exp(-2*N*epsilon^2) = 0.03 / (2*M)
  > -2*N*epsilon^2 = log(0.03/(2*M))
  > N = -log(0.03/(2*M)) / (2*epsilon^2)
  
  ```R
  f1 <- function(M, ep) { -log(0.03/(2*M)) / (2*ep^2) }
  > f1(1, 0.05)
  [1] 839.941
  > f1(10, 0.05)
  [1] 1300.458
  > f1(100, 0.05)
  [1] 1760.975
  ```
For each cases, it needs more than the numbers shown.  


4. Break Point for perceptron 3D
  d = 2 has k = 4,
  d = 3 has k = ?

  3 points make up a plane. If there are two points that lie on either side of the
  plane, where they belong to class +1, and the points that are on this plane are
  -1: you cannot shatter these points.

  k = 5


5. Growth Function

   m_H(N) <= sum_{i=0}^{k-1}(N choose i) = sum[ n!/((n-i)! i!) ]
  
   Which of the following are possible formulas
  
   i)   1 + N
   ii)  1 + N + (N choose 2)
   iii) sum_{i=1}^{|sqrt(N)|}(N choose i)
   iv)  2^(N/2)
   v)   2^N


6. 

   |x|     o      o      o  
    o    | x |    o    | x |
