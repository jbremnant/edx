% Consider the optimization problem
%   minimize   f0(x1,x2)
%   subject to 2x1 + x2 >= 1
%              x1 + 3x2 >= 1
%              x1 >= 0, x2 >= 0

f = inline('sum(x)')
% >> f([1,3,4])
l = [1;1]
A = [2 1; 1 3]
cvx_begin
  variable x(2);
  minimize( f(x) );
  subject to
    l <= A * x
    0 <= x
cvx_end

% >> x
% x =
%     0.4000
%     0.2000
