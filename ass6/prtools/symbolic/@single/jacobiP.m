function Y = jacobiP(n, a, b, x)
%JACOBIP   Jacobi polynomials.
%    Y = JACOBIP(N,A,B,X) is the N-th Jacobi polynomial.

%   Copyright 2014 The MathWorks, Inc.
Y = sym.useSymForNumeric(@jacobiP, n, a, b, x); 
end
