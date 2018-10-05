function T = texlabel(varargin)
% TEXLABEL Produces the TeX format from a symbolic expression.
%   TEXLABEL(f) converts the expression f into the TeX equivalent
%   for title/label application.  It processes transliterated Greek
%   variables to print (in titles/labels) as actual Greek letters.
%
%   If the title/label is too long to fit into a plot window, then 
%   the center of the expression is removed and an ellipsis ...
%   is inserted.
%
%   TEXLABEL is used in EZSURF, EZMESH, etc. to generate TeX format
%   for the title, x-, y-, and z-labels for these plots.
%
%   Examples:
%    syms x y lambda12 delta
%    texlabel(sin(sqrt(x^2 + y^2))/sqrt(x^2 + y^2))
%       returns
%    {sin}(({x}^{2} + {y}^{2})^{{1}/{2}})/({x}^{2} + {y}^{2})^{{1}/{2}}
%
%    texlabel(3*(1-x)^2*exp(-(x^2) - (y+1)^2) - 10*(x/5 - x^3 - y^5)* ...
%             exp(-x^2-y^2) - 1/3*exp(-(x+1)^2 - y^2))
%       returns
%    {3} {exp}(- ({y} + {1})^{2} - {x}^{2}) ({x} - {1})^{2} -...+ {exp}(- {x}^{2} - {y}^{2}) ({10} {x}^{3} - {2} {x} + {10} {y}^{5})
%
%    texlabel(lambda12^(3/2)/pi - pi*delta^(2/3))
%       returns
%    {\lambda_{12}}^{{3}/{2}}/{\pi} - {\delta}^{{2}/{3}} {\pi}
%
%    texlabel(lambda12^(3/2)/pi - pi*delta^(2/3),'literal')
%       returns
%    {lambda12}^{{3}/{2}}/{pi} - {delta}^{{2}/{3}} {pi}

%   Copyright 1993-2015 The MathWorks, Inc.

switch length(varargin)
case 1
   T = texlabel(char(varargin{1}));
otherwise
   varargin{2} = validatestring(varargin{2}, {'literal'});
   T = texlabel(char(varargin{1}),varargin{2});
end
