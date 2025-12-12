function [a, b] = PQtConst (t100, tmin, f , Fs)
% Calculate the difference equation parameters. The time
% constant of the difference equation depends on the center
% frequencies.

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:11 $

N = length (f);
for (m = 0:N-1)
    t = tmin + (100 / f(m+1)) * (t100 - tmin);
    a(m+1) = exp (-1 / (Fs * t));
    b(m+1) = (1 - a(m+1));
end
