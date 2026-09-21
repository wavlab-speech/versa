function EIN = PQIntNoise (f)
% Generate the internal noise energy vector

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:10 $

N = length (f);
for (m = 0:N-1)
    INdB = 1.456 * (f(m+1) / 1000)^(-0.8);
    EIN(m+1) = 10^(INdB / 10);
end
