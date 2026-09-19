function hw = PQHannWin (NF)
% Hann window

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:10 $

hw = zeros (1, NF);

for (n = 0:NF-1)
    hw(n+1) = 0.5 * (1 - cos(2 * pi * n / (NF-1)));
end
