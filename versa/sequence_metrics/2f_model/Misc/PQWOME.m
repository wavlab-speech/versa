function W2 = PQWOME (f)
% Generate the weighting for the outer & middle ear filtering
% Note: The output is a magnitude-squared vector

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:11 $

N = length (f);
for (k = 0:N-1)
    fkHz = f(k+1) / 1000;
    AdB = -2.184 * fkHz^(-0.8) + 6.5 * exp(-0.6 * (fkHz - 3.3)^2) ...
          - 0.001 * fkHz^(3.6);
    W2(k+1) = 10^(AdB / 10);
end
