function NL = PQmovNLoudB (M, EP)
% Noise Loudness

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:47 $

persistent Nc Et

if (isempty (Nc))
    [Nc, fc] = PQCB ('Basic');
    Et = PQIntNoise (fc);
end

% Parameters
alpha = 1.5;
TF0 = 0.15;
S0 = 0.5;
NLmin = 0;
e = 0.23;

s = 0;
for (m = 0:Nc-1)
    sref  = TF0 * M(1,m+1) + S0;
    stest = TF0 * M(2,m+1) + S0;
    beta = exp (-alpha * (EP(2,m+1) - EP(1,m+1)) / EP(1,m+1));
    a = max (stest * EP(2,m+1) - sref * EP(1,m+1), 0);
    b = Et(m+1) + sref * EP(1,m+1) * beta;
    s = s + (Et(m+1) / stest)^e * ((1 + a / b)^e - 1);
end

NL = (24 / Nc) * s;
if (NL < NLmin)
    NL = 0;
end
