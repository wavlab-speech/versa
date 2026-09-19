function Es = PQspreadCB (E, Ver)
% Spread an excitation vector (pitch pattern) - FFT model
% Both E and Es are powers

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:32:58 $

persistent Bs Version

if (~ strcmp (Ver, Version))
    Version = Ver;
    Nc = length (E);
    Bs = PQ_SpreadCB (ones(1,Nc), ones(1,Nc), Version);
end

Es = PQ_SpreadCB (E, Bs, Version);

%-------------------------
function Es = PQ_SpreadCB (E, Bs, Ver);

persistent Nc dz fc aL aUC Version

% Power law for addition of spreading
e = 0.4;

if (~ strcmp (Ver, Version))
    Version  = Ver;
    [Nc, fc, fl, fu, dz] = PQCB (Version);
end

% Allocate storage
aUCEe = zeros (1, Nc);
Ene = zeros (1, Nc);
Es = zeros (1, Nc);

% Calculate energy dependent terms
aL = 10^(-2.7 * dz);
for (m = 0:Nc-1)
    aUC = 10^((-2.4 - 23 / fc(m+1)) * dz);
    aUCE = aUC * E(m+1)^(0.2 * dz);
    gIL = (1 - aL^(m+1)) / (1 - aL);
    gIU = (1 - aUCE^(Nc-m)) / (1 - aUCE);
    En = E(m+1) / (gIL + gIU - 1);
    aUCEe(m+1) = aUCE^e;
    Ene(m+1) = En^e;
end

% Lower spreading
Es(Nc-1+1) = Ene(Nc-1+1);
aLe = aL^e;
for (m = Nc-2:-1:0)
    Es(m+1) = aLe * Es(m+1+1) + Ene(m+1);
end

% Upper spreading i > m
for (m = 0:Nc-2)
    r = Ene(m+1);
    a = aUCEe(m+1);
    for (i = m+1:Nc-1)
       r = r * a;
       Es(i+1) = Es(i+1) + r;
    end
end

for (i = 0:Nc-1)
    Es(i+1) = (Es(i+1))^(1/e) / Bs(i+1);
end
