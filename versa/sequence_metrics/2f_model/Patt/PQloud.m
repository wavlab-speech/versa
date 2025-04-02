function Ntot = PQloud (Ehs, Ver, Mod)
% Calculate the loudness

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:35:09 $

e = 0.23;

persistent Nc s Et Ets Version Model

if (~strcmp (Ver, Version) | ~strcmp (Mod, Model))
    Version = Ver;
    Model = Mod;
    if (strcmp (Model, 'FFT'))
        [Nc, fc] = PQCB (Version);
        c = 1.07664;
    else
        [Nc, fc] = PQFB;
        c = 1.26539;
    end
    E0 = 1e4;
    Et = PQ_enThresh (fc);
    s = PQ_exIndex (fc);
    for (m = 0:Nc-1)
        Ets(m+1) = c * (Et(m+1) / (s(m+1) * E0))^e;
    end
end

sN = 0;
for (m = 0:Nc-1)
    Nm = Ets(m+1) * ((1 - s(m+1) + s(m+1) * Ehs(m+1) / Et(m+1))^e - 1);
    sN = sN + max(Nm, 0);
end
Ntot = (24 / Nc) * sN;

%====================
function s = PQ_exIndex (f)
% Excitation index

N = length (f);
for (m = 0:N-1)
    sdB = -2 - 2.05 * atan(f(m+1) / 4000) - 0.75 * atan((f(m+1) / 1600)^2);
    s(m+1) = 10^(sdB / 10);
end

%--------------------
function Et = PQ_enThresh (f)
% Excitation threshold

N = length (f);
for (m = 0:N-1)
    EtdB = 3.64 * (f(m+1) / 1000)^(-0.8);
    Et(m+1) = 10^(EtdB / 10);
end
