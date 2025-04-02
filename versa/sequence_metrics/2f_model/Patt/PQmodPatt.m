function [M, ERavg, Fmem] = PQmodPatt (Es, Fmem)
% Modulation pattern processing

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:35:09 $

persistent Nc a b Fss

if (isempty (Nc))
    Fs = 48000;
    NF = 2048;
    Fss = Fs / (NF/2);
    [Nc, fc] = PQCB ('Basic');
    t100 = 0.050;
    t0 = 0.008;
    [a, b] = PQtConst (t100, t0, fc, Fss);
end

% Allocate memory
M = zeros (2, Nc);

e = 0.3;
for (i = 1:2)
    for (m = 0:Nc-1)
        Ee = Es(i,m+1)^e;
        Fmem.DE(i,m+1) = a(m+1) * Fmem.DE(i,m+1) ...
                       + b(m+1) * Fss * abs (Ee - Fmem.Ese(i,m+1));
        Fmem.Eavg(i,m+1) = a(m+1) * Fmem.Eavg(i,m+1) + b(m+1) * Ee;
        Fmem.Ese(i,m+1) = Ee;

        M(i,m+1) = Fmem.DE(i,m+1) / (1 + Fmem.Eavg(i,m+1)/0.3);
    end
end

ERavg = Fmem.Eavg(1,:);
