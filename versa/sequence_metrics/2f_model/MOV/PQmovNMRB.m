function NMR = PQmovNMRB (EbN, Ehs)
% Noise-to-mask ratio - Basic version
% NMR(1) average NMR
% NMR(2) max NMR

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:47 $

persistent Nc gm

if (isempty (Nc))
    [Nc, fc, fl, fu, dz] = PQCB ('Basic');
    gm = PQ_MaskOffset (dz, Nc);
end

NMR.NMRmax = 0;
s = 0;
for (m = 0:Nc-1)
    NMRm = EbN(m+1) / (gm(m+1) * Ehs(m+1));
    s = s + NMRm;
    if (NMRm > NMR.NMRmax)
        NMR.NMRmax = NMRm;
    end
end
NMR.NMRavg = s / Nc;

%----------------------------------------
function gm = PQ_MaskOffset (dz, Nc)

for (m = 0:Nc-1)
    if (m <= 12 / dz)
        mdB = 3;
    else
        mdB = 0.25 * m * dz;
    end
    gm(m+1) = 10^(-mdB / 10);
end
