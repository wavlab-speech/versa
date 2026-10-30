function PQframeMOV (i, MOVI)
% Copy instantaneous MOV values to a new structure
% The output struct MOVC is a global.
% For most MOV's, they are just copied to the output structure.
% The exception is for the probability of detection, where the
% MOV's measure the maximum frequency-by-frequecy between channels.

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:46 $

global MOVC

[Nchan,Nc] = size (MOVC.MDiff.Mt1B);

for (j = 1:Nchan)

    % Modulation differences
    MOVC.MDiff.Mt1B(j,i+1) = MOVI(j).MDiff.Mt1B;
    MOVC.MDiff.Mt2B(j,i+1) = MOVI(j).MDiff.Mt2B;
    MOVC.MDiff.Wt(j,i+1)   = MOVI(j).MDiff.Wt;

    % Noise loudness
    MOVC.NLoud.NL(j,i+1) = MOVI(j).NLoud.NL;

    % Total loudness
    MOVC.Loud.NRef(j,i+1)  = MOVI(j).Loud.NRef;
    MOVC.Loud.NTest(j,i+1) = MOVI(j).Loud.NTest;

    % Bandwidth
    MOVC.BW.BWRef(j,i+1) = MOVI(j).BW.BWRef;
    MOVC.BW.BWTest(j,i+1) = MOVI(j).BW.BWTest;

    % Noise-to-mask ratio
    MOVC.NMR.NMRavg(j,i+1) = MOVI(j).NMR.NMRavg;
    MOVC.NMR.NMRmax(j,i+1) = MOVI(j).NMR.NMRmax;

    % Error harmonic structure
    MOVC.EHS.EHS(j,i+1) = MOVI(j).EHS.EHS;
end

% Probability of detection (collapse frequency bands)
[MOVC.PD.Pc(i+1), MOVC.PD.Qc(i+1)] = PQ_ChanPD (MOVI);

%----------------------------------------
function [Pc, Qc] = PQ_ChanPD (MOVI)

Nc = length (MOVI(1).PD.p);
Nchan = length (MOVI);

Pr = 1;
Qc = 0;
if (Nchan > 1)
    for (m = 0:Nc-1)
        pbin = max (MOVI(1).PD.p(m+1), MOVI(2).PD.p(m+1));
        qbin = max (MOVI(1).PD.q(m+1), MOVI(2).PD.q(m+1));
        Pr = Pr * (1 - pbin);
        Qc = Qc + qbin;
    end
else
    for (m = 0:Nc-1)
        Pr = Pr * (1 - MOVI.PD.p(m+1));
        Qc = Qc + MOVI.PD.q(m+1);
    end
end

Pc = 1 - Pr;
