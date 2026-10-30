function [EP, Fmem] = PQadapt (Ehs, Fmem, Ver, Mod)
% Level and pattern adaptation

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:35:08 $

persistent a b Nc M1 M2 Version Model

if (~strcmp (Ver, Version) | ~strcmp (Mod, Model))
    Version = Ver;
    Model = Mod;
    if (strcmp (Model, 'FFT'))
        [Nc, fc] = PQCB (Version);
        NF = 2048;
        Nadv = NF / 2;
    else
        [Nc, fc] = PQFB;
        Nadv = 192;
    end
    Version = Ver;
    Model = Mod;
    Fs = 48000;
    Fss = Fs / Nadv;
    t100 = 0.050;
    tmin = 0.008;
    [a b] = PQtConst (t100, tmin, fc, Fss);
    [M1, M2] = PQ_M1M2 (Version, Model);
end

% Allocate memory
EP = zeros (2, Nc);
R = zeros (2, Nc);

% Smooth the excitation patterns
% Calculate the correlation terms
sn = 0;
sd = 0;
for (m = 0:Nc-1)
    Fmem.P(1,m+1) = a(m+1) * Fmem.P(1,m+1) + b(m+1) * Ehs(1,m+1);
    Fmem.P(2,m+1) = a(m+1) * Fmem.P(2,m+1) + b(m+1) * Ehs(2,m+1);
    sn = sn + sqrt (Fmem.P(2,m+1) * Fmem.P(1,m+1));
    sd = sd + Fmem.P(2,m+1);
end

% Level correlation
CL = (sn / sd)^2;

for (m = 0:Nc-1)

% Scale one of the signals to match levels
    if (CL > 1)
        EP(1,m+1) = Ehs(1,m+1) / CL;
        EP(2,m+1) = Ehs(2,m+1);
    else
        EP(1,m+1) = Ehs(1,m+1);
        EP(2,m+1) = Ehs(2,m+1) * CL;
    end

% Calculate a pattern match correction factor
    Fmem.Rn(m+1) = a(m+1) * Fmem.Rn(m+1) + EP(2,m+1) * EP(1,m+1);
    Fmem.Rd(m+1) = a(m+1) * Fmem.Rd(m+1) + EP(1,m+1) * EP(1,m+1);
    if (Fmem.Rd(m+1) <= 0 | Fmem.Rn(m+1) <= 0)
        error ('>>> PQadap: Rd or Rn is zero');
    end
    if (Fmem.Rn(m+1) >= Fmem.Rd(m+1))
        R(1,m+1) = 1;
        R(2,m+1) = Fmem.Rd(m+1) / Fmem.Rn(m+1);
    else
        R(1,m+1) = Fmem.Rn(m+1) / Fmem.Rd(m+1);
        R(2,m+1) = 1;
    end
end

% Average the correction factors over M channels and smooth with time
% Create spectrally adapted patterns
for (m = 0:Nc-1)
    iL = max (m - M1, 0);
    iU = min (m + M2, Nc-1);
    s1 = 0;
    s2 = 0;
    for (i = iL:iU)
        s1 = s1 + R(1,i+1);
        s2 = s2 + R(2,i+1);
    end
    Fmem.PC(1,m+1) = a(m+1) * Fmem.PC(1,m+1) + b(m+1) * s1 / (iU-iL+1);
    Fmem.PC(2,m+1) = a(m+1) * Fmem.PC(2,m+1) + b(m+1) * s2 / (iU-iL+1);

    % Final correction factor => spectrally adapted patterns
    EP(1,m+1) = EP(1,m+1) * Fmem.PC(1,m+1);
    EP(2,m+1) = EP(2,m+1) * Fmem.PC(2,m+1);
end

%--------------------------------------
function [M1, M2] = PQ_M1M2 (Version, Model)
% Return band averaging parameters

if (strcmp (Version, 'Basic'))
    M1 = 3;
    M2 = 4;
elseif (strcmp (Version, 'Advanced'))
    if (strcmp (Model, 'FFT'))
        M1 = 1;
        M2 = 2;
    else
        M1 = 1;
        M2 = 1;
    end
end
