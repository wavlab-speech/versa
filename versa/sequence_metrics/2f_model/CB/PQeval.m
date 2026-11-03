function [MOVI, Fmem] = PQeval (xR, xT, Fmem)
% PEAQ - Process one frame with the FFT model

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:32:58 $

NF = 2048;
Version = 'Basic';

% Windowed DFT
X2(1,:) = PQDFTFrame (xR);
X2(2,:) = PQDFTFrame (xT);

% Critical band grouping and frequency spreading
[EbN, Es] = PQ_excitCB (X2);

% Time domain smoothing => "Excitation patterns"
[Ehs(1,:), Fmem.TDS.Ef(1,:)] = PQ_timeSpread (Es(1,:), Fmem.TDS.Ef(1,:));
[Ehs(2,:), Fmem.TDS.Ef(2,:)] = PQ_timeSpread (Es(2,:), Fmem.TDS.Ef(2,:));

% Level and pattern adaptation => "Spectrally adapted patterns"
[EP, Fmem.Adap] = PQadapt (Ehs, Fmem.Adap, Version, 'FFT');

% Modulation patterns
[M, ERavg, Fmem.Env] = PQmodPatt (Es, Fmem.Env);
% Loudness
MOVI.Loud.NRef  = PQloud (Ehs(1,:), Version, 'FFT');
MOVI.Loud.NTest = PQloud (Ehs(2,:), Version, 'FFT');

% Modulation differences
MOVI.MDiff = PQmovModDiffB (M, ERavg);

% Noise Loudness
MOVI.NLoud.NL = PQmovNLoudB (M, EP);

% Bandwidth
MOVI.BW = PQmovBW (X2);

% Noise-to-mask ratios
MOVI.NMR = PQmovNMRB (EbN, Ehs(1,:));

% Probability of detection
MOVI.PD = PQmovPD (Ehs);

% Error harmonic structure
MOVI.EHS.EHS = PQmovEHS (xR, xT, X2);

%--------------------
function [EbN, Es] = PQ_excitCB (X2)

persistent W2 EIN

NF = 2048;
Version = 'Basic';
if (isempty (W2))
    Fs = 48000;
    f = linspace (0, Fs/2, NF/2+1);
    W2 = PQWOME (f);
    [Nc, fc] = PQCB (Version);
    EIN = PQIntNoise (fc);
end

% Allocate storage
XwN2 = zeros (1, NF/2+1);

% Outer and middle ear filtering
Xw2(1,:) = W2 .* X2(1,1:NF/2+1);
Xw2(2,:) = W2 .* X2(2,1:NF/2+1);

% Form the difference magnitude signal
for (k = 0:NF/2)
    XwN2(k+1) = (Xw2(1,k+1) - 2 * sqrt (Xw2(1,k+1) * Xw2(2,k+1)) ...
               + Xw2(2,k+1));
end

% Group into partial critical bands
Eb(1,:) = PQgroupCB (Xw2(1,:), Version);
Eb(2,:) = PQgroupCB (Xw2(2,:), Version);
EbN     = PQgroupCB (XwN2, Version);

% Add the internal noise term => "Pitch patterns"
E(1,:) = Eb(1,:) + EIN;
E(2,:) = Eb(2,:) + EIN;

% Critical band spreading => "Unsmeared excitation patterns"
Es(1,:) = PQspreadCB (E(1,:), Version);
Es(2,:) = PQspreadCB (E(2,:), Version);

%--------------------
function [Ehs, Ef] = PQ_timeSpread (Es, Ef)

persistent Nc a b

if (isempty (Nc))
    [Nc, fc] = PQCB ('Basic');
    Fs = 48000;
    NF = 2048;
    Nadv = NF / 2;
    Fss = Fs / Nadv;
    t100 = 0.030;
    tmin = 0.008;
    [a, b] = PQtConst (t100, tmin, fc, Fss);
end

% Allocate storage
Ehs = zeros (1, Nc);

% Time domain smoothing
for (m = 0:Nc-1)
    Ef(m+1) = a(m+1) * Ef(m+1) + b(m+1) * Es(m+1);
    Ehs(m+1) = max(Ef(m+1), Es(m+1));
end
