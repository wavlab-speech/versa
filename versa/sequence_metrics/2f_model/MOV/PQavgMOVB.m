function MOV = PQavgMOVB (MOVC, Nchan, Nwup)
% Time average MOV precursors

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:46 $

Fs = 48000;
NF = 2048;
Nadv = NF / 2;
Fss = Fs / Nadv;
tdel = 0.5;
tex = 0.050;

% BandwidthRefB, BandwidthTestB
[MOV(0+1), MOV(1+1)] = PQ_avgBW (MOVC.BW);

% Total NMRB, RelDistFramesB
[MOV(2+1), MOV(10+1)] = PQ_avgNMRB (MOVC.NMR);

% WinModDiff1B, AvgModDiff1B, AvgModDiff2B
N500ms = ceil (tdel * Fss);
Ndel = max (0, N500ms - Nwup);
[MOV(3+1), MOV(6+1), MOV(7+1)] = PQ_avgModDiffB (Ndel, MOVC.MDiff);

% RmsNoiseLoudB
N50ms = ceil (tex * Fss);
Nloud = PQloudTest (MOVC.Loud);
Ndel = max (Nloud + N50ms, Ndel);
MOV(8+1) = PQ_avgNLoudB (Ndel, MOVC.NLoud);

% ADBB, MFPDB
[MOV(4+1), MOV(9+1)] = PQ_avgPD (MOVC.PD);

% EHSB
MOV(5+1) = PQ_avgEHS (MOVC.EHS);

%-----------------------------------------
function EHSB = PQ_avgEHS (EHS)

[Nchan, Np] = size (EHS.EHS);

s = 0;
for (j = 0:Nchan-1)
    s = s + PQ_LinPosAvg (EHS.EHS(j+1,:));
end
EHSB = 1000 * s / Nchan;

    
%-----------------------------------------
function [ADBB, MFPDB] = PQ_avgPD (PD)

global PQopt

c0 = 0.9;
if (isempty (PQopt))
    c1 = 1;
else
    c1 = PQopt.PDfactor;
end

N = length (PD.Pc);
Phc = 0;
Pcmax = 0;
Qsum = 0;
nd = 0;
for (i = 0:N-1)
    Phc = c0 * Phc + (1 - c0) * PD.Pc(i+1);
    Pcmax = max (Pcmax * c1, Phc);

    if (PD.Pc(i+1) > 0.5)
        nd = nd + 1;
        Qsum = Qsum + PD.Qc(i+1);
    end
end

if (nd == 0)
    ADBB = 0;
elseif (Qsum > 0)
    ADBB = log10 (Qsum / nd);
else
    ADBB = -0.5;
end

MFPDB = Pcmax;

%-----------------------------------------
function [TotalNMRB, RelDistFramesB] = PQ_avgNMRB (NMR)

[Nchan, Np] = size (NMR.NMRavg);
Thr = 10^(1.5 / 10);

s = 0;
for (j = 0:Nchan-1)
    s = s + 10 * log10 (PQ_LinAvg (NMR.NMRavg(j+1,:)));
end
TotalNMRB = s / Nchan;

s = 0;
for (j = 0:Nchan-1)
    s = s + PQ_FractThr (Thr, NMR.NMRmax(j+1,:));
end
RelDistFramesB = s / Nchan;

%-----------------------------------------
function [BandwidthRefB, BandwidthTestB] = PQ_avgBW (BW)

[Nchan, Np] = size (BW.BWRef);

sR = 0;
sT = 0;
for (j = 0:Nchan-1)
    sR = sR + PQ_LinPosAvg (BW.BWRef(j+1,:));
    sT = sT + PQ_LinPosAvg (BW.BWTest(j+1,:));
end
BandwidthRefB  = sR / Nchan;
BandwidthTestB = sT / Nchan;

%-----------------------------------------
function [WinModDiff1B, AvgModDiff1B, AvgModDiff2B] = PQ_avgModDiffB (Ndel, MDiff)

NF = 2048;
Nadv = NF / 2;
Fs = 48000;

Fss = Fs / Nadv;
tavg = 0.1;

[Nchan, Np] = size (MDiff.Mt1B);

% Sliding window average - delayed average
L = floor (tavg * Fss);     % 100 ms sliding window length
s = 0;
for (j = 0:Nchan-1)
    s = s + PQ_WinAvg (L, MDiff.Mt1B(j+1,Ndel+1:Np-1+1));
end
WinModDiff1B = s / Nchan;

% Weighted linear average - delayed average
s = 0;
for (j = 0:Nchan-1)
    s = s + PQ_WtAvg (MDiff.Mt1B(j+1,Ndel+1:Np-1+1), MDiff.Wt(j+1,Ndel+1:Np-1+1));
end
AvgModDiff1B = s / Nchan;

% Weighted linear average - delayed average
s = 0;
for (j = 0:Nchan-1)
    s = s + PQ_WtAvg (MDiff.Mt2B(j+1,Ndel+1:Np-1+1), MDiff.Wt(j+1,Ndel+1:Np-1+1));
end
AvgModDiff2B = s / Nchan;

%-----------------------------------------
function RmsNoiseLoudB = PQ_avgNLoudB (Ndel, NLoud)

[Nchan, Np] = size (NLoud.NL);

% RMS average - delayed average and loudness threshold
s = 0;
for (j = 0:Nchan-1)
    s = s + PQ_RMSAvg (NLoud.NL(j+1,Ndel+1:Np-1+1));
end
RmsNoiseLoudB = s / Nchan;

%-----------------------------------
% Average values values, omitting values which are negative
function s = PQ_LinPosAvg (x)

N = length(x);

Nv = 0;
s = 0;
for (i = 0:N-1)
    if (x(i+1) >= 0)
        s = s + x(i+1);
        Nv = Nv + 1;
    end
end

if (Nv > 0)
    s = s / Nv;
end

%----------
% Fraction of values above a threshold
function Fd = PQ_FractThr (Thr, x)

N = length (x);

Nv = 0;
for (i = 0:N-1)
    if (x(i+1) > Thr)
        Nv = Nv + 1;
    end
end

if (N > 0)
    Fd = Nv / N;
else
    Fd = 0;
end

%-----------
% Sliding window (L samples) average
function s = PQ_WinAvg (L, x)

N = length (x);

s = 0;
for (i = L-1:N-1)
    t = 0;
    for (m = 0:L-1)
        t = t + sqrt (x(i-m+1));
    end
    s = s + (t / L)^4;
end

if (N >= L)
    s = sqrt (s / (N - L + 1));
end

%----------
% Weighted average
function s = PQ_WtAvg (x, W)

N = length (x);

s = 0;
sW = 0;
for (i = 0:N-1)
    s = s + W(i+1) * x(i+1);
    sW = sW + W(i+1);
end

if (N > 0)
    s = s / sW;
end

%----------
% Linear average
function LinAvg = PQ_LinAvg (x)

N = length (x);
s = 0;
for (i = 0:N-1)
    s = s + x(i+1);
end

LinAvg = s / N;

%----------
% Square root of average of squared values
function RMSAvg = PQ_RMSAvg (x)

N = length (x);
s = 0;
for (i = 0:N-1)
    s = s + x(i+1)^2;
end

if (N > 0)
    RMSAvg = sqrt(s / N);
else
    RMSAvg = 0;
end
