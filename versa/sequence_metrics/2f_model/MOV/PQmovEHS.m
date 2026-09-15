function EHS = PQmovEHS (xR, xT, X2)
% Calculate the EHS MOV values

% P. Kabal $Revision: 1.2 $  $Date: 2004/02/05 04:26:19 $

persistent NF Nadv NL M Hw

if (isempty (NL))    
    NF = 2048;
    Nadv = NF / 2;
    Fs = 48000;
    Fmax = 9000;
    NL = 2^(PQ_log2(NF * Fmax / Fs));
    M = NL;
    Hw = (1 / M) * sqrt(8 / 3) * PQHannWin (M);
end

EnThr = 8000;
kmax = NL + M - 1;

EnRef  = xR(Nadv+1:NF-1+1) * xR(Nadv+1:NF-1+1)';
EnTest = xT(Nadv+1:NF-1+1) * xT(Nadv+1:NF-1+1)';

% Set the return value to be negative for small energy frames
if (EnRef < EnThr & EnTest < EnThr)
    EHS = -1;
    return;
end

% Allocate storage
D = zeros (1, kmax);

% Differences of log values
for (k = 0:kmax-1)
    D(k+1) = log (X2(2,k+1) / X2(1,k+1));
end

% Correlation computation
C = PQ_Corr (D, NL, M);

% Normalize the correlations
Cn = PQ_NCorr (C, D, NL, M);
Cnm = (1 / NL) * sum (Cn(1:NL));

% Window the correlation
Cw = Hw .* (Cn - Cnm);

% DFT
cp = PQRFFT (Cw, NL, 1);

% Squared magnitude
c2 = PQRFFTMSq (cp, NL);

% Search for a peak after a valley
EHS = PQ_FindPeak (c2, NL/2+1);

%----------------------------------------
function log2 = PQ_log2 (a)

log2 = 0;
m = 1;
while (m < a)
    log2 = log2 + 1;
    m = 2 * m;
end
log2 = log2 - 1;

%----------
function C = PQ_Corr (D, NL, M)
% Correlation calculation

% Direct computation of the correlation
% for (i = 0:NL-1)
%    s = 0;
%    for (j = 0:M-1)
%        s = s + D(j+1) * D(i+j+1);
%    end
%    C(i+1) = s;
% end

% Calculate the correlation indirectly
NFFT = 2 * NL;
D0 = [D(1:M) zeros(1,NFFT-M)];
D1 = [D(1:M+NL-1) zeros(1,NFFT-(M+NL-1))];

% DFTs of the zero-padded sequences
d0 = PQRFFT (D0, NFFT, 1);
d1 = PQRFFT (D1, NFFT, 1);

% Multiply (complex) sequences
dx(0+1) = d0(0+1) * d1(0+1);
for (n = 1:NFFT/2-1)
    m = NFFT/2 + n;
    dx(n+1) = d0(n+1) * d1(n+1) + d0(m+1) * d1(m+1);
    dx(m+1) = d0(n+1) * d1(m+1) - d0(m+1) * d1(n+1);
end
dx(NFFT/2+1) = d0(NFFT/2+1) * d1(NFFT/2+1);

% Inverse DFT
Cx = PQRFFT (dx, NFFT, -1);
C = Cx(1:NL);

%----------
function Cn = PQ_NCorr (C, D, NL, M)
% Normalize the correlation

Cn = zeros (1, NL);

s0 = C(0+1);
sj = s0;
Cn(0+1) = 1;
for (i = 1:NL-1)
    sj = sj + (D(i+M-1+1)^2 - D(i-1+1)^2);
    d = s0 * sj;
    if (d <= 0)
        Cn(i+1) = 1;
    else
        Cn(i+1) = C(i+1) / sqrt (d);
    end
end

%----------
function EHS = PQ_FindPeak (c2, N)
% Search for a peak after a valley

cprev = c2(0+1);
cmax = 0;
for (n = 1:N-1)
    if (c2(n+1) > cprev)    % Rising from a valley
        if (c2(n+1) > cmax)
            cmax = c2(n+1);
        end
    end
end
EHS = cmax;
