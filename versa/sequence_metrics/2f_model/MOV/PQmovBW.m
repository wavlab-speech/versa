function BW = PQmovBW (X2)
% Bandwidth tests

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:46 $

persistent kx kl FR FT N

if (isempty (kx))
    NF = 2048;
    Fs = 48000;
    fx = 21586;
    kx = round (fx / Fs * NF);    % 921
    fl = 8109;
    kl = round (fl / Fs * NF);    % 346
    FRdB = 10;
    FR = 10^(FRdB / 10);
    FTdB = 5;
    FT = 10^(FTdB / 10);
    N = NF / 2;     % Limit from pseudo-code
end

Xth = X2(2,kx+1);
for (k = kx+1:N-1)
    Xth = max (Xth, X2(2,k+1));
end

% BWRef and BWTest remain negative if the BW of the test signal
% does not exceed FR * Xth for kx-1 <= k <= kl+1
BW.BWRef = -1;
XthR = FR * Xth;
for (k = kx-1:-1:kl+1)
    if (X2(1,k+1) >= XthR)
        BW.BWRef = k + 1;
        break;
    end
end

BW.BWTest = -1;
XthT = FT * Xth;
for (k = BW.BWRef-1:-1:0)
    if (X2(2,k+1) >= XthT)
        BW.BWTest = k + 1;
        break;
    end
end
