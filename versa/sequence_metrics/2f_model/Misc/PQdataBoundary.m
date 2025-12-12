function Lim = PQdataBoundary (WAV, Nchan, StartS, Ns, info)
% Search for the data boundaries in a file
% StartS - starting sample frame
% Ns     - Number of sample frames

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:10 $

PQ_L = 5;
Amax = 32768;
NBUFF = 2048;
PQ_ATHR = 200 * (Amax / 32768);

% Search from the beginning of the file
Lim(1) = -1;
is = StartS;
EndS = StartS + Ns - 1;
while (is <= EndS)
    Nf = min (EndS - is + 1, NBUFF);
    x = PQgetData (WAV, is, Nf, info);
    for (k = 0:Nchan-1)
        Lim(1) = max (Lim(1), PQ_DataStart (x(k+1,:), Nf, PQ_L, PQ_ATHR));
    end
    if (Lim(1) >= 0)
        Lim(1) = Lim(1) + is;
        break
    end
    is = is + NBUFF - (PQ_L-1);
end

% Search from the end of the file
% This loop is written as if it is going in a forward direction
%  - When the "forward" position is i, the "backward" position is
%    EndS - (i - StartS + 1) + 1
Lim(2) = -1;
is = StartS;
while (is <= EndS)
    Nf = min (EndS - is + 1, NBUFF);
    ie = is + Nf - 1;                   % Forward limits [is, ie]
    js = EndS - (ie - StartS + 1) + 1;  % Backward limits [js, js+Nf-1]
    x = PQgetData (WAV, js, Nf, info);
    for (k = 0:Nchan-1)
        Lim(2) = max (Lim(2), PQ_DataEnd (x(k+1,:), Nf, PQ_L, PQ_ATHR));
    end
    if (Lim(2) >= 0)
        Lim(2) = Lim(2) + js;
        break
    end
    is = is + NBUFF - (PQ_L-1);
end

% Sanity checks
if (~ ((Lim(1) >= 0 & Lim(2) >= 0) | (Lim(1) < 0 & Lim(2) < 0)))
    error ('>>> PQdataBoundary: limits have difference signs');
end
if (~(Lim(1) <= Lim(2)))
    error ('>>> PQdataBoundary: Lim(1) > Lim(2)');
end

if (Lim(1) < 0)
    Lim(1) = 0;
    Lim(2) = 0;
end

%----------
function ib = PQ_DataStart (x, N, L, Thr)

ib = -1;
s = 0;
M = min (N, L);
for (i = 0:M-1)
    s = s + abs (x(i+1));
end
if (s > Thr)
    ib = 0;
    return
end

for (i = 1:N-L)                                     % i is the first sample
    s = s + (abs (x(i+L-1+1)) - abs (x(i-1+1)));    % L samples apart
    if (s > Thr)
        ib = i;
        return
    end
end

%----------
function ie = PQ_DataEnd (x, N, L, Thr)

ie = -1;
s = 0;
M = min (N, L);
for (i = N-M:N-1)
    s = s + abs (x(i+1));
end
if (s > Thr)
    ie = N-1;
    return
end

for (i = N-2:-1:L-1)                                % i is the last sample
    s = s + (abs (x(i-L+1+1)) - abs (x(i+1+1)));    % L samples apart
    if (s > Thr)
        ie = i;
        return
    end
end
