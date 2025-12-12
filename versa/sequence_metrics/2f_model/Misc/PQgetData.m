function x = PQgetData (WAV, i, N, info)
% Get data from internal buffer or file
% i - file position
% N - number of samples
% x - output data (scaled to the range -32768 to +32767)

% Only two files can be "active" at a time.
% N = 0 resets the buffer


% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:10 $

persistent Buff
iB = info.iB + 1;
if (N == 0)
    Buff(iB).N = 20 * 256;     % Fixed size
    Buff(iB).x = PQ_ReadWAV (WAV, i, Buff(iB).N, info);
    Buff(iB).i = i;
end

if (N > Buff(iB).N)
    error ('>>> PQgetData: Request exceeds buffer size');
end

% Check if requested data is not already in the buffer
is = i - Buff(iB).i;
if (is < 0 | is + N - 1 > Buff(iB).N - 1)
    Buff(iB).x = PQ_ReadWAV (WAV, i, Buff(iB).N, info);
    Buff(iB).i = i;
end

% Copy the data
Nchan = info.Nchan;
is = i - Buff(iB).i;
x = Buff(iB).x(1:Nchan,is+1:is+N-1+1);
%------
function x = PQ_ReadWAV (WAV, i, N, info)
% This function considers the data to extended with zeros before and
% after the data in the file. If the starting offset i is negative,
% zeros are filled in before the data starts at offset 0. If the request
% extends beyond the end of data in the file, zeros are appended.
Amax = 32768;
Nchan = info.Nchan;
x = zeros (Nchan, N);

Nz = 0;
if (i < 0)
    Nz = min (-i, N);
    i = i + Nz;
end

Ns = min (N - Nz, info.TotalSamples - i);
if (i >= 0 & Ns > 0)
    [a, Fs] = audioread(info.Filename);
    if size(a,2) == 1
        a = repmat(a,1,2) / 2.0;
    end
    x(1:Nchan,Nz+1:Nz+Ns-1+1) = Amax * transpose(a(i+1:i+Ns-1+1,:));
end
