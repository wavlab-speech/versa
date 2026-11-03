function [WAV, info] = PQwavFilePar (File)
% Print a WAVE file header, pick up the file parameters

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:11 $

persistent iB

if (isempty (iB))
    iB = 0;
else
    iB = mod (iB + 1, 2);   % Only two files can be "active" at a time
end

[WAV Fs] = audioread(string(File));

info = audioinfo(string(File))
if size(WAV,2) == 1
    WAV = repmat(WAV,1,2) / 2.0;
end
info.Fname = File;
info.Nframe = size(WAV,1);
info.Nchan = size(WAV,2);
info.iB = iB;   % Buffer number

% Initialize the buffer
PQgetData (WAV, 0, 0, info);

fprintf (' WAVE file: %s\n', string(File));
if (info.Nchan == 1)
    fprintf ('   Number of samples : %d (%.4g s)\n', info.Nframe, info.Nframe / info.SampleRate);
else
    fprintf ('   Number of frames  : %d (%.4g s)\n', info.Nframe, info.Nframe / info.SampleRate);
end
fprintf ('   Sampling frequency: %g\n', info.SampleRate);
%fprintf ('   Number of channels: %d (%d-bit integer)\n', info.Nchan, Nbit);
