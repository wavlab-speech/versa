function X2 = PQDFTFrame (x)
% Calculate the DFT of a frame of data (NF values), returning the
% squared-magnitude DFT vector (NF/2 + 1 values)

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:32:57 $

persistent hw

NF = 2048;      % Frame size (samples)

if (isempty (hw))
    Amax = 32768;
    fc = 1019.5;
    Fs = 48000;
    Lp = 92;
    % Set up the window (including all gains)
    GL = PQ_GL (NF, Amax, fc/Fs, Lp);
    hw = GL * PQHannWin (NF);
end

% Window the data
xw = hw .* x;

% DFT (output is real followed by imaginary)
X = PQRFFT (xw, NF, 1);

% Squared magnitude
X2 = PQRFFTMSq (X, NF);

%----------------------------------------
function GL = PQ_GL (NF, Amax, fcN, Lp)
% Scaled Hann window, including loudness scaling

% Calculate the gain for the Hann Window
%  - level Lp (SPL) corresponds to a sine with normalized frequency
%    fcN and a peak value of Amax

W = NF - 1;
gp = PQ_gp (fcN, NF, W);
GL = 10^(Lp / 20) / (gp * Amax/4 * W);

%----------
function gp = PQ_gp (fcN, NF, W)
% Calculate the peak factor. The signal is a sinusoid windowed with
% a Hann window. The sinusoid frequency falls between DFT bins. The
% peak of the frequency response (on a continuous frequency scale) falls
% between DFT bins. The largest DFT bin value is the peak factor times
% the peak of the continuous response.
%  fcN - Normalized sinusoid frequency (0-1)
%  NF  - Frame (DFT) length samples
%  NW  - Window length samples

% Distance to the nearest DFT bin
df = 1 / NF;
k = floor (fcN / df);
dfN = min ((k+1) * df - fcN, fcN - k * df);

dfW = dfN * W;
gp = sin(pi * dfW) / (pi * dfW * (1 - dfW^2));

