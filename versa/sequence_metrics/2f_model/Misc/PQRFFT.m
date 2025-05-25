function X = PQRFFT (x, N, ifn)
% Calculate the DFT of a real N-point sequence or the inverse
% DFT corresponding to a real N-point sequence.
% ifn > 0, forward transform
%          input x(n)  - N real values
%          output X(k) - The first N/2+1 points are the real
%            parts of the transform, the next N/2-1 points
%            are the imaginary parts of the transform. However
%            the imaginary part for the first point and the
%            middle point which are known to be zero are not
%            stored.
% ifn < 0, inverse transform
%          input X(k) - The first N/2+1 points are the real
%            parts of the transform, the next N/2-1 points
%            are the imaginary parts of the transform. However
%            the imaginary part for the first point and the
%            middle point which are known to be zero are not
%            stored.
%          output x(n) - N real values

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:11 $

if (ifn > 0)
    X = fft (x, N);
    XR = real(X(0+1:N/2+1));
    XI = imag(X(1+1:N/2-1+1));
    X = [XR XI];
else
    xR = [x(0+1:N/2+1)];
    xI = [0 x(N/2+1+1:N-1+1) 0];
    x = complex ([xR xR(N/2-1+1:-1:1+1)], [xI -xI(N/2-1+1:-1:1+1)]);
    X = real (ifft (x, N));
end
