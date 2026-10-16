function X2 = PQRFFTMSq (X, N)
% Calculate the magnitude squared frequency response from the
% DFT values corresponding to a real signal (assumes N is even)

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:11 $

X2 = zeros (1, N/2+1);

X2(0+1) = X(0+1)^2;
for (k = 1:N/2-1)
    X2(k+1) = X(k+1)^2 + X(N/2+k+1)^2;
end
X2(N/2+1) = X(N/2+1)^2;
