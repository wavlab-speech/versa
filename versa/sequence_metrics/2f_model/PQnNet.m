function ODG = PQnNetB (MOV)
% Neural net to get the final ODG

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:27:44 $

persistent amin amax wx wxb wy wyb bmin bmax I J CLIPMOV
global PQopt

if (isempty (amin))
    I = length (MOV);
    if (I == 11)
        [amin, amax, wx, wxb, wy, wyb, bmin, bmax] = NNetPar ('Basic');
    else
        [amin, amax, wx, wxb, wy, wyb, bmin, bmax] = NNetPar ('Advanced');
    end
    [I, J] = size (wx);
end

sigmoid = inline ('1 / (1 + exp(-x))');

% Scale the MOV's
Nclip = 0;
MOVx = zeros (1, I);
for (i = 0:I-1)
    MOVx(i+1) = (MOV(i+1) - amin(i+1)) / (amax(i+1) - amin(i+1));
    if (~ isempty (PQopt) & PQopt.ClipMOV ~= 0)
        if (MOVx(i+1) < 0)
            MOVx(i+1) = 0;
            Nclip = Nclip + 1;
        elseif (MOVx(i+1) > 1)
            MOVx(i+1) = 1;
            Nclip = Nclip + 1;
        end
    end
end
if (Nclip > 0)
    fprintf ('>>> %d MOVs clipped\n', Nclip);
end

% Neural network
DI = wyb;
for (j = 0:J-1)
    arg = wxb(j+1);
    for (i = 0:I-1)
        arg = arg + wx(i+1,j+1) * MOVx(i+1);
    end
    DI = DI + wy(j+1) * sigmoid (arg);
end

ODG = bmin + (bmax - bmin) * sigmoid (DI);

function [amin, amax, wx, wxb, wy, wyb, bmin, bmax] = NNetPar (Version)

if (strcmp (Version, 'Basic'))
    amin = ...
        [393.916656,  361.965332,  -24.045116,    1.110661,  -0.206623, ...
           0.074318,    1.113683,    0.950345,    0.029985,   0.000101, ...
           0];
    amax = ...
        [921,         881.131226,   16.212030,  107.137772,   2.886017, ...
          13.933351,   63.257874, 1145.018555,   14.819740,   1,        ...
           1];
    wx = ...
        [ [ -0.502657,  0.436333,   1.219602 ];
          [  4.307481,  3.246017,   1.123743 ];
          [  4.984241, -2.211189,  -0.192096 ];
          [  0.051056, -1.762424,   4.331315 ];
          [  2.321580,  1.789971,  -0.754560 ];
          [ -5.303901, -3.452257, -10.814982 ];
          [  2.730991, -6.111805,   1.519223 ];
          [  0.624950, -1.331523,  -5.955151 ];
          [  3.102889,  0.871260,  -5.922878 ];
          [ -1.051468, -0.939882,  -0.142913 ];
          [ -1.804679, -0.503610,  -0.620456 ] ];
    wxb = ...
        [ -2.518254,  0.654841, -2.207228 ];
    wy = ...
        [ -3.817048,  4.107138,  4.629582 ];
    wyb = -0.307594;
    bmin = -3.98;
    bmax = 0.22;
else
    amin = ...
        [   13.298751,  0.041073,  -25.018791,  0.061560,   0.024523 ];
    amax = ...
        [ 2166.5,      13.24326,    13.46708,  10.226771,  14.224874 ];
    wx = ...
        [ [  21.211773, -39.913052, -1.382553, -14.545348,  -0.320899 ];
          [  -8.981803,  19.956049,  0.935389,  -1.686586,  -3.238586 ];
          [   1.633830,  -2.877505, -7.442935,   5.606502,  -1.783120 ]; 
          [   6.103821,  19.587435, -0.240284,   1.088213,  -0.511314 ];
          [  11.556344,   3.892028,  9.720441,  -3.287205, -11.031250 ] ];
    wxb = ...
        [   1.330890,   2.686103,    2.096598, -1.327851,   3.087055 ];
    wy = ...    
        [ -4.696996,   -3.289959,    7.004782,  6.651897,   4.009144 ];
    wyb = -1.360308;
    bmin = -3.98;
    bmax = 0.22;    
end
