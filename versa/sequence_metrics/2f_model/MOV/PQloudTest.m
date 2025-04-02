function Ndel = PQloudTest (Loud)
% Loudness threshold

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:46 $

[Nchan, Np] = size (Loud.NRef);

Thr = 0.1;

% Loudness threshold
Ndel = Np;
for (j = 0:Nchan-1)
    Ndel = min (Ndel, PQ_LThresh (Thr, Loud.NRef(j+1,:), Loud.NTest(j+1,:)));
end

%-----------
function it = PQ_LThresh (Thr, NRef, NTest)
% Loudness check: Look for the first time, the loudness exceeds a threshold
% for both the test and reference signals.

Np = length (NRef);

it = Np;
for (i = 0:Np-1)
    if (NRef(i+1) > Thr & NTest(i+1) > Thr)
        it = i;
        break;
    end
end
