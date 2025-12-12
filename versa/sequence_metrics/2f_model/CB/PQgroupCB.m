function Eb = PQgroupCB (X2, Ver)
% Group a DFT energy vector into critical bands
% X2 - Squared-magnitude vector (DFT bins)
% Eb - Excitation vector (fractional critical bands)

% P. Kabal $Revision: 1.2 $  $Date: 2004/02/05 04:25:46 $

persistent Nc kl ku Ul Uu Version

Emin = 1e-12;

if (~ strcmp (Ver, Version))
    Version = Ver;
    % Set up the DFT bin to critical band mapping
    NF = 2048;
    Fs = 48000;
    [Nc, kl, ku, Ul, Uu] = PQ_CBMapping (NF, Fs, Version);
end

% Allocate storage
Eb = zeros (1, Nc);

% Compute the excitation in each band
for (i = 0:Nc-1)
    Ea = Ul(i+1) * X2(kl(i+1)+1);       % First bin
    for (k = (kl(i+1)+1):(ku(i+1)-1))
        Ea = Ea + X2(k+1);              % Middle bins
    end
    Ea = Ea + Uu(i+1) * X2(ku(i+1)+1);  % Last bin
    Eb(i+1) = max(Ea, Emin);
end

%---------------------------------------
function [Nc, kl, ku, Ul, Uu] = PQ_CBMapping (NF, Fs, Version)

[Nc, fc, fl, fu] = PQCB (Version);

% Fill in the DFT bin to critical band mappings
df = Fs / NF;
for (i = 0:Nc-1)
    fli = fl(i+1);
    fui = fu(i+1);
    for (k = 0:NF/2)
        if ((k+0.5)*df > fli)
            kl(i+1) = k;        % First bin in band i
            Ul(i+1) = (min(fui, (k+0.5)*df) ...
                       - max(fli, (k-0.5)*df)) / df;
            break;
        end
    end
    for (k = NF/2:-1:0)
        if ((k-0.5)*df < fui)
            ku(i+1) = k;        % Last bin in band i
            if (kl(i+1) == ku(i+1))
                Uu(i+1) = 0;       % Single bin in band
            else
                Uu(i+1) = (min(fui, (k+0.5)*df) ...
                           - max(fli, (k-0.5)*df)) / df;
            end
            break;
        end
    end
end
