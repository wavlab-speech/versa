function MDiff = PQmovModDiffB (M, ERavg)
% Modulation difference related MOV precursors (Basic version)

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:46 $

persistent Nc Ete

if (isempty (Nc))
    e = 0.3;
    [Nc, fc] = PQCB ('Basic');
    Et = PQIntNoise (fc);
    for (m = 0:Nc-1)
        Ete(m+1) = Et(m+1)^e;
    end
end

% Parameters
negWt2B = 0.1;
offset1B = 1.0;
offset2B = 0.01;
levWt = 100;

s1B = 0;
s2B = 0;
Wt = 0;
for (m = 0:Nc-1)
    if (M(1,m+1) > M(2,m+1))
        num1B = M(1,m+1) - M(2,m+1);
        num2B = negWt2B * num1B;
    else
        num1B = M(2,m+1) - M(1,m+1);
        num2B = num1B;
    end
    MD1B = num1B / (offset1B + M(1,m+1));
    MD2B = num2B / (offset2B + M(1,m+1));
    s1B = s1B + MD1B;
    s2B = s2B + MD2B;
    Wt = Wt + ERavg(m+1) / (ERavg(m+1) + levWt * Ete(m+1));
end

MDiff.Mt1B = (100 / Nc) * s1B;
MDiff.Mt2B = (100 / Nc) * s2B;
MDiff.Wt = Wt;
