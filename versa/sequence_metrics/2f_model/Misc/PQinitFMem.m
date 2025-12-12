function Fmem = PQinitFMem (Nc, PCinit)
% Initialize the filter memories

% P. Kabal $Revision: 1.1 $  $Date: 2003/12/07 13:34:10 $

Fmem.TDS.Ef(1:2,1:Nc) = 0;
Fmem.Adap.P(1:2,1:Nc) = 0;
Fmem.Adap.Rn(1:Nc) = 0;
Fmem.Adap.Rd(1:Nc) = 0;
Fmem.Adap.PC(1:2,1:Nc) = PCinit;
Fmem.Env.Ese(1:2,1:Nc) = 0;
Fmem.Env.DE(1:2,1:Nc) = 0;
Fmem.Env.Eavg(1:2,1:Nc) = 0;
